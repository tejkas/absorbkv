# SPDX-License-Identifier: Apache-2.0
"""Regenerate every reported number from the captures.

Each function here produces one result that appears in the write-up, and
writes it to ``results/<name>.json`` so a reader can check a claim against
the code that made it. Run::

    PYTHONPATH=src python -m absorbkv.report --captures captures --out results

Results fall into three groups:

* **validation** -- the identity residual and the noise floor, which say
  whether anything else can be believed;
* **staleness** -- rho, which signal ranks like the exact one, and whether
  neighbouring layers agree;
* **relevance** -- how concentrated the absorbed query is, how much
  attention the selected tokens capture, and the decode-traffic accounting.

Deliberately excluded, having been measured and discarded: simulations on
isotropic deviations, distant-layer check prediction (which tests a claim
CacheBlend does not make), and `W^UK` singular spectra (rho explains the
same failures more directly).
"""

# Standard
import argparse
import glob
import json
import math
from pathlib import Path

# Third Party
import torch

# First Party
from absorbkv.concentration import absorbed_query, group_mean_abs, topk_mass
from absorbkv.scores import captured_mass, exact_scores, rotate_query
from absorbkv.selection import Budget, select_latent
from absorbkv.signals import (
    build_gram_matrix,
    compute_latent_signals,
    extract_w_uk,
    quadratic_form,
)

HEADS = 16
QK_NOPE = 128
V_HEAD = 128
TOP_FRACTION = 0.15
FRACTIONS = (0.04, 0.0625, 0.125, 0.25, 0.5)
BUDGETS = (16, 64, 256)


def median(values: list[float]) -> float:
    return float(torch.tensor(values).median())


def documents(root: str, kind: str, limit: int):
    for path in sorted(glob.glob(f"{root}/{kind}/*.pt"))[:limit]:
        yield path, torch.load(path, weights_only=True, mmap=True)


def grams(weights: torch.Tensor) -> tuple[list, list, dict]:
    """Per-layer ``M_K`` and ``M_KV``, plus ``W^UK`` for the query side."""
    w_uk = {
        i: extract_w_uk(weights[i].float(), HEADS, QK_NOPE, V_HEAD)
        for i in range(weights.shape[0])
    }
    return (
        [build_gram_matrix(w_uk[i]) for i in range(weights.shape[0])],
        [build_gram_matrix(weights[i].float()) for i in range(weights.shape[0])],
        w_uk,
    )


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra, rb = a.argsort().argsort().float(), b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / (ra.norm() * rb.norm()).clamp_min(1e-12))


def jaccard(a: torch.Tensor, b: torch.Tensor, fraction=TOP_FRACTION) -> float:
    k = max(1, int(len(a) * fraction))
    first, second = set(a.topk(k).indices.tolist()), set(b.topk(k).indices.tolist())
    return len(first & second) / len(first | second)


def deviations(document: dict, per_chunk: int, layer: int):
    chunked = document["chunked"][per_chunk]["capture"]
    return (
        document["truth"]["latent"][layer].float() - chunked["latent"][layer].float(),
        document["truth"]["k_rope"][layer].float() - chunked["k_rope"][layer].float(),
    )


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def validation(root: str, weights: torch.Tensor, limit: int) -> dict:
    """The identity on real captures, the noise floor, and layer 0.

    The residual rebuilds per-head key deviations from the same latents, so
    it checks arithmetic in bf16 at scale, not the capture -- that was
    checked against the model's own keys in Phase 1.1. The noise floor is
    the first chunk, whose tokens see nothing either way and so deviate
    only by floating point. Layer 0 is reported because its latent comes
    from context-independent embeddings and therefore cannot go stale.
    """
    gram_k, gram_kv, w_uk = grams(weights)
    residuals, floor, stale, layer_zero = [], [], [], []
    for _, document in documents(root, "mla", limit):
        width = int(document["chunked"][1]["chunk_starts"][1])
        for layer in (1, 13, 26):
            delta_c, delta_rope = deviations(document, 1, layer)
            signals = compute_latent_signals(
                delta_c, delta_rope, gram_k[layer], gram_kv[layer], HEADS
            )
            per_head = w_uk[layer].view(HEADS, QK_NOPE, -1)
            rebuilt = torch.cat(
                [
                    torch.einsum("hdr,tr->thd", per_head, delta_c),
                    delta_rope.unsqueeze(1).expand(-1, HEADS, -1),
                ],
                dim=-1,
            )
            truth = (rebuilt**2).sum(dim=(1, 2))
            residuals.append(
                float((signals.exact - truth).abs().max() / truth.abs().max())
            )
            norms = delta_c.pow(2).sum(-1).sqrt()
            floor.append(float(norms[:width].median()))
            stale.append(float(norms[width:].median()))
        delta_c, _ = deviations(document, 1, 0)
        layer_zero.append(float(delta_c.pow(2).sum(-1).sqrt().median()))

    return {
        "identity_residual_max": max(residuals),
        "noise_floor_first_chunk": median(floor),
        "stale_tokens": median(stale),
        "separation": median(stale) / median(floor),
        "layer_0_deviation": median(layer_zero),
        "documents": limit,
    }


# --------------------------------------------------------------------------
# staleness
# --------------------------------------------------------------------------


def rho(root: str, weights: torch.Tensor, limit: int) -> dict:
    """``H * ||dk_rope||^2 / (dc^T M_K dc)``, per layer.

    How much of a token's staleness comes from the RoPE component every
    head shares, against the content each head projects for itself.
    Reported with percentiles because the *spread* is what reorders
    tokens: a constant rho would rescale every score alike and leave the
    ranking, which is all that is used, untouched.
    """
    gram_k, _, _ = grams(weights)
    by_layer: dict[int, list] = {}
    for _, document in documents(root, "mla", limit):
        for layer in range(1, weights.shape[0]):
            delta_c, delta_rope = deviations(document, 1, layer)
            content = quadratic_form(delta_c, gram_k[layer])
            live = content > 0
            ratio = HEADS * (delta_rope**2).sum(-1)[live] / content[live]
            by_layer.setdefault(layer, []).append(ratio)

    out = {}
    for layer, chunks in by_layer.items():
        values = torch.cat(chunks)
        p10, p50, p90 = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9]))
        out[str(layer)] = {
            "p10": float(p10),
            "median": float(p50),
            "p90": float(p90),
            "rope_share": float(p50 / (1 + p50)),
        }
    return {"by_layer": out, "documents": limit}


def signal_agreement(root: str, weights: torch.Tensor, limit: int) -> dict:
    """Do the cheap signals rank tokens like ``S_exact``?

    Jaccard over the top 15% is the set CacheBlend would recompute;
    Spearman is the whole ordering. Reported for both chunkings, since a
    result that only holds at one cache granularity is not a result.
    """
    gram_k, gram_kv, _ = grams(weights)
    overall: dict[tuple, list] = {}
    by_layer: dict[tuple, list] = {}
    for _, document in documents(root, "mla", limit):
        for per_chunk in document["chunked"]:
            for layer in range(1, weights.shape[0]):
                delta_c, delta_rope = deviations(document, per_chunk, layer)
                signals = compute_latent_signals(
                    delta_c, delta_rope, gram_k[layer], gram_kv[layer], HEADS
                )
                for name, candidate in (
                    ("kv", signals.kv),
                    ("row", signals.row),
                    ("naive", signals.naive),
                ):
                    pair = (
                        jaccard(candidate, signals.exact),
                        spearman(candidate, signals.exact),
                    )
                    overall.setdefault((per_chunk, name), []).append(pair)
                    if per_chunk == 1:
                        by_layer.setdefault((layer, name), []).append(pair[0])

    return {
        "overall": {
            f"chunk{per_chunk}_{name}": {
                "jaccard": median([j for j, _ in rows]),
                "spearman": median([s for _, s in rows]),
            }
            for (per_chunk, name), rows in overall.items()
        },
        "jaccard_by_layer": {
            f"{layer}_{name}": median(rows)
            for (layer, name), rows in by_layer.items()
        },
        "documents": limit,
    }


def consecutive_layers(root: str, weights: torch.Tensor, limit: int) -> dict:
    """CacheBlend's Insight 2: the worst tokens on one layer are the worst
    on the next. Their Figure 8 shows this for GQA and MHA models; this is
    the same measurement on MLA."""
    gram_k, gram_kv, _ = grams(weights)
    pairs: dict[int, list] = {}
    for _, document in documents(root, "mla", limit):
        exact = {}
        for layer in range(1, weights.shape[0]):
            delta_c, delta_rope = deviations(document, 1, layer)
            exact[layer] = compute_latent_signals(
                delta_c, delta_rope, gram_k[layer], gram_kv[layer], HEADS
            ).exact
        for layer in range(1, weights.shape[0] - 1):
            pairs.setdefault(layer, []).append(
                (
                    spearman(exact[layer], exact[layer + 1]),
                    jaccard(exact[layer], exact[layer + 1]),
                )
            )

    flat = [row for rows in pairs.values() for row in rows]
    return {
        "by_pair": {
            f"{layer}->{layer + 1}": {
                "spearman": median([s for s, _ in rows]),
                "jaccard": median([j for _, j in rows]),
            }
            for layer, rows in pairs.items()
        },
        "median_spearman": median([s for s, _ in flat]),
        "median_jaccard": median([j for _, j in flat]),
        "documents": limit,
    }


def chunk_size_effect(root: str, weights: torch.Tensor, limit: int) -> dict:
    """Does a token cached with more of its context around it deviate less?"""
    gram_k, gram_kv, _ = grams(weights)
    out: dict[int, list] = {}
    for _, document in documents(root, "mla", limit):
        for per_chunk in document["chunked"]:
            width = int(document["chunked"][per_chunk]["chunk_starts"][1])
            for layer in range(1, weights.shape[0]):
                delta_c, delta_rope = deviations(document, per_chunk, layer)
                signals = compute_latent_signals(
                    delta_c, delta_rope, gram_k[layer], gram_kv[layer], HEADS
                )
                out.setdefault(per_chunk, []).append(
                    float(signals.exact[width:].median())
                )
    return {
        "median_exact_beyond_first_chunk": {
            str(per_chunk): median(rows) for per_chunk, rows in out.items()
        },
        "documents": limit,
    }


# --------------------------------------------------------------------------
# relevance
# --------------------------------------------------------------------------


def concentration(root: str, weights: torch.Tensor, limit: int) -> dict:
    """Top-k absolute mass of the query, before and after absorption.

    Absolute rather than squared, because their selection is ``topk(|q|)``
    and their rescaling is a ratio of absolute sums. Reported per head and
    averaged over the heads sharing one coordinate set -- on MLA that is
    every head, which is the stronger assumption.
    """
    _, _, w_uk = grams(weights)
    rows: dict[str, list] = {}
    for _, document in documents(root, "mla", limit):
        query = document["queries"]["plain"]["query"]
        for layer in range(1, weights.shape[0]):
            nope = query[layer][..., :QK_NOPE].float()
            absorbed = absorbed_query(nope, w_uk[layer])
            tokens, heads, _ = nope.shape
            rows.setdefault("q_per_head", []).append(
                topk_mass(nope.abs().reshape(tokens * heads, -1), FRACTIONS).mass
            )
            rows.setdefault("q_group", []).append(
                topk_mass(group_mean_abs(nope, 1), FRACTIONS).mass
            )
            rows.setdefault("absorbed_per_head", []).append(
                topk_mass(
                    absorbed.abs().reshape(tokens * heads, -1), FRACTIONS
                ).mass
            )
            rows.setdefault("absorbed_group", []).append(
                topk_mass(group_mean_abs(absorbed, 1), FRACTIONS).mass
            )
    return {
        "fractions": list(FRACTIONS),
        "mass": {
            name: torch.stack(stack).median(0).values.tolist()
            for name, stack in rows.items()
        },
        "documents": limit,
    }


def fidelity(root: str, weights: torch.Tensor, config, limit: int) -> dict:
    """Attention mass captured by the selected tokens, against three
    reference points.

    ``ideal`` is the top ``k`` by true attention, the ceiling for any
    selection of that size. ``all_coords`` switches off the head-dimension
    reduction, so the gap to ``full`` is what that step costs. ``random``
    is the floor. Budgets come from their §3.6 derivation, not from the
    raw compression ratio.
    """
    _, _, w_uk = grams(weights)
    # Every layer, layer 0 included. The staleness side excludes it because
    # its latent comes from context-independent embeddings and cannot go
    # stale; relevance has no such exemption, and the baseline covers all
    # of its layers, so excluding one here would compare unmatched sets.
    layers = range(weights.shape[0])
    out: dict[tuple, list] = {}
    for _, document in documents(root, "mla", limit):
        query = document["queries"]["plain"]["query"]
        positions = document["queries"]["plain"]["position_ids"]
        for layer in layers:
            last = query[layer][-1:].float()
            scored = exact_scores(
                last,
                positions[-1:],
                document["truth"]["latent"][layer],
                document["truth"]["k_rope"][layer],
                w_uk[layer],
                config,
            )
            tokens = scored.weights.shape[-1]
            content = absorbed_query(last[..., :QK_NOPE], w_uk[layer])[0]
            rope = rotate_query(last[..., QK_NOPE:], positions[-1:], config)[0]
            latent = document["truth"]["latent"][layer].float()
            k_rope = document["truth"]["k_rope"][layer].float()
            for overall in BUDGETS:
                budget = Budget.of(overall)
                # Their §3.6 budget covers estimation *and* attention and is
                # split evenly, so step 3 attends t/2 = S/(2c) tokens, not
                # S/c. Using the full budget here would measure a setting
                # twice as generous as anything they run.
                keep = max(1, tokens // (2 * overall))
                picks = {
                    "ideal": scored.weights.mean(dim=(0, 1)).topk(keep).indices,
                    "all_coords": select_latent(
                        content, rope, latent, k_rope, keep,
                        budget.sparse_ratio, False,
                    ),
                    "full": select_latent(
                        content, rope, latent, k_rope, keep,
                        budget.sparse_ratio, True,
                    ),
                    "random": torch.randperm(tokens)[:keep],
                }
                for name, chosen in picks.items():
                    out.setdefault((overall, layer, name), []).append(
                        float(captured_mass(scored.weights, chosen).mean())
                    )

    by_budget = {
        f"c{overall}_L{layer}_{name}": median(rows)
        for (overall, layer, name), rows in out.items()
    }
    return {
        "summary": fidelity_summary(by_budget, layers),
        "by_budget": by_budget,
        "budgets": {
            str(overall): {
                "page_size": Budget.of(overall).page_size,
                "kept_fraction": Budget.of(overall).kept_fraction,
                "eviction_ratio": Budget.of(overall).eviction_ratio,
            }
            for overall in BUDGETS
        },
        "documents": limit,
    }


def gqa_fidelity(root: str, limit: int) -> dict:
    """The same measurement on Mistral, using *their* selection code.

    Without this, MLA's fidelity has no reference: a number like 0.86 of
    the oracle is only interpretable against what the method achieves on
    the architecture it was built for. Their code is imported from
    ``papers/rocketkv/code/`` rather than reimplemented, so a difference
    here is a difference in the model, not in the implementation.

    Queries are captured before rotation and the cache stores keys after
    it, so the query is rotated with Mistral's own rotary module -- the
    same trap as on the MLA side.

    ``ideal`` is one top ``k`` set per KV group, chosen on the group's mean
    attention -- the same shape of oracle ``fidelity()`` uses for MLA, where
    the one set is shared by all heads. Each side is scored against the
    best choice a selector of its own shape could have made.
    """
    # Standard
    import sys

    # Third Party
    from transformers import AutoConfig
    from transformers.models.mistral.modeling_mistral import (
        MistralRotaryEmbedding,
        apply_rotary_pos_emb,
    )

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "papers/rocketkv/code")
    )
    # Third Party
    from hybrid_sparse_attention import select_tokens

    manifest = json.loads(Path(root, "gqa/manifest.json").read_text())
    config = AutoConfig.from_pretrained(manifest["model"])
    heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    group = heads // kv_heads
    dim = config.hidden_size // heads
    rotary = MistralRotaryEmbedding(config)

    out: dict[tuple, list] = {}
    for _, document in documents(root, "gqa", limit):
        query = document["queries"]["plain"]["query"]
        positions = document["queries"]["plain"]["position_ids"]
        for layer in range(query.shape[0]):
            last = query[layer][-1:].float()
            cos, sin = rotary(last, positions[-1:].unsqueeze(0))
            shaped = last.permute(1, 0, 2).unsqueeze(0)
            rotated, _ = apply_rotary_pos_emb(shaped, shaped, cos, sin)
            per_head = rotated[0, :, 0, :]
            keys = document["keys"][layer].float().permute(1, 0, 2)
            tokens = keys.shape[1]

            logits = torch.stack(
                [per_head[h] @ keys[h // group].T for h in range(heads)]
            )
            weights = (logits / math.sqrt(dim)).softmax(-1)

            packed_q = per_head.view(1, kv_heads, group, 1, dim)
            packed_k = keys.view(1, kv_heads, 1, tokens, dim)
            for overall in BUDGETS:
                budget = Budget.of(overall)
                keep = max(1, tokens // (2 * overall))   # t/2, see fidelity()
                chosen = {
                    "full": select_tokens(
                        packed_q, packed_k, keep, budget.sparse_ratio, True
                    )[0, :, 0, 0],
                    "all_coords": select_tokens(
                        packed_q, packed_k, keep, budget.sparse_ratio, False
                    )[0, :, 0, 0],
                }
                # One top-k set per KV group, chosen on the group's mean
                # attention: the ceiling a per-group selector can reach.
                # Per-head top-k is unreachable by construction and scored
                # Mistral 0.02-0.03 low (LOG.md, 2026-10-18).
                grouped = weights.view(kv_heads, group, tokens)
                best = [
                    grouped[g].mean(0).topk(keep).indices for g in range(kv_heads)
                ]
                masses = {
                    "ideal": torch.stack(
                        [grouped[g][:, best[g]].sum(-1) for g in range(kv_heads)]
                    ).flatten(),
                    "random": weights[:, torch.randperm(tokens)[:keep]].sum(-1),
                }
                for name, picked in chosen.items():
                    masses[name] = torch.stack(
                        [
                            weights[h][picked[h // group]].sum()
                            for h in range(heads)
                        ]
                    )
                for name, mass in masses.items():
                    out.setdefault((overall, layer, name), []).append(
                        float(mass.mean())
                    )

    by_budget = {
        f"c{overall}_L{layer}_{name}": median(rows)
        for (overall, layer, name), rows in out.items()
    }
    return {
        "summary": fidelity_summary(by_budget, range(query.shape[0])),
        "by_budget": by_budget,
        "model": manifest["model"],
        "layers": query.shape[0],
        "documents": limit,
    }


def fidelity_summary(by_budget: dict, layers: range) -> dict:
    """Turn per-layer measurements into the numbers that get quoted.

    ``share_of_ideal`` is what the selection captures against the best any
    selection of that size could -- the headline. ``reduction_cost`` is
    ``full`` minus ``all_coords``, i.e. what the head-dimension reduction
    costs; the project's hypothesis predicted this would be strongly
    negative on MLA.

    The range is reported beside the median because layer-to-layer spread
    is wide enough to swamp differences between models: a median alone
    invites a comparison the data cannot support.
    """
    out = {}
    for overall in BUDGETS:
        ratios, costs = [], []
        for layer in layers:
            key = f"c{overall}_L{layer}_"
            ideal = by_budget.get(key + "ideal")
            if ideal is None:
                continue
            ratios.append(by_budget[key + "full"] / ideal)
            costs.append(by_budget[key + "full"] - by_budget[key + "all_coords"])
        out[str(overall)] = {
            "share_of_ideal_median": median(ratios),
            "share_of_ideal_min": min(ratios),
            "share_of_ideal_max": max(ratios),
            "reduction_cost_median": median(costs),
            "layers": len(ratios),
        }
    return out


def accounting() -> dict:
    """Decode traffic under *their* budget convention, in units of ``t``.

    Paper §3.6: the token budget ``t = S/c`` is defined to cover **both**
    steps -- unlike Quest, where it counts only the attention -- and they
    "evenly split the token budget between these two steps". So step 2
    (estimating which tokens matter) is meant to cost ``t/2``, and step 3
    attends ``t/2`` tokens.

    Step-2 traffic is ``(S / c**r / page) * summary_width`` values. With
    ``kept = page / c**(1-r)`` the page terms cancel and it comes to
    ``(S/c) * summary_width``, i.e. ``t * summary_width / stored``. On GQA
    the summary covers keys only, half of what a token stores, so this is
    exactly ``t/2`` -- their derivation is built around that identity.

    On MLA the value is derived from the latent being scored against, so
    the summary covers essentially the whole stored token and the same
    settings cost ``~t`` for step 2 alone. The budget is not merely split
    badly; it is exceeded. Reported as ``delivered_ratio``: what the user
    actually gets when they ask for ``c``.

    Pure arithmetic from their formulas; no captures involved.
    """
    out = {}
    for name, stored, summary in (
        ("mla", 576, lambda kept: kept * 512 + 64),
        ("mla_truncate_rope", 576, lambda kept: kept * 576),
        ("gqa_mistral", 2048, lambda kept: kept * 1024),
    ):
        for overall in BUDGETS:
            budget = Budget.of(overall)
            estimating = (
                summary(budget.kept_fraction)
                * overall
                / (budget.eviction_ratio * budget.page_size * stored)
            )
            attending = 0.5
            total = estimating + attending
            out[f"{name}_c{overall}"] = {
                "estimating_in_t": estimating,
                "attending_in_t": attending,
                "total_in_t": total,
                "delivered_ratio": overall / total,
                "requested_ratio": overall,
                "kept_fraction": budget.kept_fraction,
                "kept_fraction_to_meet_budget": _kept_for_half(
                    overall, stored, summary
                ),
            }
    return out


def _kept_for_half(overall: float, stored: int, summary) -> float:
    """The kept fraction that would put step 2 back on its ``t/2`` target."""
    budget = Budget.of(overall)
    allowed = (
        0.5 * stored * budget.eviction_ratio * budget.page_size / overall
    )
    fixed = summary(0.0)
    per_unit = summary(1.0) - fixed
    return max(0.0, (allowed - fixed) / per_unit)


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", default="captures")
    parser.add_argument("--weights", default="weights/kv_b_proj.pt")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--documents", type=int, default=60)
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Third Party
    from transformers import AutoConfig

    weights = torch.load(args.weights, weights_only=True)["kv_b_proj"]
    manifest = json.loads(Path(args.captures, "mla/manifest.json").read_text())
    config = AutoConfig.from_pretrained(manifest["model"])

    jobs = {
        "validation": lambda: validation(args.captures, weights, args.documents),
        "rho": lambda: rho(args.captures, weights, args.documents),
        "signal_agreement": lambda: signal_agreement(
            args.captures, weights, args.documents
        ),
        "consecutive_layers": lambda: consecutive_layers(
            args.captures, weights, args.documents
        ),
        "chunk_size": lambda: chunk_size_effect(
            args.captures, weights, args.documents
        ),
        "concentration": lambda: concentration(
            args.captures, weights, args.documents
        ),
        "accounting": accounting,
        "gqa_fidelity": lambda: gqa_fidelity(args.captures, args.documents),
    }
    jobs["fidelity"] = lambda: fidelity(
        args.captures, weights, config, args.documents
    )

    for name, job in jobs.items():
        if args.only and name != args.only:
            continue
        print(f"{name} ...", flush=True)
        (args.out / f"{name}.json").write_text(json.dumps(job(), indent=2))
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
