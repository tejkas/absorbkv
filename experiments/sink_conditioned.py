"""Is `k^R`'s lead retrieval, or does it mostly find the attention sink?

The first `keep` context tokens alone capture 0.632 of ideal attention
mass at c=64 (`recency.json`), so part of any selector's score is the
sink. The control gives that part away for free: every selector keeps the
first `m` context tokens and picks the remaining `keep - m` from
positions `>= m`, scored against the usual ideal -- the top `keep` tokens
by true attention over the *whole* context. If `k^R`'s lead over the
latent survives, the claim is about retrieval. If it collapses, `k^R`
mostly finds the sink.

`m` must stay below `keep` or there is no budget left to select with, and
`keep = tokens // (2c)` is 18 at c=64 and 4 at c=256, which is why the
sweep stops where it does. `m = 0` reproduces the unconditioned
measurement, so each row is comparable to `sharpness_matched.json`.

Two things to read carefully:

  prefix_only : what the free part scores on its own. Every selector
      starts from this, so the gaps between variants compress as `m`
      grows and absolute numbers rise for reasons that have nothing to do
      with selection quality.
  random : the free prefix plus `keep - m` uniform picks. This, not zero,
      is the floor a conditioned number must be judged against.

Restricting candidates to positions `>= m` is done by slicing the cache,
so page boundaries shift by `m`. At page size 2 that changes which tokens
are summarized together; the effect is small but it is not nothing, and
it applies to every variant alike.

Also reported, independent of the conditioning: `sink_overlap`, the share
of each unconditioned selector's picks that land in the first `keep`
positions, next to the same share for the true top `keep`. That says
directly how much of what `k^R` picks is the sink.
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from _mirror import page_scores

from absorbkv.concentration import absorbed_query
from absorbkv.report import QK_NOPE, documents, grams, median, spearman  # noqa: F401
from absorbkv.scores import captured_mass, rotate_query, softmax_scale
from absorbkv.selection import Budget, select_latent

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 60
BUDGETS = (16, 64, 256)
PREFIXES = (0, 1, 2, 4, 9)
SEED = 0
VARIANTS = {
    "theirs": "mixed",
    "latent_exact": "latent",
    "rope_page2": "rope",
    "rope_page1": "rope",
}
RATIOS = {"theirs": None, "latent_exact": 1.0, "rope_page2": 4.0, "rope_page1": 1.0}
REDUCE = {"theirs": True, "latent_exact": False, "rope_page2": False,
          "rope_page1": False}


def select(kind, content_q, rope_q, latent, k_rope, name, keep, target):
    """One variant at the sharpness-matched divisor, over whatever is passed."""
    zq = torch.zeros(content_q.shape[0], 0)
    zk = torch.zeros(latent.shape[0], 0)
    if kind == "mixed":
        qc, qr, lat, kr = content_q, rope_q, latent, k_rope
    elif kind == "latent":
        qc, qr, lat, kr = content_q, zq, latent, zk
    else:
        qc, qr, lat, kr = rope_q, zq, k_rope, zk
    ratio, reduce = RATIOS[name], REDUCE[name]
    scores, _, _ = page_scores(
        torch.cat([qc, qr], -1), torch.cat([lat, kr], -1), ratio, reduce
    )
    scale = (scores.std(dim=-1, keepdim=True) / target.clamp_min(1e-9)).clamp_min(1e-9)
    return select_latent(qc, qr, lat, kr, keep, ratio, reduce, scale)


weights = torch.load("weights/kv_b_proj.pt", weights_only=True)["kv_b_proj"]
manifest = json.loads(Path("captures/mla/manifest.json").read_text())
config = AutoConfig.from_pretrained(manifest["model"])
_, _, w_uk = grams(weights)
SCALE = softmax_scale(config)
generator = torch.Generator().manual_seed(SEED)

acc: dict[tuple, list] = {}
for n, (_, document) in enumerate(documents("captures", "mla", LIMIT)):
    query = document["queries"]["plain"]["query"]
    positions = document["queries"]["plain"]["position_ids"]
    for layer in range(weights.shape[0]):
        last = query[layer][-1:].float()
        latent = document["truth"]["latent"][layer].float()
        k_rope = document["truth"]["k_rope"][layer].float()
        tokens = latent.shape[0]

        content_q = absorbed_query(last[..., :QK_NOPE], w_uk[layer])
        rope_q = rotate_query(last[..., QK_NOPE:], positions[-1:], config)
        total = (
            torch.einsum("qhr,tr->qht", content_q, latent)
            + torch.einsum("qhd,td->qht", rope_q, k_rope)
        )[0] * SCALE
        weights_i = total.softmax(dim=-1).unsqueeze(0)
        target = total.std(dim=-1, keepdim=True)

        for overall in BUDGETS:
            keep = max(1, tokens // (2 * overall))
            RATIOS["theirs"] = Budget.of(overall).sparse_ratio
            ideal_idx = weights_i.mean(dim=(0, 1)).topk(keep).indices
            ideal = float(captured_mass(weights_i, ideal_idx).mean())
            sink = torch.arange(keep)
            acc.setdefault((overall, "sink_overlap", "ideal"), []).append(
                float(torch.isin(ideal_idx, sink).float().mean())
            )

            for m in PREFIXES:
                if m >= keep:
                    continue
                free = torch.arange(m)
                pick = keep - m
                acc.setdefault((overall, m, "prefix_only"), []).append(
                    float(captured_mass(weights_i, free).mean()) / max(ideal, 1e-9)
                    if m
                    else 0.0
                )
                floor = torch.cat(
                    [free, m + torch.randperm(tokens - m, generator=generator)[:pick]]
                )
                acc.setdefault((overall, m, "random"), []).append(
                    float(captured_mass(weights_i, floor).mean()) / max(ideal, 1e-9)
                )
                for name, kind in VARIANTS.items():
                    chosen = m + select(
                        kind, content_q[0], rope_q[0], latent[m:], k_rope[m:],
                        name, pick, target,
                    )
                    picked = torch.cat([free, chosen])
                    got = float(captured_mass(weights_i, picked).mean())
                    acc.setdefault((overall, m, name), []).append(
                        got / max(ideal, 1e-9)
                    )
                    if m == 0:
                        acc.setdefault(
                            (overall, "sink_overlap", name), []
                        ).append(float(torch.isin(chosen, sink).float().mean()))
    if (n + 1) % 10 == 0:
        print(f"  {n + 1} documents", flush=True)

out: dict = {}
names = ["prefix_only", *VARIANTS, "random"]
for overall in BUDGETS:
    print(f"\nc={overall}: share of ideal, first m tokens free to every selector")
    print(f"{'m':>3}" + "".join(f"{k:>14}" for k in names))
    for m in PREFIXES:
        if (overall, m, "random") not in acc:
            continue
        row = {k: median(acc[(overall, m, k)]) for k in names}
        out.setdefault(f"c{overall}", {})[str(m)] = row
        print(f"{m:>3}" + "".join(f"{row[k]:>14.3f}" for k in names))
    overlap = {
        k: median(acc[(overall, "sink_overlap", k)])
        for k in ("ideal", *VARIANTS)
    }
    out.setdefault(f"c{overall}", {})["sink_overlap"] = overlap
    print(
        f"    share of picks inside the first {'keep'} positions:  "
        + "  ".join(f"{k} {v:.3f}" for k, v in overlap.items())
    )

out["documents"] = LIMIT
out["layers"] = int(weights.shape[0])
out["prefixes"] = list(PREFIXES)
out["seed"] = SEED
with open("results/sink_conditioned.json", "w") as handle:
    json.dump(out, handle, indent=1)
print("\nwritten to results/sink_conditioned.json")
