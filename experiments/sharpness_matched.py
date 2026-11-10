"""Every variant with its softmax spread forced to equal the model's.

Forcing a common divisor (``temperature.py``) equalizes the divisor, not
the softmax: the latent's page scores have an intrinsically narrow spread
and stay about 2.9x flatter than the model's logits even after matching.
This is the control that removes sharpness as a variable outright.

For each document, layer and head, the divisor is

    scale[h] = sd(page_scores[h]) / sd(model_logits[h])

so that ``sd(page_scores[h] / scale[h]) == sd(model_logits[h])`` by
construction. Every variant then feeds step 6 a softmax of the same
sharpness as the attention it is trying to approximate, and any remaining
difference is the ordering of the scores rather than their scale.

The page scores come from the mirror in ``_mirror.py``, which is asserted
against ``select_latent`` before anything is measured.
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from _mirror import check_mirror, page_scores

from absorbkv.concentration import absorbed_query
from absorbkv.report import BUDGETS, QK_NOPE, documents, grams, median
from absorbkv.scores import captured_mass, exact_scores, rotate_query
from absorbkv.selection import Budget, select_latent

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 60

weights = torch.load("weights/kv_b_proj.pt", weights_only=True)["kv_b_proj"]
manifest = json.loads(Path("captures/mla/manifest.json").read_text())
config = AutoConfig.from_pretrained(manifest["model"])
_, _, w_uk = grams(weights)

out: dict[tuple, list] = {}
spread: dict[tuple, list] = {}
checked = False
for n, (_, document) in enumerate(documents("captures", "mla", LIMIT)):
    query = document["queries"]["plain"]["query"]
    positions = document["queries"]["plain"]["position_ids"]
    for layer in range(weights.shape[0]):
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
        heads = content.shape[0]
        zq, zk = torch.zeros(heads, 0), torch.zeros(tokens, 0)
        # per head, the spread the selector must reproduce
        target = scored.logits[0].std(dim=-1, keepdim=True)

        if not checked:
            check_mirror(select_latent, rope, zq, k_rope, zk, 4.0, False, 20)
            check_mirror(select_latent, content, rope, latent, k_rope, 9.0, True, 20)
            print("mirror verified against select_latent")
            checked = True

        for overall in BUDGETS:
            sparse = Budget.of(overall).sparse_ratio
            keep = max(1, tokens // (2 * overall))
            out.setdefault((overall, layer, "ideal"), []).append(
                float(
                    captured_mass(
                        scored.weights,
                        scored.weights.mean(dim=(0, 1)).topk(keep).indices,
                    ).mean()
                )
            )
            variants = {
                "theirs": (content, rope, latent, k_rope, sparse, True),
                "latent_pooled": (content, zq, latent, zk, sparse, False),
                "latent_exact": (content, zq, latent, zk, 1.0, False),
                "rope_page3": (rope, zq, k_rope, zk, 9.0, False),
                "rope_page2": (rope, zq, k_rope, zk, 4.0, False),
                "rope_page1": (rope, zq, k_rope, zk, 1.0, False),
            }
            for name, (qc, qr, lat, kr, ratio, reduce) in variants.items():
                scores, _, _ = page_scores(
                    torch.cat([qc, qr], -1), torch.cat([lat, kr], -1), ratio, reduce
                )
                scale = scores.std(dim=-1, keepdim=True) / target.clamp_min(1e-9)
                chosen = select_latent(
                    qc, qr, lat, kr, keep, ratio, reduce, scale.clamp_min(1e-9)
                )
                out.setdefault((overall, layer, name), []).append(
                    float(captured_mass(scored.weights, chosen).mean())
                )
                spread.setdefault(name, []).append(
                    float((scores / scale.clamp_min(1e-9)).std(dim=-1).median())
                )
    if (n + 1) % 15 == 0:
        print(f"  {n + 1} documents", flush=True)

layers = range(weights.shape[0])
names = [
    "theirs",
    "latent_pooled",
    "latent_exact",
    "rope_page3",
    "rope_page2",
    "rope_page1",
]
print("\nsoftmax spread after forcing (should all equal the model's):")
for name in names:
    print(f"  {name:>14} {median(spread[name]):.3f}")

summary: dict[str, dict] = {}
for overall in BUDGETS:
    ideal = [median(out[(overall, L, "ideal")]) for L in layers]
    print(f"\nc={overall}")
    for name in names:
        got = [median(out[(overall, L, name)]) for L in layers]
        ratio = [g / i for g, i in zip(got, ideal, strict=True)]
        share = sorted(ratio)
        m = share[len(share) // 2]
        summary[f"c{overall}_{name}"] = {
            "median": m,
            "min": share[0],
            "max": share[-1],
            "by_layer": ratio,
        }
        print(f"{name:>16}   {m:.3f} ({share[0]:.2f}-{share[-1]:.2f})")

with open("results/sharpness_matched.json", "w") as handle:
    json.dump(
        {
            "summary": summary,
            "spread_after_forcing": {k: median(v) for k, v in spread.items()},
            "documents": LIMIT,
            "layers": weights.shape[0],
        },
        handle,
        indent=1,
    )
print("\nwritten to results/sharpness_matched.json")
