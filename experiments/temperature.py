"""Every variant again at one matched selector temperature.

Step 5 of their pipeline divides the page scores by

    sqrt(width * |kept coords of q| / |q|)

and step 6 averages the resulting softmax weights across heads before
taking the top k. That average depends on sharpness: a sharp softmax
makes it a head-by-head vote, a flat one lets whichever head has the
widest logit range dominate. Because the divisor is built from the
*width* of whatever is being scored, each variant ran at a different
temperature -- 8.0 for the 64-wide k^R, 22.6 for the 512-wide latent,
20.2-21.6 for theirs -- against the model's own 8.717.

k^R therefore happened to sit at the model's temperature and the latent
2.6x flatter, which flatters k^R for a reason unrelated to how much
signal it carries. ``force_scale`` overrides the divisor, so every
variant here is run twice: at its natural temperature and at the model's.

The matched column is the one to quote. Note what it does and does not
equalize: it makes the divisor identical, not the effective sharpness,
since a 512-wide dot product has a larger spread than a 64-wide one.
``sharpness.py`` measures what is left.
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig

sys.path.insert(0, "src")
from absorbkv.concentration import absorbed_query
from absorbkv.report import BUDGETS, QK_NOPE, documents, grams, median
from absorbkv.scores import captured_mass, exact_scores, rotate_query, softmax_scale
from absorbkv.selection import Budget, select_latent

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 60
PAGE_RATIO = {1: 1.0, 2: 4.0, 3: 9.0}

weights = torch.load("weights/kv_b_proj.pt", weights_only=True)["kv_b_proj"]
manifest = json.loads(Path("captures/mla/manifest.json").read_text())
config = AutoConfig.from_pretrained(manifest["model"])
_, _, w_uk = grams(weights)
MATCHED = 1.0 / softmax_scale(config)
print(f"matched temperature = the model's own = {MATCHED:.3f}")

out: dict[tuple, list] = {}
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
        empty_q = torch.zeros(heads, 0)
        empty_k = torch.zeros(tokens, 0)
        forced = torch.full((heads, 1), MATCHED)

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
            for temp, scale in (("natural", None), ("matched", forced)):
                picks = {
                    "theirs": select_latent(
                        content, rope, latent, k_rope, keep, sparse, True, scale
                    ),
                    "latent_pooled": select_latent(
                        content, empty_q, latent, empty_k, keep, sparse, False, scale
                    ),
                    "latent_exact": select_latent(
                        content, empty_q, latent, empty_k, keep, 1.0, False, scale
                    ),
                }
                for page, ratio in PAGE_RATIO.items():
                    picks[f"rope_page{page}"] = select_latent(
                        rope, empty_q, k_rope, empty_k, keep, ratio, False, scale
                    )
                for name, chosen in picks.items():
                    out.setdefault((overall, layer, f"{name}/{temp}"), []).append(
                        float(captured_mass(scored.weights, chosen).mean())
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
summary: dict[str, dict] = {}
for overall in BUDGETS:
    ideal = [median(out[(overall, L, "ideal")]) for L in layers]
    print(f"\nc={overall}")
    print(f"{'variant':>16} {'natural':>22} {'matched':>22} {'delta':>8}")
    for name in names:
        cells, meds = [], {}
        for temp in ("natural", "matched"):
            got = [median(out[(overall, L, f"{name}/{temp}")]) for L in layers]
            share = sorted(g / i for g, i in zip(got, ideal, strict=True))
            meds[temp] = share[len(share) // 2]
            cells.append(f"{meds[temp]:.3f} ({share[0]:.2f}-{share[-1]:.2f})")
            summary.setdefault(f"c{overall}_{name}", {})[temp] = {
                "median": meds[temp],
                "min": share[0],
                "max": share[-1],
                "by_layer": [g / i for g, i in zip(got, ideal, strict=True)],
            }
        delta = meds["matched"] - meds["natural"]
        print(f"{name:>16} {cells[0]:>22} {cells[1]:>22} {delta:>+8.3f}")

with open("results/temperature.json", "w") as handle:
    json.dump(
        {
            "matched_temperature": MATCHED,
            "summary": summary,
            "documents": LIMIT,
            "layers": weights.shape[0],
        },
        handle,
        indent=1,
    )
print("\nwritten to results/temperature.json")
