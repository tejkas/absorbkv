"""Does selection survive at the kept fraction the traffic budget demands?

Measurement 2 tested step 3 at their derived kept fraction (0.48-0.64).
The traffic remedy needs 0.15-0.23 (RoPE kept whole) or 0.24-0.32 (RoPE
truncated too). Nothing measured says selection holds there. This runs
the same share-of-ideal measurement at those fractions.

The softmax temperature is computed naturally in every variant, not
pinned: this is the end-to-end behaviour of a proposed setting, not an
isolation of "which coordinates".
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig

sys.path.insert(0, "src")
from absorbkv.concentration import absorbed_query
from absorbkv.report import BUDGETS, QK_NOPE, documents, grams, median
from absorbkv.scores import captured_mass, exact_scores, rotate_query
from absorbkv.selection import Budget, select_latent

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 60
with open("results/accounting.json") as handle:
    acct = json.load(handle)
REMEDY = {c: acct[f"mla_c{c}"]["kept_fraction_to_meet_budget"] for c in BUDGETS}
REMEDY_TRUNC = {
    c: acct[f"mla_truncate_rope_c{c}"]["kept_fraction_to_meet_budget"] for c in BUDGETS
}
print("remedy fractions      ", {c: round(v, 4) for c, v in REMEDY.items()})
print("remedy fractions trunc", {c: round(v, 4) for c, v in REMEDY_TRUNC.items()})

weights = torch.load("weights/kv_b_proj.pt", weights_only=True)["kv_b_proj"]
manifest = json.loads(Path("captures/mla/manifest.json").read_text())
config = AutoConfig.from_pretrained(manifest["model"])
_, _, w_uk = grams(weights)

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
        for overall in BUDGETS:
            budget = Budget.of(overall)
            keep = max(1, tokens // (2 * overall))
            args = (content, rope, latent, k_rope, keep, budget.sparse_ratio)
            picks = {
                "ideal": scored.weights.mean(dim=(0, 1)).topk(keep).indices,
                "theirs": select_latent(*args, True),
                "remedy": select_latent(*args, True, force_kept=REMEDY[overall]),
                "remedy_trunc": select_latent(
                    *args,
                    True,
                    force_kept=REMEDY_TRUNC[overall],
                    truncate_rope=True,
                ),
            }
            for name, chosen in picks.items():
                out.setdefault((overall, layer, name), []).append(
                    float(captured_mass(scored.weights, chosen).mean())
                )
    if (n + 1) % 10 == 0:
        print(f"  {n + 1} documents", flush=True)

layers = range(weights.shape[0])
summary = {}
print()
for overall in BUDGETS:
    ideal = [median(out[(overall, L, "ideal")]) for L in layers]
    print(f"c={overall:<4}", end="")
    row = {}
    for name in ("theirs", "remedy", "remedy_trunc"):
        got = [median(out[(overall, L, name)]) for L in layers]
        share = sorted(g / i for g, i in zip(got, ideal, strict=True))
        row[name] = {
            "median": share[len(share) // 2],
            "min": share[0],
            "max": share[-1],
        }
        print(
            f"  {name:13} {row[name]['median']:.3f} ({share[0]:.2f}-{share[-1]:.2f})",
            end="",
        )
    print()
    summary[str(overall)] = row

with open("results/remedy_fidelity.json", "w") as handle:
    json.dump(
        {
            "summary": summary,
            "kept_fraction": {str(c): REMEDY[c] for c in BUDGETS},
            "kept_fraction_truncate_rope": {str(c): REMEDY_TRUNC[c] for c in BUDGETS},
            "documents": LIMIT,
            "layers": weights.shape[0],
        },
        handle,
        indent=1,
    )
print("\nwritten to results/remedy_fidelity.json")
