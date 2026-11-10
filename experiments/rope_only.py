"""Score only k^R: what it costs in traffic and what it buys in quality.

Their step 2 summarizes the whole 576-wide stored token, which on MLA is
why it blows the budget. If the 64-wide k^R carries the ranking, step 2
can read k^R alone -- 64 of 576 bytes -- and the page size becomes a free
parameter trading traffic against quality.

Traffic, in units of t = S/c tokens' worth of bytes (W_token = 576):

    step 2 = (S / c^r / page) * 64   ->   c^(1-r) / page * 64/576  of t
    step 3 = t/2 tokens * 576        ->   0.500 of t   (unchanged)

delivered = c / total. Quality is share of ideal, measured here at the
same keep = n/(2c) as everything else, 60 documents, every layer.
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig

sys.path.insert(0, "src")
from absorbkv.report import BUDGETS, QK_NOPE, documents, grams, median
from absorbkv.scores import captured_mass, exact_scores, rotate_query
from absorbkv.selection import Budget, select_latent

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 60
RATIO = {1: 1.0, 2: 4.0, 3: 9.0}  # ceil(sqrt(cr)) -> page size
STORED, SUMMARY = 576, 64

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
        rope = rotate_query(last[..., QK_NOPE:], positions[-1:], config)[0]
        k_rope = document["truth"]["k_rope"][layer].float()
        empty_q = torch.zeros(rope.shape[0], 0)
        empty_k = torch.zeros(tokens, 0)
        for overall in BUDGETS:
            keep = max(1, tokens // (2 * overall))
            out.setdefault((overall, layer, "ideal"), []).append(
                float(
                    captured_mass(
                        scored.weights,
                        scored.weights.mean(dim=(0, 1)).topk(keep).indices,
                    ).mean()
                )
            )
            for page, ratio in RATIO.items():
                chosen = select_latent(
                    rope, empty_q, k_rope, empty_k, keep, ratio, False
                )
                out.setdefault((overall, layer, page), []).append(
                    float(captured_mass(scored.weights, chosen).mean())
                )
    if (n + 1) % 20 == 0:
        print(f"  {n + 1} documents", flush=True)

layers = range(weights.shape[0])
rows = {}
print(
    f"\n{'c':>5} {'page':>5} {'step2':>7} {'step3':>7} {'total':>7}"
    f" {'delivered':>10} {'asked':>6} {'quality':>18}"
)
for overall in BUDGETS:
    budget = Budget.of(overall)
    ideal = [median(out[(overall, L, "ideal")]) for L in layers]
    for page in RATIO:
        step2 = budget.sparse_ratio / page * SUMMARY / STORED
        total = step2 + 0.5
        got = [median(out[(overall, L, page)]) for L in layers]
        share = sorted(g / i for g, i in zip(got, ideal, strict=True))
        q = share[len(share) // 2]
        rows[f"c{overall}_page{page}"] = {
            "step2_in_t": step2,
            "step3_in_t": 0.5,
            "total_in_t": total,
            "delivered_ratio": overall / total,
            "requested_ratio": overall,
            "quality_median": q,
            "quality_min": share[0],
            "quality_max": share[-1],
            # per layer, in layer order -- the median alone hides how much
            # of the gain is a few layers moving a long way
            "quality_by_layer": [
                g / i for g, i in zip(got, ideal, strict=True)
            ],
        }
        print(
            f"{overall:>5} {page:>5} {step2:>7.3f} {0.5:>7.3f} {total:>7.3f}"
            f" {overall / total:>10.1f} {overall:>6}"
            f"   {q:>.3f} ({share[0]:.2f}-{share[-1]:.2f})"
        )

with open("results/rope_only.json", "w") as handle:
    json.dump(
        {
            "by_setting": rows,
            "stored_width": STORED,
            "summary_width": SUMMARY,
            "documents": LIMIT,
            "layers": weights.shape[0],
        },
        handle,
        indent=1,
    )
print("\nwritten to results/rope_only.json")
