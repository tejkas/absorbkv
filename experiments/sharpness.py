"""What a matched divisor still does not equalize.

``temperature.py`` forces every variant to divide its page scores by the
model's 8.717. That makes the divisor identical, not the softmax. A
512-wide dot product has a wider spread across pages than a 64-wide one,
so the same divisor leaves the latent's softmax *sharper*, not matched.

Reported per variant: the standard deviation across pages of
``page_scores / scale``, which is what the softmax in step 5 actually
sees, under both the natural and the matched divisor. The model's own
attention logits are included as the reference the selector is trying to
approximate.

Steps 1-4 of the selector are mirrored here to get at the intermediate
scores, and ``check_mirror`` asserts the mirror reproduces
``select_latent``'s own output before any number is reported.
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
from absorbkv.scores import exact_scores, rotate_query, softmax_scale
from absorbkv.selection import Budget, select_latent

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 20


weights = torch.load("weights/kv_b_proj.pt", weights_only=True)["kv_b_proj"]
manifest = json.loads(Path("captures/mla/manifest.json").read_text())
config = AutoConfig.from_pretrained(manifest["model"])
_, _, w_uk = grams(weights)
MATCHED = 1.0 / softmax_scale(config)

acc: dict[tuple, list] = {}
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
        content = absorbed_query(last[..., :QK_NOPE], w_uk[layer])[0]
        rope = rotate_query(last[..., QK_NOPE:], positions[-1:], config)[0]
        latent = document["truth"]["latent"][layer].float()
        k_rope = document["truth"]["k_rope"][layer].float()
        full_q = torch.cat([content, rope], dim=-1)
        full_k = torch.cat([latent, k_rope], dim=-1)

        if not checked:
            zq = torch.zeros(rope.shape[0], 0)
            zk = torch.zeros(k_rope.shape[0], 0)
            check_mirror(select_latent, rope, zq, k_rope, zk, 4.0, False, 20)
            check_mirror(select_latent, content, zq, latent, zk, 9.0, False, 20)
            check_mirror(select_latent, content, rope, latent, k_rope, 9.0, True, 20)
            print("mirror verified against select_latent")
            checked = True

        # the model's own logits, the thing being approximated
        acc.setdefault(("model logits", "-"), []).append(
            float(scored.logits[0].std(dim=-1).median())
        )
        for overall in BUDGETS:
            sparse = Budget.of(overall).sparse_ratio
            for name, q, k, ratio, reduce in (
                (f"theirs c={overall}", full_q, full_k, sparse, True),
                (f"latent_pooled c={overall}", content, latent, sparse, False),
                ("latent_exact", content, latent, 1.0, False),
                ("rope_page2", rope, k_rope, 4.0, False),
                ("rope_page1", rope, k_rope, 1.0, False),
            ):
                scores, natural, _ = page_scores(q, k, ratio, reduce)
                acc.setdefault((name, "natural"), []).append(
                    float((scores / natural).std(dim=-1).median())
                )
                acc.setdefault((name, "matched"), []).append(
                    float((scores / MATCHED).std(dim=-1).median())
                )
    if (n + 1) % 10 == 0:
        print(f"  {n + 1} documents", flush=True)

print(f"\nmatched divisor = {MATCHED:.3f}")
print("\nspread across pages of what the softmax sees"
      " (median over docs x layers x heads):")
print(f"{'variant':>24} {'natural divisor':>16} {'matched divisor':>16}")
rows = {}
seen = []
for name, _ in acc:
    if name not in seen:
        seen.append(name)
for name in seen:
    nat = acc.get((name, "natural"))
    mat = acc.get((name, "matched"))
    if nat is None:
        print(f"{name:>24} {median(acc[(name, '-')]):>16.2f} {'(reference)':>16}")
        rows[name] = {"reference": median(acc[(name, "-")])}
        continue
    print(f"{name:>24} {median(nat):>16.2f} {median(mat):>16.2f}")
    rows[name] = {"natural": median(nat), "matched": median(mat)}

with open("results/sharpness.json", "w") as handle:
    json.dump(
        {
            "matched_divisor": MATCHED,
            "by_variant": rows,
            "documents": LIMIT,
            "layers": weights.shape[0],
        },
        handle,
        indent=1,
    )
print("\nwritten to results/sharpness.json")
