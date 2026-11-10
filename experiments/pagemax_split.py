"""Is the latent's weak selection its own fault, or the page summary's?

``content_only`` scored 0.024 of ideal -- near random -- while the
content term alone ranks true logits at Spearman 0.52. Something between
the term and the selection is destroying the signal. The candidate is
step 2's elementwise max over each page: the latent passes through
``kv_a_layernorm``, so its coordinates are scale-controlled, and a max
over 3 such tokens may come out nearly the same for every page.

Four variants, each at ``keep = n/(2c)`` tokens:

  content_pooled : latent only, their page size (3) -- the number above
  content_exact  : latent only, page size 1, so no max pooling at all
  rope_pooled    : 64 RoPE coordinates only, page size 3
  rope_exact     : 64 RoPE coordinates only, page size 1

If content_exact recovers and content_pooled does not, the page max is
the cause. If neither recovers, the latent does not rank tokens here.
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
weights = torch.load("weights/kv_b_proj.pt", weights_only=True)["kv_b_proj"]
manifest = json.loads(Path("captures/mla/manifest.json").read_text())
config = AutoConfig.from_pretrained(manifest["model"])
_, _, w_uk = grams(weights)
NONE = "no pooling, no reduction"

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
        empty_q = torch.zeros(content.shape[0], 0)
        empty_k = torch.zeros(tokens, 0)
        for overall in BUDGETS:
            pooled = Budget.of(overall).sparse_ratio  # page size 3
            keep = max(1, tokens // (2 * overall))
            picks = {
                "ideal": scored.weights.mean(dim=(0, 1)).topk(keep).indices,
                "content_pooled": select_latent(
                    content, empty_q, latent, empty_k, keep, pooled, False
                ),
                "content_exact": select_latent(
                    content, empty_q, latent, empty_k, keep, 1.0, False
                ),
                "rope_pooled": select_latent(
                    rope, empty_q, k_rope, empty_k, keep, pooled, False
                ),
                "rope_exact": select_latent(
                    rope, empty_q, k_rope, empty_k, keep, 1.0, False
                ),
            }
            for name, chosen in picks.items():
                out.setdefault((overall, layer, name), []).append(
                    float(captured_mass(scored.weights, chosen).mean())
                )
    if (n + 1) % 15 == 0:
        print(f"  {n + 1} documents", flush=True)

layers = range(weights.shape[0])
print(f"\n{'variant':>16}", end="")
for overall in BUDGETS:
    print(f"   c={overall:<14}", end="")
print()
summary = {}
for name in ("content_pooled", "content_exact", "rope_pooled", "rope_exact"):
    print(f"{name:>16}", end="")
    row = {}
    for overall in BUDGETS:
        ideal = [median(out[(overall, L, "ideal")]) for L in layers]
        got = [median(out[(overall, L, name)]) for L in layers]
        share = sorted(g / i for g, i in zip(got, ideal, strict=True))
        row[str(overall)] = {
            "median": share[len(share) // 2],
            "min": share[0],
            "max": share[-1],
        }
        print(
            f"   {share[len(share) // 2]:.3f} ({share[0]:.2f}-{share[-1]:.2f})", end=""
        )
    print()
    summary[name] = row

with open("results/pagemax_split.json", "w") as handle:
    json.dump(
        {"summary": summary, "documents": LIMIT, "layers": weights.shape[0]},
        handle,
        indent=1,
    )
print("\nwritten to results/pagemax_split.json")
