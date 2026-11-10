"""Which of the two terms in q.k carries the ranking of context tokens?

    q . k_t  =  q' . c_t  +  q^R . k^R_t

Selection ranks tokens, so what matters is not each term's size but how
much each one *varies across tokens* -- a term identical for every token
shifts all logits alike and changes no ranking. Reported per layer:

  sd_content, sd_rope : standard deviation across context tokens of each
      term, median over heads and documents. Comparable units; both are
      pre-softmax-scale.
  rho_rope, rho_content : Spearman rank correlation between the full
      logit and each term alone. 1.0 means that term alone reproduces the
      true ordering of tokens; 0.0 means it carries none of it.
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig

sys.path.insert(0, "src")
from absorbkv.concentration import absorbed_query
from absorbkv.report import QK_NOPE, documents, grams, median, spearman
from absorbkv.scores import rotate_query

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 60
weights = torch.load("weights/kv_b_proj.pt", weights_only=True)["kv_b_proj"]
manifest = json.loads(Path("captures/mla/manifest.json").read_text())
config = AutoConfig.from_pretrained(manifest["model"])
_, _, w_uk = grams(weights)

acc: dict[tuple, list] = {}
for n, (_, document) in enumerate(documents("captures", "mla", LIMIT)):
    query = document["queries"]["plain"]["query"]
    positions = document["queries"]["plain"]["position_ids"]
    for layer in range(weights.shape[0]):
        last = query[layer][-1:].float()
        latent = document["truth"]["latent"][layer].float()
        k_rope = document["truth"]["k_rope"][layer].float()
        absorbed = absorbed_query(last[..., :QK_NOPE], w_uk[layer])
        content = torch.einsum("qhr,tr->qht", absorbed, latent)[0]
        rotated = rotate_query(last[..., QK_NOPE:], positions[-1:], config)
        rope = torch.einsum("qhd,td->qht", rotated, k_rope)[0]
        total = content + rope
        for h in range(content.shape[0]):
            acc.setdefault((layer, "sd_content"), []).append(float(content[h].std()))
            acc.setdefault((layer, "sd_rope"), []).append(float(rope[h].std()))
            acc.setdefault((layer, "rho_rope"), []).append(spearman(total[h], rope[h]))
            acc.setdefault((layer, "rho_content"), []).append(
                spearman(total[h], content[h])
            )
    if (n + 1) % 15 == 0:
        print(f"  {n + 1} documents", flush=True)

layers = range(weights.shape[0])
print(
    f"\n{'layer':>5} {'sd_content':>11} {'sd_rope':>9} {'ratio':>7}"
    f" {'rho_rope':>9} {'rho_content':>12}"
)
rows = {}
for L in layers:
    sc = median(acc[(L, "sd_content")])
    sr = median(acc[(L, "sd_rope")])
    rr = median(acc[(L, "rho_rope")])
    rc = median(acc[(L, "rho_content")])
    rows[L] = {"sd_content": sc, "sd_rope": sr, "rho_rope": rr, "rho_content": rc}
    print(
        f"{L:>5} {sc:>11.3f} {sr:>9.3f} {sr / max(sc, 1e-9):>7.2f}"
        f" {rr:>9.3f} {rc:>12.3f}"
    )

ratio = [rows[L]["sd_rope"] / max(rows[L]["sd_content"], 1e-9) for L in layers]
print(
    f"\nmedian over layers:  sd ratio {median(ratio):.2f}"
    f"   rho_rope {median([rows[L]['rho_rope'] for L in layers]):.3f}"
    f"   rho_content {median([rows[L]['rho_content'] for L in layers]):.3f}"
)

with open("results/logit_split.json", "w") as handle:
    json.dump(
        {
            "by_layer": {str(L): rows[L] for L in layers},
            "documents": LIMIT,
            "heads": 16,
        },
        handle,
        indent=1,
    )
print("written to results/logit_split.json")
