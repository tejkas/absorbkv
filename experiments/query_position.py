"""Does the subspace result hold away from the last prompt token?

Every relevance number so far comes from one query token. `q^R` rotates
with the query's position and `q'` does not, so the 2.08x spread ratio
the subspace finding rests on was measured at a single query-key
distance. Two sweeps, because the captures can only answer half of it.

**real** -- all stored query positions. The question is appended after
the whole context (context 0..2344, query 2345..2370), so every stored
position can attend to all of it and no masking is needed. But that also
means the positions span 26 tokens out of ~2345: this varies query
*content*, not distance. It is the honest sweep and it tests the weaker
claim.

**synthetic** -- the last query token's content re-rotated to positions
far past the context, holding `q'` and the cache fixed. Distance is then
the only thing that changes, which is exactly the concern. It is not a
real decode state, since a query actually generated 16k tokens later
would carry different content; it isolates the rotation and nothing else.
YaRN gives this model an effective 163840 positions over an original
4096, so the offsets stay in distribution.

Reported at each position: the logit split (`sd_rope / sd_content` and
each term's rank correlation with the full logit) and selection quality
for the variants that matter, at c=64 and the sharpness-matched divisor.
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
from absorbkv.report import QK_NOPE, documents, grams, median, spearman
from absorbkv.scores import captured_mass, rotate_query, softmax_scale
from absorbkv.selection import Budget, select_latent

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 60
BUDGET = 64
SELECT_AT = (0, 8, 17, 25)          # indices into the stored query tokens
OFFSETS = (0, 64, 256, 1024, 4096, 16384, 65536)

weights = torch.load("weights/kv_b_proj.pt", weights_only=True)["kv_b_proj"]
manifest = json.loads(Path("captures/mla/manifest.json").read_text())
config = AutoConfig.from_pretrained(manifest["model"])
_, _, w_uk = grams(weights)
SCALE = softmax_scale(config)
SPARSE = Budget.of(BUDGET).sparse_ratio
VARIANTS = {
    "theirs": (True, SPARSE, True),
    "latent_exact": (False, 1.0, False),
    "rope_page2": (None, 4.0, False),
    "rope_page1": (None, 1.0, False),
}


def select(kind, content, rope, latent, k_rope, ratio, reduce, keep, target):
    """One variant, with the softmax spread forced to the model's."""
    zq = torch.zeros(content.shape[0], 0)
    zk = torch.zeros(latent.shape[0], 0)
    if kind is True:
        qc, qr, lat, kr = content, rope, latent, k_rope
    elif kind is False:
        qc, qr, lat, kr = content, zq, latent, zk
    else:
        qc, qr, lat, kr = rope, zq, k_rope, zk
    scores, _, _ = page_scores(
        torch.cat([qc, qr], -1), torch.cat([lat, kr], -1), ratio, reduce
    )
    scale = (scores.std(dim=-1, keepdim=True) / target.clamp_min(1e-9)).clamp_min(1e-9)
    return select_latent(qc, qr, lat, kr, keep, ratio, reduce, scale)


def terms(content_q, rope_q, latent, k_rope):
    """The two halves of q.k for a batch of query tokens, pre-scale."""
    content = torch.einsum("qhr,tr->qht", content_q, latent)
    positional = torch.einsum("qhd,td->qht", rope_q, k_rope)
    return content, positional


acc: dict[tuple, list] = {}
for n, (_, document) in enumerate(documents("captures", "mla", LIMIT)):
    query = document["queries"]["plain"]["query"]
    positions = document["queries"]["plain"]["position_ids"]
    num_q = query.shape[1]
    for layer in range(weights.shape[0]):
        raw = query[layer].float()
        latent = document["truth"]["latent"][layer].float()
        k_rope = document["truth"]["k_rope"][layer].float()
        tokens = latent.shape[0]
        keep = max(1, tokens // (2 * BUDGET))

        absorbed = absorbed_query(raw[..., :QK_NOPE], w_uk[layer])

        for mode in ("real", "synthetic"):
            if mode == "real":
                labels = list(range(num_q))
                rope_q = rotate_query(raw[..., QK_NOPE:], positions, config)
                content_q = absorbed
                select_idx = [i for i in SELECT_AT if i < num_q]
            else:
                labels = list(OFFSETS)
                last = raw[-1:, :, QK_NOPE:]
                shifted = torch.tensor(
                    [int(positions[-1]) + o for o in OFFSETS], dtype=torch.long
                )
                rope_q = rotate_query(
                    last.expand(len(OFFSETS), -1, -1).contiguous(), shifted, config
                )
                content_q = absorbed[-1:].expand(len(OFFSETS), -1, -1)
                select_idx = list(range(len(OFFSETS)))

            content, positional = terms(content_q, rope_q, latent, k_rope)
            total = content + positional
            for i, label in enumerate(labels):
                key = (mode, label)
                for h in range(content.shape[1]):
                    acc.setdefault((*key, "sd_content"), []).append(
                        float(content[i, h].std())
                    )
                    acc.setdefault((*key, "sd_rope"), []).append(
                        float(positional[i, h].std())
                    )
                    acc.setdefault((*key, "rho_rope"), []).append(
                        spearman(total[i, h], positional[i, h])
                    )
                    acc.setdefault((*key, "rho_content"), []).append(
                        spearman(total[i, h], content[i, h])
                    )
                if i not in select_idx:
                    continue
                weights_i = (total[i] * SCALE).softmax(dim=-1).unsqueeze(0)
                target = (total[i] * SCALE).std(dim=-1, keepdim=True)
                ideal = float(
                    captured_mass(
                        weights_i, weights_i.mean(dim=(0, 1)).topk(keep).indices
                    ).mean()
                )
                acc.setdefault((*key, "ideal"), []).append(ideal)
                for name, (kind, ratio, reduce) in VARIANTS.items():
                    chosen = select(
                        kind, content_q[i], rope_q[i], latent, k_rope,
                        ratio, reduce, keep, target,
                    )
                    got = float(captured_mass(weights_i, chosen).mean())
                    acc.setdefault((*key, name), []).append(got / max(ideal, 1e-9))
    if (n + 1) % 10 == 0:
        print(f"  {n + 1} documents", flush=True)

out = {}
layer_count = weights.shape[0]
for mode, header in (("real", "stored query positions (index into the question)"),
                     ("synthetic", "last token re-rotated, offset past the context")):
    labels = sorted({lab for m, lab, _ in acc if m == mode})
    print(f"\n=== {mode}: {header}")
    print(
        f"{'pos':>7} {'docs':>5} {'ratio':>6} {'rho_rope':>9} {'rho_cont':>9}"
        f" {'ideal':>7}   {'theirs':>7} {'latent':>7} {'rope_p2':>8} {'rope_p1':>8}"
    )
    for lab in labels:
        sc = median(acc[(mode, lab, "sd_content")])
        sr = median(acc[(mode, lab, "sd_rope")])
        row = {
            "sd_content": sc, "sd_rope": sr, "ratio": sr / max(sc, 1e-9),
            "rho_rope": median(acc[(mode, lab, "rho_rope")]),
            "rho_content": median(acc[(mode, lab, "rho_content")]),
        }
        cells = ""
        for name in VARIANTS:
            vals = acc.get((mode, lab, name))
            if vals is None:
                cells += f" {'-':>8}"
            else:
                row[name] = median(vals)
                cells += f" {row[name]:>8.3f}"
        # absolute attention mass on the top k: if this approaches 1.0 the
        # selection task has degenerated and 'share of ideal' means little
        ideal = acc.get((mode, lab, "ideal"))
        row["ideal_mass"] = median(ideal) if ideal else None
        row["heads_sampled"] = len(acc[(mode, lab, "sd_content")])
        row["documents"] = row["heads_sampled"] // (16 * layer_count)
        out[f"{mode}_{lab}"] = row
        mass = f"{row['ideal_mass']:.3f}" if ideal else "-"
        print(
            f"{lab:>7} {row['documents']:>5} {row['ratio']:>6.2f}"
            f" {row['rho_rope']:>9.3f} {row['rho_content']:>9.3f}"
            f" {mass:>7}  {cells}"
        )

with open("results/query_position.json", "w") as handle:
    json.dump(
        {"by_position": out, "budget": BUDGET, "offsets": list(OFFSETS),
         "select_at": list(SELECT_AT), "documents": LIMIT,
         "layers": weights.shape[0]},
        handle, indent=1,
    )
print("\nwritten to results/query_position.json")
