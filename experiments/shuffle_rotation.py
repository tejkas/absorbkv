"""Is `k^R`'s ranking power content or position?

`k^R_t = RoPE(W^KR x_t, t)` -- a 64-wide content projection that is then
rotated by the token's position. The subspace finding says scoring that
64-wide half ranks context tokens almost as well as the full logit. Two
explanations: the projected content carries the signal, or the rotation
pattern alone correlates with what the query attends to.

The control separates them. Keep every rotation at its own position but
permute which token's content is rotated there. RoPE is a rotation, so
placing token `p(t)`'s content at position `t` needs no inverse and no
hidden states -- one extra rotation of the stored key by the position
difference:

    k~^R_t = R(t - p(t)) . k^R_p(t)

`check_rotation` asserts R(+d) then R(-d) returns the original, which is
what makes that step legitimate: it checks the path really is a group
action under YaRN's frequency scaling, and that DeepSeek's rotate_half
convention is being applied consistently.

**What is informative.** The spread ratio `sd_rope / sd_content` survives
the shuffle almost by construction -- permuting which content sits at
which position gives a statistically similar set of 64-wide dot products
-- so a surviving ratio is not evidence of anything. Read the selection
quality and `rho_rope`, both against the *real* unshuffled logit. They
collapse if content is what is being found and hold if rotation is.

`latent_exact` cannot see `k^R` at all, so its number must come out
unchanged; that is the harness check on the shuffle.

Tejas predicted (before the run): quality drops toward random, possibly a
mix biased toward content, on the reasoning that DeepSeek's lightning
indexer suggests those 64 dimensions do store content and that MuSiQue
needs real retrieval rather than position.
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
BUDGETS = (16, 64, 256)
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


def rotate_key(k_rope, delta, config):
    """Rotate a [tokens, dim] key by a per-token position difference."""
    shaped = k_rope.unsqueeze(1)                       # [tokens, 1, dim]
    return rotate_query(shaped, delta, config)[:, 0, :]


def check_rotation(k_rope, config):
    """R(-d) . R(+d) must be the identity, else the shuffle is not a shuffle."""
    tokens = k_rope.shape[0]
    delta = torch.randint(-4096, 4096, (tokens,))
    there = rotate_key(k_rope, delta, config)
    back = rotate_key(there, -delta, config)
    residual = (back - k_rope).abs().max() / k_rope.abs().max().clamp_min(1e-9)
    assert residual < 1e-4, f"rotation is not invertible: {residual:.2e}"
    moved = (there - k_rope).abs().max() / k_rope.abs().max().clamp_min(1e-9)
    assert moved > 1e-2, f"rotation changed nothing: {moved:.2e}"
    return float(residual), float(moved)


def select(kind, content_q, rope_q, latent, k_rope, name, keep, target):
    """One variant at the sharpness-matched divisor."""
    zq = torch.zeros(content_q.shape[0], 0)
    zk = torch.zeros(latent.shape[0], 0)
    if kind == "mixed":
        qc, qr, lat, kr = content_q, rope_q, latent, k_rope
    elif kind == "latent":
        qc, qr, lat, kr = content_q, zq, latent, zk
    else:
        qc, qr, lat, kr = rope_q, zq, k_rope, zk
    ratio = RATIOS[name]
    reduce = REDUCE[name]
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
checked = None
for n, (_, document) in enumerate(documents("captures", "mla", LIMIT)):
    query = document["queries"]["plain"]["query"]
    positions = document["queries"]["plain"]["position_ids"]
    for layer in range(weights.shape[0]):
        last = query[layer][-1:].float()
        latent = document["truth"]["latent"][layer].float()
        k_rope = document["truth"]["k_rope"][layer].float()
        tokens = latent.shape[0]
        if checked is None:
            checked = check_rotation(k_rope, config)
            print(f"rotation check: residual {checked[0]:.2e}, moved {checked[1]:.2f}")

        content_q = absorbed_query(last[..., :QK_NOPE], w_uk[layer])
        rope_q = rotate_query(last[..., QK_NOPE:], positions[-1:], config)
        content = torch.einsum("qhr,tr->qht", content_q, latent)[0]
        rope = torch.einsum("qhd,td->qht", rope_q, k_rope)[0]
        total = content + rope

        # content of token p(t) rotated to position t; context positions
        # are arange(tokens), so the difference is t - p(t)
        perm = torch.randperm(tokens, generator=generator)
        delta = torch.arange(tokens) - perm
        k_shuffled = rotate_key(k_rope[perm], delta, config)
        rope_shuffled = torch.einsum("qhd,td->qht", rope_q, k_shuffled)[0]

        for h in range(content.shape[0]):
            acc.setdefault(("real", "rho_rope"), []).append(spearman(total[h], rope[h]))
            acc.setdefault(("shuffled", "rho_rope"), []).append(
                spearman(total[h], rope_shuffled[h])
            )
            for tag, term in (("real", rope), ("shuffled", rope_shuffled)):
                acc.setdefault((tag, "ratio"), []).append(
                    float(term[h].std()) / max(float(content[h].std()), 1e-9)
                )

        scaled = total * SCALE
        weights_i = scaled.softmax(dim=-1).unsqueeze(0)
        target = scaled.std(dim=-1, keepdim=True)
        for overall in BUDGETS:
            keep = max(1, tokens // (2 * overall))
            ideal = float(
                captured_mass(
                    weights_i, weights_i.mean(dim=(0, 1)).topk(keep).indices
                ).mean()
            )
            RATIOS["theirs"] = Budget.of(overall).sparse_ratio
            floor = torch.randperm(tokens, generator=generator)[:keep]
            acc.setdefault((overall, "random"), []).append(
                float(captured_mass(weights_i, floor).mean()) / max(ideal, 1e-9)
            )
            for name, kind in VARIANTS.items():
                for tag, kr in (("real", k_rope), ("shuffled", k_shuffled)):
                    chosen = select(
                        kind, content_q[0], rope_q[0], latent, kr, name, keep, target
                    )
                    got = float(captured_mass(weights_i, chosen).mean())
                    acc.setdefault((overall, tag, name), []).append(
                        got / max(ideal, 1e-9)
                    )
    if (n + 1) % 10 == 0:
        print(f"  {n + 1} documents", flush=True)

out: dict = {"rotation_check": {"residual": checked[0], "moved": checked[1]}}
print("\nlogit term, median over docs x layers x heads:")
print(f"{'keys':>10} {'ratio':>7} {'rho_rope':>9}")
for tag in ("real", "shuffled"):
    row = {k: median(acc[(tag, k)]) for k in ("ratio", "rho_rope")}
    out[f"terms_{tag}"] = row
    print(f"{tag:>10} {row['ratio']:>7.2f} {row['rho_rope']:>9.3f}")

print("\nshare of ideal attention mass (median over docs x layers):")
print(f"{'variant':>14}" + "".join(f"{f'c={c}':>19}" for c in BUDGETS))
print(f"{'':>14}" + "".join(f"{'real':>9}{'shuffled':>10}" for _ in BUDGETS))
for name in VARIANTS:
    cells = ""
    for overall in BUDGETS:
        for tag in ("real", "shuffled"):
            value = median(acc[(overall, tag, name)])
            out.setdefault(f"c{overall}", {}).setdefault(tag, {})[name] = value
            cells += f"{value:>9.3f}" if tag == "real" else f"{value:>10.3f}"
    print(f"{name:>14}{cells}")
cells = ""
for overall in BUDGETS:
    value = median(acc[(overall, "random")])
    out.setdefault(f"c{overall}", {})["random"] = value
    cells += f"{value:>9.3f}{'':>10}"
print(f"{'random':>14}{cells}")

out["documents"] = LIMIT
out["layers"] = int(weights.shape[0])
out["seed"] = SEED
with open("results/shuffle_rotation.json", "w") as handle:
    json.dump(out, handle, indent=1)
print("\nwritten to results/shuffle_rotation.json")
