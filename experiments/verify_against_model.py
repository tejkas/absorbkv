"""Does scoring from the latent reproduce the model's own attention?

Needs a GPU and the real weights. The laptop version of this check runs
against a small MLA model in `tests/test_scores.py`; it validates the
formula but not the weights. This one validates both, on one real
document.

The comparison: forward the document and its question, hook `q_proj` for
the last token's query before any rotation, read the latent and `k^R` out
of the model's own cache, score with `exact_scores`, and compare against
the attention the eager path computed by expanding per-head keys.

`output_attentions=True` would keep every layer's `[1, heads, T, T]`
tensor -- 9.7 GB at 27 layers and 2371 tokens, on top of ~31 GB of
weights. The hook captures the one row that is needed and returns `None`
in the tensor's place, so each layer's matrix is freed as soon as it is
read and the peak is one layer's worth.

Expect agreement near bf16 precision, not the 1e-08 of the fp32 test:
the model computes in bf16 and `exact_scores` promotes to fp32, so the
two differ by rounding. The rank statistics are the ones every relevance
claim depends on, so they are reported next to the raw difference.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "src")

from absorbkv.data import load_documents
from absorbkv.report import jaccard, spearman
from absorbkv.scores import exact_scores
from absorbkv.signals import extract_w_uk

MODEL = "deepseek-ai/DeepSeek-V2-Lite"


def capture_rows(model, target: int):
    """Hook every attention module; keep one row, drop the matrices."""
    rows: dict[int, torch.Tensor] = {}
    queries: dict[int, torch.Tensor] = {}
    config = model.config
    head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

    def attention_hook(index):
        def hook(module, args, output):
            if isinstance(output, tuple) and len(output) > 1:
                if output[1] is not None:
                    rows[index] = output[1][0, :, -1, :].detach().float().cpu()
                return (output[0], None)
            return output

        return hook

    def query_hook(module, args, output):
        queries[target] = (
            output[0, -1:]
            .unflatten(-1, (config.num_attention_heads, head_dim))
            .detach()
            .float()
            .cpu()
            .clone()
        )

    handles = []
    for index, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_hook(attention_hook(index)))
    handles.append(
        model.model.layers[target].self_attn.q_proj.register_forward_hook(query_hook)
    )
    return rows, queries, handles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument(
        "--document", type=int, default=0, help="index into the seed's order"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-context", type=int, default=0, help="0 means the whole document"
    )
    # Stamped with the layer by default: a fixed path means a second run
    # silently destroys the first, which is how layer 0 was lost once.
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    documents = load_documents(tokenizer, args.document + 1, args.seed)
    document = documents[args.document]
    context = torch.cat(document.paragraphs, dim=1)
    if args.max_context:
        context = context[:, : args.max_context]
    ids = torch.cat([context, document.question_plain], dim=1)
    tokens = ids.shape[1]
    print(
        f"{document.record_id}: {context.shape[1]} context + "
        f"{document.question_plain.shape[1]} question = {tokens} tokens",
        flush=True,
    )

    config = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(args.device)
    model.eval()
    print(f"loaded {args.model} on {args.device}", flush=True)

    rows, queries, handles = capture_rows(model, args.layer)
    positions = torch.arange(tokens, device=args.device).unsqueeze(0)
    try:
        with torch.no_grad():
            out = model(
                input_ids=ids.to(args.device),
                position_ids=positions,
                use_cache=True,
                output_attentions=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    layer = model.model.layers[args.layer].self_attn
    cache = out.past_key_values
    w_uk = extract_w_uk(
        layer.kv_b_proj.weight.float().cpu(),
        config.num_attention_heads,
        config.qk_nope_head_dim,
        config.v_head_dim,
    )
    scored = exact_scores(
        queries[args.layer],
        torch.tensor([tokens - 1]),
        cache.layers[args.layer].keys[0, 0].float().cpu(),
        cache.layers[args.layer].values[0, 0].float().cpu(),
        w_uk,
        config,
    )
    mine = scored.weights[0]
    theirs = rows[args.layer]

    peak = float(theirs.max())
    difference = float((mine - theirs).abs().max())
    relative = float(((mine - theirs).abs() / theirs.clamp_min(1e-6)).median())
    per_head = [
        (spearman(mine[h], theirs[h]), jaccard(mine[h], theirs[h]))
        for h in range(mine.shape[0])
    ]
    rho = sorted(r for r, _ in per_head)
    jac = sorted(j for _, j in per_head)
    print(f"\nlayer {args.layer}, last token, {mine.shape[0]} heads")
    flat = 1 / tokens
    print(f"  peak attention weight        {peak:.4f}   (flat would be {flat:.5f})")
    print(f"  max absolute difference      {difference:.3e}")
    print(f"  median relative difference   {relative:.3e}")
    print(f"  Spearman, min / median       {rho[0]:.4f} / {rho[len(rho) // 2]:.4f}")
    print(f"  top-15% Jaccard, min / med   {jac[0]:.4f} / {jac[len(jac) // 2]:.4f}")
    if peak < 2.0 / tokens:
        print("\n  WARNING: this row is nearly flat, so agreement proves little.")

    out_path = args.out or Path(
        f"results/model_attention_layer{args.layer}.json"
    )
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "record_id": document.record_id,
                "layer": args.layer,
                "tokens": tokens,
                "peak_weight": peak,
                "max_abs_difference": difference,
                "median_relative_difference": relative,
                "spearman_per_head": [r for r, _ in per_head],
                "jaccard_per_head": [j for _, j in per_head],
            },
            indent=1,
        )
    )
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
