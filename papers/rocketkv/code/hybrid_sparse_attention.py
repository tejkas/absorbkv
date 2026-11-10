# Lifted from NVlabs/RocketKV, pipeline/inf_stream_llm/inf_llm/attention/rocket.py
# (the `else:` branch of `rocket_forward`, "hybrid sparse attention", which runs
# at decode when len_q == 1). Covered by the NVIDIA licence in ../LICENSE, whose
# non-commercial restriction propagates to this file. Not Apache-2.0 like the
# rest of this repository, and kept out of `src/` for that reason.
#
# Modified only to be callable outside their forward pass:
#   * returns the selected token indices instead of computing the attention
#     output, since selection is what is being measured;
#   * takes tensors as arguments rather than reading them from `self`;
#   * drops the `_gather` helper's reuse for V, which is not needed.
# The arithmetic of steps 1-6 is unchanged, including `absQ.mean(dim=2)`,
# `chunk_size = ceil(sqrt(compression_ratio))` and the sqrt rescaling.

import math

import torch


def _gather(t: torch.Tensor, dim: int, i: torch.Tensor) -> torch.Tensor:
    dim += (dim < 0) * t.ndim
    return t.gather(dim, i.expand(*t.shape[:dim], i.shape[dim], *t.shape[dim + 1 :]))


def _scaled_softmax(x: torch.Tensor, divscale, dim: int) -> torch.Tensor:
    return torch.softmax(x / divscale, dim=dim)


def select_tokens(
    Q: torch.Tensor,
    K: torch.Tensor,
    topk: int,
    compression_ratio: float,
    reduce_head_dim: bool = True,
) -> torch.Tensor:
    """Their token selection, returning indices instead of attention output.

    Args:
        Q: ``[batch, num_heads_kv, group, len_q, dim_head]``.
        K: ``[batch, num_heads_kv, 1, len_k, dim_head]``.
        topk: How many tokens to select.
        compression_ratio: Sets both the page size and the kept coordinates.
        reduce_head_dim: False keeps every coordinate, isolating what the
            head-dimension reduction (step 3) costs. Not part of their code;
            added for the ablation.

    Returns:
        ``[batch, num_heads_kv, 1, len_q, k]`` selected token indices.
    """
    len_k = K.shape[-2]
    len_q = Q.shape[-2]
    dist = (
        torch.arange(0, len_q, device=Q.device)[:, None]
        - torch.arange(0, len_k, device=Q.device)[None, :]
        + len_k
        - len_q
    )
    attention_mask = (dist >= 0).view(1, 1, len_q, len_k)

    sign = (Q.sum(dim=2, keepdim=True) > 0) + (~(Q.sum(dim=2, keepdim=True) > 0)) * -1
    max_key = K * sign
    positive_query = Q * sign

    chunk_size = math.ceil(math.sqrt(compression_ratio))
    chunk_size = chunk_size if chunk_size <= compression_ratio else 1

    padding_length = chunk_size - ((len_k - 1) % chunk_size + 1)
    max_key = torch.cat(
        [
            max_key,
            torch.ones(
                (
                    max_key.shape[0],
                    max_key.shape[1],
                    max_key.shape[2],
                    padding_length,
                    max_key.shape[4],
                ),
                device=max_key.device,
                dtype=max_key.dtype,
            )
            * torch.tensor(torch.finfo(max_key.dtype).min),
        ],
        dim=-2,
    )
    chunk_max_key = max_key.reshape(
        max_key.shape[0],
        max_key.shape[1],
        max_key.shape[2],
        max_key.shape[3] // chunk_size,
        chunk_size,
        max_key.shape[4],
    ).amax(dim=-2)

    absQ = torch.abs(Q)
    if reduce_head_dim:
        r = int(Q.shape[-1] * chunk_size / compression_ratio)
        r = max(1, r)
        i1 = torch.topk(absQ.mean(dim=2, keepdim=True), r, dim=-1).indices
        Q_hat = _gather(positive_query, -1, i1)
        K_hat = _gather(chunk_max_key, -1, i1)
    else:
        Q_hat, K_hat = positive_query, chunk_max_key

    QK_hat = Q_hat @ K_hat.transpose(-1, -2)
    QK_hat = (
        QK_hat.unsqueeze(-1)
        .repeat(1, 1, 1, 1, 1, chunk_size)
        .reshape(
            QK_hat.shape[0], QK_hat.shape[1], QK_hat.shape[2], QK_hat.shape[3], -1
        )[:, :, :, :, :len_k]
    )
    masked_QK_hat = torch.where(attention_mask.unsqueeze(2), QK_hat, float("-inf"))
    scale = torch.sqrt(
        Q.shape[-1]
        * torch.abs(Q_hat).sum(dim=-1, keepdim=True)
        / absQ.sum(dim=-1, keepdim=True)
    )
    s_hat = _scaled_softmax(masked_QK_hat, scale, dim=-1)

    k = min(topk, len_k)
    _, i2 = torch.topk(s_hat.mean(dim=2, keepdim=True), k, dim=-1)
    return i2
