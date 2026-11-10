# SPDX-License-Identifier: Apache-2.0
"""Query-key scores computed from MLA's latent, without rebuilding keys.

A relevance method ranks cached tokens by ``q . k_t``. On MLA that score
splits in two, and only the first half absorbs:

    q . k_t  =  q' . c_t  +  q^R . k^R_t,       q' = W^UK^T q

The first term scores the 512-wide latent directly and needs no per-head
key. The second cannot absorb: RoPE puts a rotation that depends on the
*distance* between query and key in between, and a rotation by a distance
cannot be folded into a static weight. That is why MLA carries a separate
64-wide component at all.

The rotation is also why ``q`` cannot be used as captured. A hook on
``q_proj`` sees it before any rotation, while the cache stores ``k^R``
after one. Scoring those together computes the score for a query at
position 0 -- off by a rotation of however far the query actually sits
from the start, which scrambles the high-frequency dimensions. The query
is rotated here, at analysis time, using the positions the capture
stored.

Staleness plays no part in any of this: these scores come from the truth
pass alone.
"""

# Standard
from dataclasses import dataclass

# Third Party
import torch
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2RotaryEmbedding,
    apply_rotary_emb,
)

# First Party
from absorbkv.concentration import absorbed_query


@dataclass(frozen=True)
class Scores:
    """Exact attention scores and weights for one layer.

    Attributes:
        logits: ``q . k`` before the softmax, shape ``[num_query_tokens,
            num_heads, num_context_tokens]``.
        weights: Softmax over the context, same shape. Sums to one along
            the last axis, so a selection's share of it is the fraction of
            attention that selection captures.
    """

    logits: torch.Tensor
    weights: torch.Tensor


def rotate_query(
    query_rope: torch.Tensor, position_ids: torch.Tensor, config
) -> torch.Tensor:
    """Apply RoPE to the captured query, which was stored before rotation.

    Uses the model's own rotary module and rotation function rather than
    reimplementing them, so YaRN's frequency scaling comes along for free.

    Args:
        query_rope: RoPE half of the query, shape ``[num_tokens,
            num_heads, qk_rope_head_dim]``.
        position_ids: Absolute position of each of those tokens, shape
            ``[num_tokens]``.
        config: The model config the capture was taken with.

    Returns:
        The rotated query, same shape, float32.
    """
    rotary = DeepseekV2RotaryEmbedding(config)
    positions = position_ids.unsqueeze(0)
    freqs = rotary(torch.zeros(1, dtype=torch.float32), positions)

    # apply_rotary_emb wants [batch, heads, seq, dim] and rotates two
    # tensors against one set of positions; we only need the first.
    shaped = query_rope.to(torch.float32).permute(1, 0, 2).unsqueeze(0)
    rotated, _ = apply_rotary_emb(shaped, shaped, freqs)
    return rotated[0].permute(1, 0, 2)


def exact_scores(
    query: torch.Tensor,
    query_positions: torch.Tensor,
    latent: torch.Tensor,
    k_rope: torch.Tensor,
    w_uk: torch.Tensor,
    config,
) -> Scores:
    """Score every context token against every query token, from the latent.

    Args:
        query: Captured query for one layer, pre-RoPE, shape
            ``[num_query_tokens, num_heads, qk_head_dim]``.
        query_positions: Absolute positions of the query tokens.
        latent: Context latents for that layer, ``[num_context_tokens,
            kv_lora_rank]``.
        k_rope: Context RoPE keys as cached, i.e. already rotated,
            ``[num_context_tokens, qk_rope_head_dim]``.
        w_uk: ``W^UK`` for that layer, ``[num_heads * qk_nope_head_dim,
            kv_lora_rank]``.
        config: The model config, for head dims and the softmax scale.

    Returns:
        A :class:`Scores` with logits and softmax weights.

    Raises:
        ValueError: If the query's trailing dimension does not match the
            config's head dims.
    """
    nope, rope = config.qk_nope_head_dim, config.qk_rope_head_dim
    if query.shape[-1] != nope + rope:
        raise ValueError(
            f"query has head dim {query.shape[-1]}, expected {nope + rope}"
        )

    absorbed = absorbed_query(query[..., :nope], w_uk)
    content = torch.einsum("qhr,tr->qht", absorbed, latent.to(torch.float32))

    rotated = rotate_query(query[..., nope:], query_positions, config)
    positional = torch.einsum(
        "qhd,td->qht", rotated, k_rope.to(torch.float32)
    )

    logits = (content + positional) * softmax_scale(config)
    return Scores(logits=logits, weights=logits.softmax(dim=-1))


def softmax_scale(config) -> float:
    """The model's own attention scale, YaRN adjustment included.

    Reads it off a freshly built attention module rather than
    recomputing, since the adjustment is a function of the rope settings
    and is easy to get subtly wrong.
    """
    # Third Party
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
        DeepseekV2Attention,
    )

    return DeepseekV2Attention(config, layer_idx=0).scaling


def captured_mass(weights: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """Share of the true attention held by the selected context tokens.

    The quantity the whole relevance measurement turns on: a token that
    can be ignored at decode is one carrying negligible weight, so the
    share a selection captures says how much of the attention survives
    restricting to it. One means the selection kept everything that
    mattered; ``k / num_context_tokens`` is what picking at random gives.

    Args:
        weights: Softmax weights, ``[num_query_tokens, num_heads,
            num_context_tokens]``.
        selected: Indices of the selected context tokens, either
            ``[num_selected]`` shared by every query and head, or
            ``[num_query_tokens, num_selected]``.

    Returns:
        ``[num_query_tokens, num_heads]`` shares in ``[0, 1]``.

    Raises:
        ValueError: If ``selected`` is neither one- nor two-dimensional.
    """
    if selected.ndim == 1:
        return weights[..., selected].sum(dim=-1)
    if selected.ndim == 2:
        expanded = selected.unsqueeze(1).expand(-1, weights.shape[1], -1)
        return weights.gather(-1, expanded).sum(dim=-1)
    raise ValueError(
        f"selected must be [num_selected] or [num_query_tokens, "
        f"num_selected], got {tuple(selected.shape)}"
    )
