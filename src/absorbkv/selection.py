# SPDX-License-Identifier: Apache-2.0
"""RocketKV's decode-time token selection, translated to MLA's latent.

Their pipeline scores pages of keys cheaply, then attends exactly to the
few tokens those scores rank highest. Six steps, of which only the third
is in question here:

1. flip signs per coordinate so the query is non-negative, and flip the
   keys to match, which leaves the product unchanged;
2. take an elementwise max over each page of keys -- now an upper bound;
3. keep only the ``r`` coordinates where the query is largest;
4. score pages with what survives;
5. rescale by the share of query magnitude kept, and softmax;
6. take the top ``k`` tokens.

On MLA the key is the shared latent, so one page summary serves every
head, where GQA writes one per KV head. The sign in step 1 is therefore
shared across heads too: a per-head sign would need a per-head page
summary and give back the advantage. That weakens the bound, and whether
it costs anything is the measurement.

Step 3 keeps the RoPE coordinates always. They are 64 against the
latent's 512, they carry the positional term that cannot absorb, and
their code has no analogue to truncate since GQA keys are not split.

The reference implementation these steps follow is in
``papers/rocketkv/code/``, under its own licence.
"""

# Standard
import math
from dataclasses import dataclass

# Third Party
import torch


@dataclass(frozen=True)
class Budget:
    """What an overall compression ratio implies, by their §3.6 derivation.

    The user sets one number, ``c = sequence_length / token_budget``, and
    it is split across both stages and both dimensions::

        r          = min(0.2 + 0.06 * log2(c), 0.8)   split factor
        stage one  = c ** r                           permanent eviction
        stage two  = c ** (1 - r)                     the sparse attention
        page_size  = ceil(sqrt(stage two))            sequence dimension
        kept       = page_size / stage two            head dimension

    Because ``r`` grows with ``c``, stage two always lands near 4-6x
    however aggressive the overall target, so the page size stays at 2-3
    and roughly half the coordinates survive. Permanent eviction does the
    heavy lifting. Their worked example: ``c = 64`` gives ``r = 0.56``,
    10.3x in stage one and 6.2x in stage two, split into 3x along the
    sequence and 2.1x along the head dimension.

    Attributes:
        overall: ``c``, the user-set ratio.
        eviction_ratio: ``c ** r``, how much stage one discards.
        sparse_ratio: ``c ** (1 - r)``, what stage two is left to do.
        page_size: Tokens summarized together.
        kept_fraction: Share of coordinates step 3 keeps.
    """

    overall: float
    eviction_ratio: float
    sparse_ratio: float
    page_size: int
    kept_fraction: float

    @classmethod
    def of(cls, overall: float) -> "Budget":
        split = min(0.2 + 0.06 * math.log2(overall), 0.8)
        sparse = overall ** (1 - split)
        page_size = math.ceil(math.sqrt(sparse))
        page_size = page_size if page_size <= sparse else 1
        return cls(
            overall=overall,
            eviction_ratio=overall**split,
            sparse_ratio=sparse,
            page_size=page_size,
            kept_fraction=page_size / sparse,
        )


def select_latent(
    query_content: torch.Tensor,
    query_rope: torch.Tensor,
    latent: torch.Tensor,
    k_rope: torch.Tensor,
    topk: int,
    compression_ratio: float,
    reduce_head_dim: bool = True,
    force_scale: torch.Tensor | None = None,
    force_kept: float | None = None,
    truncate_rope: bool = False,
) -> torch.Tensor:
    """Select context tokens for one query token, scoring the latent.

    Args:
        query_content: Absorbed query ``q'``, ``[num_heads, kv_lora_rank]``.
        query_rope: Rotated RoPE query, ``[num_heads, qk_rope_head_dim]``.
            May be width zero, which makes this their pipeline exactly.
        latent: Context latents, ``[num_tokens, kv_lora_rank]``.
        k_rope: Context RoPE keys as cached, ``[num_tokens,
            qk_rope_head_dim]``.
        topk: How many tokens to select.
        compression_ratio: The **sparse-attention stage's** ratio, their
            ``c ** (1 - r)``, exactly as their forward takes it -- not the
            overall ratio. Use :meth:`Budget.of` to derive it.
        reduce_head_dim: False keeps every coordinate, which isolates what
            step 3 costs when compared against True.
        force_scale: Overrides the step-5 rescaling. Switching off the
            reduction also changes that rescaling, since it depends on how
            much query magnitude survives -- and the rescaling is a softmax
            temperature, which changes the ranking once heads are averaged.
            Passing the reduced run's scale into an unreduced run separates
            "which coordinates" from "how sharp".
        force_kept: Overrides the share of coordinates step 3 keeps, while
            leaving the page size at whatever ``compression_ratio``
            derives. Their derivation ties the two together; on MLA the
            traffic accounting says the kept share has to drop on its own
            for step 2 to stay inside its budget, so the two have to be
            separable to measure what that costs.
        truncate_rope: True puts the RoPE coordinates into the same pool as
            the latent's, so step 3 may drop them too. The default keeps
            them whole, which is the faithful translation; this variant
            exists because truncating them is the cheaper way to meet the
            traffic budget and the question is what it costs.

    Returns:
        Selected token indices, ``[min(topk, num_tokens)]``.

    Raises:
        ValueError: If the query and key widths disagree.
    """
    if query_content.shape[-1] != latent.shape[-1]:
        raise ValueError(
            f"query_content is {query_content.shape[-1]} wide but latent is "
            f"{latent.shape[-1]}"
        )
    if query_rope.shape[-1] != k_rope.shape[-1]:
        raise ValueError(
            f"query_rope is {query_rope.shape[-1]} wide but k_rope is "
            f"{k_rope.shape[-1]}"
        )

    page_size = math.ceil(math.sqrt(compression_ratio))
    page_size = page_size if page_size <= compression_ratio else 1
    kept_fraction = page_size / compression_ratio
    rank = latent.shape[-1]
    num_tokens = latent.shape[0]

    query = torch.cat([query_content, query_rope], dim=-1).to(torch.float32)
    key = torch.cat([latent, k_rope], dim=-1).to(torch.float32)

    # 1. one sign per coordinate, shared by every head
    sign = torch.where(query.sum(dim=0) > 0, 1.0, -1.0)
    positive_query = query * sign
    flipped_key = key * sign

    # 2. elementwise max over each page, after padding to a whole number
    padding = (-num_tokens) % page_size
    if padding:
        flipped_key = torch.cat(
            [
                flipped_key,
                torch.full(
                    (padding, flipped_key.shape[-1]),
                    torch.finfo(flipped_key.dtype).min,
                ),
            ]
        )
    pages = flipped_key.view(-1, page_size, flipped_key.shape[-1]).amax(dim=1)

    # 3. the coordinates the query is largest on, RoPE always kept
    #    unless truncate_rope throws them into the same pool
    if reduce_head_dim:
        fraction = kept_fraction if force_kept is None else force_kept
        if truncate_rope:
            kept = max(1, int(query.shape[-1] * fraction))
            coordinates = query.abs().mean(dim=0).topk(kept).indices
        else:
            kept = max(1, int(rank * fraction))
            chosen = query[:, :rank].abs().mean(dim=0).topk(kept).indices
            coordinates = torch.cat(
                [chosen, torch.arange(rank, query.shape[-1])]
            )
    else:
        coordinates = torch.arange(query.shape[-1])

    # 4-5. score pages on what survives, rescale by the magnitude kept
    page_scores = positive_query[:, coordinates] @ pages[:, coordinates].T
    scale = (
        force_scale
        if force_scale is not None
        else torch.sqrt(
            query.shape[-1]
            * positive_query[:, coordinates].abs().sum(dim=-1, keepdim=True)
            / query.abs().sum(dim=-1, keepdim=True)
        )
    )
    weights = torch.softmax(page_scores / scale, dim=-1)

    # 6. spread page scores back over their tokens, then take the top k
    per_token = weights.repeat_interleave(page_size, dim=-1)[:, :num_tokens]
    return per_token.mean(dim=0).topk(min(topk, num_tokens)).indices
