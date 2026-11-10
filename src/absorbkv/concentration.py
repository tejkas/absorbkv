# SPDX-License-Identifier: Apache-2.0
"""How concentrated a query is, and what absorption does to it.

Token-selection methods that keep only the largest coordinates of a query
are betting that most of the query's magnitude sits in a few of them. That
bet is made in whatever basis the query happens to live in. On MLA the
query that scores against the stored latent is the absorbed one,
``q' = W^UK^T q``, which is 512-wide instead of 128 and sits in a basis the
model never trained a query to be sparse in.

Everything here measures the quantity such a method actually truncates:

* magnitudes are **absolute values**, not squares, because the selection is
  ``topk(|q|)`` and the compensation that follows it is a ratio of absolute
  sums;
* heads are averaged **within a group** first, because one coordinate set
  serves the whole group, and on MLA that group is every head;
* the result is a curve over kept fractions rather than one number at one
  cut, since the fraction is derived from a budget and varies.
"""

# Standard
from dataclasses import dataclass

# Third Party
import torch

# First Party
from absorbkv.signals import extract_w_uk

FRACTIONS = (0.04, 0.0625, 0.094, 0.125, 0.1875, 0.25)


@dataclass(frozen=True)
class Curve:
    """Top-k absolute mass against the fraction of coordinates kept.

    Attributes:
        fractions: Fractions of coordinates kept, ascending.
        mass: Fraction of total absolute magnitude held by that many
            largest coordinates, same length as ``fractions``.
        dimension: How many coordinates the vector had, since the same
            fraction means a different count for ``q`` and ``q'``.
    """

    fractions: tuple[float, ...]
    mass: torch.Tensor
    dimension: int


def absorbed_query(query_nope: torch.Tensor, w_uk: torch.Tensor) -> torch.Tensor:
    """Map per-head queries into the latent space: ``q'_i = W^UK_i^T q_i``.

    Args:
        query_nope: Non-RoPE half of the query, shape ``[num_tokens,
            num_heads, qk_nope_head_dim]``.
        w_uk: The K up-projection for this layer, shape ``[num_heads *
            qk_nope_head_dim, kv_lora_rank]``, as :func:`extract_w_uk`
            returns it.

    Returns:
        ``[num_tokens, num_heads, kv_lora_rank]`` in float32.

    Raises:
        ValueError: If the head count implied by ``w_uk`` does not divide
            evenly, or disagrees with ``query_nope``.
    """
    num_tokens, num_heads, head_dim = query_nope.shape
    if w_uk.shape[0] != num_heads * head_dim:
        raise ValueError(
            f"w_uk has {w_uk.shape[0]} rows, expected "
            f"{num_heads * head_dim} for {num_heads} heads of {head_dim}"
        )
    per_head = w_uk.to(torch.float32).view(num_heads, head_dim, w_uk.shape[1])
    return torch.einsum("thd,hdr->thr", query_nope.to(torch.float32), per_head)


def group_mean_abs(query: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Average ``|q|`` over the query heads that share one coordinate set.

    A GQA model picks one set per KV head, averaging the query heads in
    that group. MLA has a single latent serving every head, so there is
    one group and every head is averaged together -- a stronger
    assumption, since each head's ``q'`` sits in its own basis.

    Args:
        query: ``[num_tokens, num_heads, dim]``.
        num_groups: How many coordinate sets the layer uses. Query heads
            are grouped contiguously, matching how ``repeat_kv`` expands
            KV heads.

    Returns:
        ``[num_tokens, num_groups, dim]`` of non-negative values.

    Raises:
        ValueError: If the heads do not divide evenly into the groups.
    """
    num_tokens, num_heads, dim = query.shape
    if num_heads % num_groups:
        raise ValueError(
            f"{num_heads} heads do not divide into {num_groups} groups"
        )
    grouped = query.to(torch.float32).abs().view(
        num_tokens, num_groups, num_heads // num_groups, dim
    )
    return grouped.mean(dim=2)


def topk_mass(values: torch.Tensor, fractions=FRACTIONS) -> Curve:
    """Fraction of total absolute magnitude held by the largest coordinates.

    A flat vector scores exactly ``f`` at fraction ``f``; a vector that is
    one coordinate scores ~1 everywhere. The comparison between ``q`` and
    ``q'`` has to be at matched *fractions*, since they differ in
    dimension, and a flat vector then scores the same in both.

    Args:
        values: Non-negative magnitudes, ``[..., dim]``.
        fractions: Fractions of coordinates to keep, ascending.

    Returns:
        A :class:`Curve` whose ``mass`` is averaged over every leading
        axis.

    Raises:
        ValueError: If any fraction is outside ``(0, 1]``.
    """
    if any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError(f"fractions must lie in (0, 1], got {fractions}")

    dimension = values.shape[-1]
    ordered = values.to(torch.float32).sort(dim=-1, descending=True).values
    cumulative = ordered.cumsum(dim=-1)
    total = cumulative[..., -1].clamp_min(torch.finfo(torch.float32).tiny)

    mass = []
    for fraction in fractions:
        kept = max(1, round(fraction * dimension))
        mass.append((cumulative[..., kept - 1] / total).mean())
    return Curve(
        fractions=tuple(fractions),
        mass=torch.stack(mass),
        dimension=dimension,
    )


def head_agreement(values: torch.Tensor, fraction: float) -> torch.Tensor:
    """How much two heads agree on which coordinates matter.

    Mean pairwise Jaccard between the heads' top-coordinate sets, per
    token. This is the assumption a shared coordinate set encodes: if
    heads disagree, averaging them picks a set that suits none of them.

    Args:
        values: Non-negative per-head magnitudes, ``[num_tokens,
            num_heads, dim]``.
        fraction: Fraction of coordinates in each head's set.

    Returns:
        Scalar: mean over tokens and head pairs. 1.0 is total agreement;
        ``fraction`` is what independent random sets score.

    Raises:
        ValueError: If there are fewer than two heads.
    """
    num_tokens, num_heads, dimension = values.shape
    if num_heads < 2:
        raise ValueError("head agreement needs at least two heads")

    kept = max(1, round(fraction * dimension))
    top = values.to(torch.float32).topk(kept, dim=-1).indices
    membership = torch.zeros(
        num_tokens, num_heads, dimension, dtype=torch.bool
    ).scatter_(-1, top, True)

    overlaps = []
    for first in range(num_heads):
        for second in range(first + 1, num_heads):
            both = (membership[:, first] & membership[:, second]).sum(-1)
            either = (membership[:, first] | membership[:, second]).sum(-1)
            overlaps.append(both.float() / either.float())
    return torch.stack(overlaps).mean()


def layer_w_uk(
    kv_b_proj: torch.Tensor, num_heads: int, qk_nope_head_dim: int, v_head_dim: int
) -> torch.Tensor:
    """``W^UK`` for one layer, from that layer's packed ``kv_b_proj``."""
    return extract_w_uk(kv_b_proj, num_heads, qk_nope_head_dim, v_head_dim)
