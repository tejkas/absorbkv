# SPDX-License-Identifier: Apache-2.0
"""Candidate token-selection signals for CacheBlend on DeepSeek MLA.

CacheBlend ranks tokens for selective recompute by per-head K deviation
(see ``lmcache/v1/compute/blend/blender.py``). MLA never materializes
per-head K, so this module defines four candidate signals computable from
the low-rank latent, plus the reference signal used to validate them.

Every signal is a function of the *deviation* between the cached and the
freshly computed tensors. Because RoPE is orthogonal, that deviation is
invariant to any rotation applied identically to both sides, so all inputs
here are pre-RoPE and this module is free of positional concerns.
"""
from dataclasses import dataclass
import torch

def extract_w_uk(
    kv_b_proj_weight: torch.Tensor,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
) -> torch.Tensor:
    """Split the K up-projection out of MLA's packed ``kv_b_proj`` weight.

    The weight is packed head-major: row block ``h`` holds
    ``qk_nope_head_dim`` rows of ``W^UK`` followed by ``v_head_dim`` rows
    of ``W^UV``.

    Args:
        kv_b_proj_weight: Packed weight of shape
            ``[num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]``.
        num_heads: Number of attention heads on this rank.
        qk_nope_head_dim: Per-head width of the non-RoPE key component.
        v_head_dim: Per-head width of the value.

    Returns:
        ``W^UK`` of shape ``[num_heads * qk_nope_head_dim, kv_lora_rank]``,
        contiguous, in head-major row order.

    Raises:
        ValueError: If the weight's leading dimension is inconsistent with
            the supplied head count and per-head dims.
    """
    expected_rows = num_heads * (qk_nope_head_dim + v_head_dim)
    if kv_b_proj_weight.shape[0] != expected_rows:
        raise ValueError(
            f"kv_b_proj has {kv_b_proj_weight.shape[0]} rows, expected "
            f"{expected_rows} for num_heads={num_heads}, "
            f"qk_nope_head_dim={qk_nope_head_dim}, v_head_dim={v_head_dim}"
        )

    kv_lora_rank = kv_b_proj_weight.shape[1]
    per_head = kv_b_proj_weight.view(
        num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank
    )
    return per_head[:, :qk_nope_head_dim, :].contiguous().view(-1, kv_lora_rank)

def build_gram_matrix(
        projection_weight: torch.Tensor,
        compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Form the Gram matrix ``Wᵀ W`` of an MLA up-projection weight.

    Both the K-only and the joint K+V signals are quadratic forms in the
    latent deviation; this builds the matrix that defines them. Pass the
    output of :func:`extract_w_uk` for the K-only matrix ``M_K``, or the
    unmodified ``kv_b_proj`` weight for the joint matrix ``M_KV``.

    The result is ``[kv_lora_rank, kv_lora_rank]`` however tall the input
    is, which is why the joint signal costs no more than the K-only one.

    Args:
        projection_weight: Up-projection of shape
        ``[out_features, kv_lora_rank]``.
        compute_dtype: Precision in which the product is accumulated. A
        Gram matrix squares the condition number of its input, so
        reduced-precision accumulation destroys the small
        eigendirections the signal depends on.

    Returns:
        Symmetric positive semi-definite matrix of shape
        ``[kv_lora_rank, kv_lora_rank]`` in ``compute_dtype``.

    Raises:
        ValueError: If ``compute_dtype`` is narrower than float32.
    """
    if compute_dtype not in (torch.float32, torch.float64):
        raise ValueError(
             f"compute_dtype must be float32 or float64, got {compute_dtype}; "
             "a Gram matrix squares its input's condition number and is not "
             "safe to accumulate in reduced precision"
        )

    weight = projection_weight.to(compute_dtype)
    return weight.T @ weight

def quadratic_form(delta: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
    """
    Evaluate ``delta_t^T G delta_t`` independently for every row of ``delta``.

    Computed as a row-wise contraction rather than ``delta @ G @ delta.T``:
    the intermediate is ``[num_tokens, rank]`` and the reduction runs over
    the feature axis, so nothing of size ``num_tokens ** 2`` is ever
    materialized.

    Args:
        delta: Per-token latent deviations of shape ``[num_tokens, rank]``.
        gram: Symmetric positive semi-definite matrix of shape
            ``[rank, rank]``, as returned by :func:`build_gram_matrix`.

    Returns:
        Scores of shape ``[num_tokens]`` in ``gram``'s dtype. Non-negative
        in exact arithmetic; not clamped, so rounding may drive very small
        values slightly below zero.

    Raises:
        ValueError: If ``gram`` is not square, or if its rank does not
            match the trailing dimension of ``delta``.
    """
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"gram must be a square matrix, got {tuple(gram.shape)}")
    if delta.shape[-1] != gram.shape[0]:
        raise ValueError(
            f"delta has trailing dimension {delta.shape[-1]} but gram is "
            f"{tuple(gram.shape)}; the two must agree"
        )
    promoted = delta.to(gram.dtype)
    return torch.sum((promoted @ gram) * promoted, dim=-1)

@dataclass(frozen=True)
class LatentSignals:
    """
    The four latent-space candidate signals for one check layer.

    Every field is a per-token score of shape ``[num_tokens]``, larger
    meaning the cached KV deviates further from a fresh forward pass.
    CacheBlend consumes only the ranking, so scores are comparable within
    a signal but carry no meaning across signals.

    Attributes:
        exact: ``dc^T M_K dc + H * ||dk_rope||^2``. Equals the true
            per-head K deviation; validated against
            :func:`reference_signal`.
        kv: As ``exact`` but built on the joint Gram matrix, scoring the
            combined K and V deviation for the same cost.
        row: ``||dc||^2 + ||dk_rope||^2`` -- the stored latent row
            differenced as-is, with no per-head weighting.
        naive: ``||dc||^2`` -- content only, discarding the decoupled
            RoPE component entirely.
    """
    exact: torch.Tensor
    kv: torch.Tensor
    row: torch.Tensor
    naive: torch.Tensor

def reference_signal(delta_k: torch.Tensor) -> torch.Tensor:
    """
    Sum the squared per-head K deviation -- the signal CacheBlend uses today.

    A deliberate transcription of ``blender.py``'s
    ``torch.sum((k - old_k) ** 2, dim=[1])`` over materialized per-head
    keys. Its only job is to validate the latent signals, so it stays
    literal: any cleverness here would weaken it as ground truth.

    Args:
        delta_k: Per-head key deviations of shape ``[num_tokens,
            num_heads, qk_nope_head_dim + qk_rope_head_dim]``, with the
            shared RoPE component already broadcast into every head,
            exactly as attention assembles it.

    Returns:
        Scores of shape ``[num_tokens]`` in float32.

    Raises:
        ValueError: If ``delta_k`` is not three-dimensional.
    """
    if delta_k.ndim != 3:
        raise ValueError(
            f"delta_k must be [num_tokens, num_heads, head_dim], got "
            f"{tuple(delta_k.shape)}"
        )
    return torch.sum(delta_k.to(torch.float32) ** 2, dim=[1, 2])

def compute_latent_signals(
    delta_latent: torch.Tensor,
    delta_k_rope: torch.Tensor,
    gram_k: torch.Tensor,
    gram_kv: torch.Tensor,
    num_heads: int,
) -> LatentSignals:
    """Compute every latent-space candidate signal for one check layer.

    Args:
        delta_latent: Fresh minus cached post-``kv_a_layernorm`` latent,
            shape ``[num_tokens, kv_lora_rank]``.
        delta_k_rope: Fresh minus cached decoupled RoPE key, shape
            ``[num_tokens, qk_rope_head_dim]``. Pre-RoPE on both sides or
            post-RoPE on both -- rotation is orthogonal and cancels from
            the difference either way, but the two sides must match.
        gram_k: ``(W^UK)^T W^UK``, from :func:`build_gram_matrix`.
        gram_kv: ``kv_b_proj^T kv_b_proj``, from :func:`build_gram_matrix`.
        num_heads: Attention heads on this rank -- the multiplicity with
            which the shared RoPE key enters the per-head sum.

    Returns:
        A :class:`LatentSignals` holding four ``[num_tokens]`` score
        tensors in ``gram_k``'s dtype.

    Raises:
        ValueError: If ``num_heads`` is not positive, or if the two
            deviation tensors disagree on token count.
    """
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if delta_latent.shape[0] != delta_k_rope.shape[0]:
        raise ValueError(
            f"token count mismatch: delta_latent has {delta_latent.shape[0]} "
            f"rows, delta_k_rope has {delta_k_rope.shape[0]}"
        )

    dtype = gram_k.dtype
    rope_energy = torch.sum(delta_k_rope.to(dtype) ** 2, dim=-1)
    content_energy = torch.sum(delta_latent.to(dtype) ** 2, dim=-1)
    weighted_rope = num_heads * rope_energy

    return LatentSignals(
        exact=quadratic_form(delta_latent, gram_k) + weighted_rope,
        kv=quadratic_form(delta_latent, gram_kv) + weighted_rope,
        row=content_energy + rope_energy,
        naive=content_energy,
    )
