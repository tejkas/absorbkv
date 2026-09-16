# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the latent identity and the functions it rests on.

The identity
    sum_i ||dk_i||^2  =  dc^T M dc  +  H ||dk_rope||^2
is pure tensor algebra, so it is provable on a laptop against a synthetic
weight before any GPU time is spent. If these fail, no capture is worth
running.

Tolerance is asserted on a *relative residual*, ``max|a - b| / max|b|``,
rather than elementwise ``allclose``. Per-token scores span orders of
magnitude, and an elementwise relative tolerance is meaningless on the
near-zero entries while being blind to absolute error on the large ones --
and it is the large ones that decide a top-k ranking. This is the same
metric ``report.py`` will use on real captures.
"""

# Standard
import dataclasses

# Third Party
import pytest
import torch

# First Party
from absorbkv.signals import (
    LatentSignals,
    build_gram_matrix,
    compute_latent_signals,
    extract_w_uk,
    quadratic_form,
    reference_signal,
)

NUM_HEADS = 4
QK_NOPE = 8
# Deliberately unequal to QK_NOPE: with equal widths, an extract_w_uk that
# confused the K and V block widths would still slice the right rows.
V_HEAD = 6
KV_LORA = 16
QK_ROPE = 4
NUM_TOKENS = 32


def relative_residual(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Max absolute deviation, scaled by the largest expected magnitude."""
    scale = expected.abs().max()
    if scale == 0:
        return float(actual.abs().max())
    return float((actual - expected).abs().max() / scale)


@pytest.fixture
def weights() -> dict[str, torch.Tensor]:
    """A synthetic MLA up-projection, packed head-major.

    Row block ``h`` holds ``QK_NOPE`` rows of ``W^UK`` then ``V_HEAD`` rows
    of ``W^UV``, matching what the real ``kv_b_proj`` is asserted to be.
    """
    torch.manual_seed(0)
    w_uk = torch.randn(NUM_HEADS, QK_NOPE, KV_LORA, dtype=torch.float64)
    w_uv = torch.randn(NUM_HEADS, V_HEAD, KV_LORA, dtype=torch.float64)
    kv_b_proj = torch.cat([w_uk, w_uv], dim=1).reshape(-1, KV_LORA)
    return {
        "w_uk": w_uk.reshape(-1, KV_LORA),
        "w_uv": w_uv.reshape(-1, KV_LORA),
        "kv_b_proj": kv_b_proj,
    }


@pytest.fixture
def deltas() -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    return {
        "latent": torch.randn(NUM_TOKENS, KV_LORA, dtype=torch.float64),
        "k_rope": torch.randn(NUM_TOKENS, QK_ROPE, dtype=torch.float64),
    }


def assemble_per_head_delta_k(
    delta_latent: torch.Tensor,
    delta_k_rope: torch.Tensor,
    w_uk: torch.Tensor,
) -> torch.Tensor:
    """Build the per-head key deviation attention would actually see.

    The content half is head-specific; the RoPE half is one vector per
    token broadcast into every head. That broadcast is the physical origin
    of the factor ``H`` in the identity.
    """
    per_head_w = w_uk.reshape(NUM_HEADS, QK_NOPE, KV_LORA)
    content = torch.einsum("hdr,tr->thd", per_head_w, delta_latent)
    rope = delta_k_rope.unsqueeze(1).expand(-1, NUM_HEADS, -1)
    return torch.cat([content, rope], dim=-1)


# --------------------------------------------------------------------------
# extract_w_uk
# --------------------------------------------------------------------------


def test_extract_w_uk_recovers_the_k_rows(weights):
    extracted = extract_w_uk(weights["kv_b_proj"], NUM_HEADS, QK_NOPE, V_HEAD)
    assert extracted.shape == (NUM_HEADS * QK_NOPE, KV_LORA)
    torch.testing.assert_close(extracted, weights["w_uk"])


def test_extract_w_uk_result_is_contiguous(weights):
    extracted = extract_w_uk(weights["kv_b_proj"], NUM_HEADS, QK_NOPE, V_HEAD)
    assert extracted.is_contiguous()


def test_extract_w_uk_rejects_inconsistent_head_count(weights):
    with pytest.raises(ValueError, match="rows, expected"):
        extract_w_uk(weights["kv_b_proj"], NUM_HEADS + 1, QK_NOPE, V_HEAD)


# --------------------------------------------------------------------------
# build_gram_matrix
# --------------------------------------------------------------------------


def test_gram_is_square_and_rank_sized(weights):
    gram = build_gram_matrix(weights["w_uk"], torch.float64)
    assert gram.shape == (KV_LORA, KV_LORA)


def test_gram_shape_is_independent_of_input_height(weights):
    """The point of the Gram form: cost is capped by rank, not expansion."""
    gram_k = build_gram_matrix(weights["w_uk"], torch.float64)
    gram_kv = build_gram_matrix(weights["kv_b_proj"], torch.float64)
    assert gram_k.shape == gram_kv.shape


def test_gram_is_symmetric_psd(weights):
    gram = build_gram_matrix(weights["w_uk"], torch.float64)
    torch.testing.assert_close(gram, gram.T)
    assert torch.linalg.eigvalsh(gram).min() > -1e-10


def test_gram_rejects_reduced_precision(weights):
    with pytest.raises(ValueError, match="condition number"):
        build_gram_matrix(weights["w_uk"], torch.bfloat16)


def test_gram_is_invariant_to_row_permutation(weights):
    """``B^T B = sum_j b_j b_j^T`` does not care about row order.

    This is why ``S_kv`` is immune to the head-packing assumption that
    ``extract_w_uk`` depends on.
    """
    torch.manual_seed(2)
    shuffled = weights["kv_b_proj"][torch.randperm(weights["kv_b_proj"].shape[0])]
    torch.testing.assert_close(
        build_gram_matrix(shuffled, torch.float64),
        build_gram_matrix(weights["kv_b_proj"], torch.float64),
    )


# --------------------------------------------------------------------------
# quadratic_form
# --------------------------------------------------------------------------


def test_quadratic_form_matches_explicit_loop(weights, deltas):
    gram = build_gram_matrix(weights["w_uk"], torch.float64)
    actual = quadratic_form(deltas["latent"], gram)
    expected = torch.stack([row @ gram @ row for row in deltas["latent"]])
    assert relative_residual(actual, expected) < 1e-12


def test_quadratic_form_avoids_the_token_squared_intermediate(weights, deltas):
    """Equivalent to the diagonal of the full product, without building it."""
    gram = build_gram_matrix(weights["w_uk"], torch.float64)
    delta = deltas["latent"]
    expected = (delta @ gram @ delta.T).diagonal()
    assert relative_residual(quadratic_form(delta, gram), expected) < 1e-12


def test_quadratic_form_rejects_non_square_gram(weights, deltas):
    with pytest.raises(ValueError, match="square"):
        quadratic_form(deltas["latent"], weights["w_uk"])


def test_quadratic_form_rejects_rank_mismatch(deltas):
    gram = torch.eye(KV_LORA + 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="must agree"):
        quadratic_form(deltas["latent"], gram)


# --------------------------------------------------------------------------
# reference_signal
# --------------------------------------------------------------------------


def test_reference_signal_sums_over_heads_and_dims(weights, deltas):
    delta_k = assemble_per_head_delta_k(
        deltas["latent"], deltas["k_rope"], weights["w_uk"]
    )
    actual = reference_signal(delta_k)
    expected = (delta_k.to(torch.float32) ** 2).sum(dim=(1, 2))
    assert actual.shape == (NUM_TOKENS,)
    assert relative_residual(actual, expected) < 1e-6


def test_reference_signal_rejects_flattened_input(weights, deltas):
    delta_k = assemble_per_head_delta_k(
        deltas["latent"], deltas["k_rope"], weights["w_uk"]
    )
    with pytest.raises(ValueError, match="num_tokens, num_heads, head_dim"):
        reference_signal(delta_k.reshape(NUM_TOKENS, -1))


# --------------------------------------------------------------------------
# the identity
# --------------------------------------------------------------------------


def test_identity_holds_in_float64(weights, deltas):
    """sum_i ||dk_i||^2 == dc^T M dc + H ||dk_rope||^2.

    In float64 there is no judgement call about tolerance: the algebra is
    either right or it is not.
    """
    delta_k = assemble_per_head_delta_k(
        deltas["latent"], deltas["k_rope"], weights["w_uk"]
    )
    truth = (delta_k**2).sum(dim=(1, 2))

    gram_k = build_gram_matrix(weights["w_uk"], torch.float64)
    gram_kv = build_gram_matrix(weights["kv_b_proj"], torch.float64)
    signals = compute_latent_signals(
        deltas["latent"], deltas["k_rope"], gram_k, gram_kv, NUM_HEADS
    )

    assert relative_residual(signals.exact, truth) < 1e-12


def test_identity_survives_float32(weights, deltas):
    """Documents the precision the production dtype actually delivers.

    float32 is what ``build_gram_matrix`` defaults to, so this is the
    number a real capture should expect. bfloat16 is refused outright, and
    this test is why.
    """
    delta_k = assemble_per_head_delta_k(
        deltas["latent"], deltas["k_rope"], weights["w_uk"]
    )
    truth = (delta_k**2).sum(dim=(1, 2))

    gram_k = build_gram_matrix(weights["w_uk"], torch.float32)
    gram_kv = build_gram_matrix(weights["kv_b_proj"], torch.float32)
    signals = compute_latent_signals(
        deltas["latent"].to(torch.float32),
        deltas["k_rope"].to(torch.float32),
        gram_k,
        gram_kv,
        NUM_HEADS,
    )

    assert relative_residual(signals.exact.to(torch.float64), truth) < 1e-5


def test_identity_is_sensitive_to_head_count(weights, deltas):
    """A wrong H must not quietly produce a plausible number.

    This is the likeliest way a real run yields a residual that reads as
    broken algebra, so it is asserted rather than assumed.
    """
    delta_k = assemble_per_head_delta_k(
        deltas["latent"], deltas["k_rope"], weights["w_uk"]
    )
    truth = (delta_k**2).sum(dim=(1, 2))
    gram_k = build_gram_matrix(weights["w_uk"], torch.float64)
    gram_kv = build_gram_matrix(weights["kv_b_proj"], torch.float64)

    wrong = compute_latent_signals(
        deltas["latent"], deltas["k_rope"], gram_k, gram_kv, NUM_HEADS + 1
    )
    assert relative_residual(wrong.exact, truth) > 1e-3


def test_s_kv_adds_the_value_deviation(weights, deltas):
    """``S_kv`` should equal ``S_truth`` plus the summed V deviation."""
    delta_k = assemble_per_head_delta_k(
        deltas["latent"], deltas["k_rope"], weights["w_uk"]
    )
    truth = (delta_k**2).sum(dim=(1, 2))
    delta_v = deltas["latent"] @ weights["w_uv"].T
    v_energy = (delta_v**2).sum(dim=-1)

    gram_k = build_gram_matrix(weights["w_uk"], torch.float64)
    gram_kv = build_gram_matrix(weights["kv_b_proj"], torch.float64)
    signals = compute_latent_signals(
        deltas["latent"], deltas["k_rope"], gram_k, gram_kv, NUM_HEADS
    )

    assert relative_residual(signals.kv, truth + v_energy) < 1e-12


# --------------------------------------------------------------------------
# compute_latent_signals
# --------------------------------------------------------------------------


def test_row_and_naive_differ_only_by_the_rope_term(weights, deltas):
    gram_k = build_gram_matrix(weights["w_uk"], torch.float64)
    gram_kv = build_gram_matrix(weights["kv_b_proj"], torch.float64)
    signals = compute_latent_signals(
        deltas["latent"], deltas["k_rope"], gram_k, gram_kv, NUM_HEADS
    )
    rope_energy = (deltas["k_rope"] ** 2).sum(dim=-1)
    assert relative_residual(signals.row - signals.naive, rope_energy) < 1e-12


def test_signals_are_frozen(weights, deltas):
    gram_k = build_gram_matrix(weights["w_uk"], torch.float64)
    gram_kv = build_gram_matrix(weights["kv_b_proj"], torch.float64)
    signals = compute_latent_signals(
        deltas["latent"], deltas["k_rope"], gram_k, gram_kv, NUM_HEADS
    )
    assert isinstance(signals, LatentSignals)
    with pytest.raises(dataclasses.FrozenInstanceError):
        signals.exact = torch.zeros(NUM_TOKENS)


def test_rejects_non_positive_head_count(weights, deltas):
    gram_k = build_gram_matrix(weights["w_uk"], torch.float64)
    gram_kv = build_gram_matrix(weights["kv_b_proj"], torch.float64)
    with pytest.raises(ValueError, match="num_heads must be positive"):
        compute_latent_signals(
            deltas["latent"], deltas["k_rope"], gram_k, gram_kv, 0
        )


def test_rejects_token_count_mismatch(weights, deltas):
    gram_k = build_gram_matrix(weights["w_uk"], torch.float64)
    gram_kv = build_gram_matrix(weights["kv_b_proj"], torch.float64)
    with pytest.raises(ValueError, match="token count mismatch"):
        compute_latent_signals(
            deltas["latent"],
            deltas["k_rope"][: NUM_TOKENS - 1],
            gram_k,
            gram_kv,
            NUM_HEADS,
        )
