# SPDX-License-Identifier: Apache-2.0
"""Tests for the concentration metrics.

The anchors matter more than the plumbing here. A flat vector has to score
exactly its kept fraction and a one-coordinate vector has to score ~1,
because every claim about whether concentration survives absorption is
read against those two reference points.
"""

# Standard
import pytest

# Third Party
import torch

# First Party
from absorbkv.concentration import (
    absorbed_query,
    group_mean_abs,
    head_agreement,
    topk_mass,
)

TOKENS = 6
HEADS = 4
NOPE = 8
RANK = 16


# --------------------------------------------------------------------------
# topk_mass -- the anchors
# --------------------------------------------------------------------------


def test_flat_vector_scores_its_fraction():
    """The baseline every result is read against."""
    flat = torch.ones(3, 100)
    curve = topk_mass(flat, (0.1, 0.25, 0.5))
    torch.testing.assert_close(
        curve.mass, torch.tensor([0.1, 0.25, 0.5]), atol=1e-6, rtol=0
    )


def test_single_coordinate_scores_one():
    spike = torch.zeros(3, 100)
    spike[:, 7] = 1.0
    curve = topk_mass(spike, (0.01, 0.25))
    assert float(curve.mass[0]) == pytest.approx(1.0)
    assert float(curve.mass[1]) == pytest.approx(1.0)


def test_mass_is_monotone_in_the_fraction():
    torch.manual_seed(0)
    curve = topk_mass(torch.rand(5, 64).abs(), (0.05, 0.1, 0.25, 0.5, 1.0))
    assert torch.all(curve.mass[1:] >= curve.mass[:-1])
    assert float(curve.mass[-1]) == pytest.approx(1.0)


def test_same_fraction_is_comparable_across_dimensions():
    """Why the curve is indexed by fraction and not by count."""
    small = topk_mass(torch.ones(2, 128), (0.125,))
    large = topk_mass(torch.ones(2, 512), (0.125,))
    assert float(small.mass[0]) == pytest.approx(float(large.mass[0]))
    assert small.dimension == 128
    assert large.dimension == 512


def test_rejects_fractions_out_of_range():
    with pytest.raises(ValueError, match="fractions must lie"):
        topk_mass(torch.ones(2, 8), (0.0, 0.5))


# --------------------------------------------------------------------------
# absorbed_query
# --------------------------------------------------------------------------


@pytest.fixture
def w_uk() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(HEADS * NOPE, RANK)


def test_absorbed_query_shape_and_dtype(w_uk):
    query = torch.randn(TOKENS, HEADS, NOPE, dtype=torch.bfloat16)
    absorbed = absorbed_query(query, w_uk)
    assert absorbed.shape == (TOKENS, HEADS, RANK)
    assert absorbed.dtype == torch.float32


def test_absorbed_query_matches_the_score_it_stands_for(w_uk):
    """``q . (W^UK c)`` must equal ``q' . c``: that identity is the only
    reason the absorbed query is the right thing to measure."""
    torch.manual_seed(2)
    query = torch.randn(TOKENS, HEADS, NOPE)
    latent = torch.randn(RANK)

    per_head = w_uk.view(HEADS, NOPE, RANK)
    direct = torch.einsum("thd,hdr,r->th", query, per_head, latent)
    through_absorbed = absorbed_query(query, w_uk) @ latent

    torch.testing.assert_close(through_absorbed, direct, atol=1e-4, rtol=1e-4)


def test_absorbed_query_rejects_a_mismatched_weight(w_uk):
    query = torch.randn(TOKENS, HEADS + 1, NOPE)
    with pytest.raises(ValueError, match="expected"):
        absorbed_query(query, w_uk)


# --------------------------------------------------------------------------
# group_mean_abs
# --------------------------------------------------------------------------


def test_group_mean_averages_within_contiguous_groups():
    query = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [0.0, 5.0], [0.0, 7.0]]])
    grouped = group_mean_abs(query, num_groups=2)
    assert grouped.shape == (1, 2, 2)
    torch.testing.assert_close(grouped[0, 0], torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(grouped[0, 1], torch.tensor([0.0, 6.0]))


def test_group_mean_takes_absolute_values_first():
    """Selection is on magnitude: opposite signs must not cancel."""
    query = torch.tensor([[[2.0], [-2.0]]])
    assert float(group_mean_abs(query, num_groups=1)[0, 0, 0]) == 2.0


def test_one_group_averages_every_head():
    """What MLA is forced into: a single latent serves all heads."""
    torch.manual_seed(3)
    query = torch.randn(TOKENS, HEADS, NOPE)
    grouped = group_mean_abs(query, num_groups=1)
    assert grouped.shape == (TOKENS, 1, NOPE)
    torch.testing.assert_close(grouped[:, 0], query.abs().mean(dim=1))


def test_group_mean_rejects_uneven_groups():
    with pytest.raises(ValueError, match="do not divide"):
        group_mean_abs(torch.randn(TOKENS, 3, NOPE), num_groups=2)


# --------------------------------------------------------------------------
# head_agreement
# --------------------------------------------------------------------------


def test_identical_heads_agree_completely():
    one = torch.rand(TOKENS, 1, 32)
    same = one.expand(TOKENS, HEADS, 32).contiguous()
    assert float(head_agreement(same, 0.25)) == pytest.approx(1.0)


def test_disjoint_heads_agree_not_at_all():
    values = torch.zeros(1, 2, 8)
    values[0, 0, :4] = 1.0
    values[0, 1, 4:] = 1.0
    assert float(head_agreement(values, 0.5)) == pytest.approx(0.0)


def test_head_agreement_needs_two_heads():
    with pytest.raises(ValueError, match="at least two heads"):
        head_agreement(torch.rand(TOKENS, 1, 8), 0.25)
