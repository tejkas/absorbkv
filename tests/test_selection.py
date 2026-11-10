# SPDX-License-Identifier: Apache-2.0
"""Tests for the MLA translation of RocketKV's decode-time selection.

The test that matters is ``test_matches_their_code_when_rope_is_empty``:
with no RoPE component, the translation has nothing MLA-specific left to
do and must reproduce their implementation exactly. Anything it then
reports about MLA is a difference caused by MLA, not by a rewrite.

Their code lives in ``papers/rocketkv/code/`` under its own licence.
"""

# Standard
import sys
from pathlib import Path

# Third Party
import pytest
import torch

# First Party
from absorbkv.selection import Budget, select_latent

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/rocketkv/code"))

# Third Party
from hybrid_sparse_attention import select_tokens  # noqa: E402

HEADS = 16
WIDTH = 576
TOKENS = 300


@pytest.fixture
def query() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(HEADS, WIDTH)


@pytest.fixture
def keys() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(TOKENS, WIDTH)


# --------------------------------------------------------------------------
# their budget derivation
# --------------------------------------------------------------------------


def test_budget_reproduces_their_worked_example():
    """Paper §3.6: c=64 gives r=0.56, 10.3x in stage one, 6.2x in stage two,
    which splits into 3x along the sequence and 2.1x along the head dim."""
    budget = Budget.of(64)
    assert budget.eviction_ratio == pytest.approx(10.3, abs=0.1)
    assert budget.sparse_ratio == pytest.approx(6.2, abs=0.1)
    assert budget.page_size == 3
    assert 1 / budget.kept_fraction == pytest.approx(2.1, abs=0.1)


def test_the_split_factor_is_capped():
    """``r = min(0.2 + 0.06 log2 c, 0.8)``, so stage two never vanishes."""
    assert Budget.of(10**6).sparse_ratio > 1.0


def test_stage_two_stays_mild_however_aggressive_the_target():
    """Why the reduction keeps roughly half the coordinates at every budget:
    the split factor grows with c, so stage two always lands near 4-6x."""
    for overall in (16, 64, 256, 400):
        budget = Budget.of(overall)
        assert 4.0 < budget.sparse_ratio < 7.0
        assert 0.4 < budget.kept_fraction < 0.7


# --------------------------------------------------------------------------
# equivalence with their implementation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("overall", (16, 64, 256))
@pytest.mark.parametrize("reduce_head_dim", (True, False))
def test_matches_their_code_when_rope_is_empty(query, keys, overall, reduce_head_dim):
    """With no RoPE half, ours is their pipeline and must agree exactly."""
    budget = Budget.of(overall)
    mine = select_latent(
        query,
        torch.zeros(HEADS, 0),
        keys,
        torch.zeros(TOKENS, 0),
        topk=20,
        compression_ratio=budget.sparse_ratio,
        reduce_head_dim=reduce_head_dim,
    )
    theirs = select_tokens(
        query.view(1, 1, HEADS, 1, WIDTH),
        keys.view(1, 1, 1, TOKENS, WIDTH),
        topk=20,
        compression_ratio=budget.sparse_ratio,
        reduce_head_dim=reduce_head_dim,
    )
    assert set(mine.tolist()) == set(theirs.flatten().tolist())


# --------------------------------------------------------------------------
# the MLA-specific parts
# --------------------------------------------------------------------------


def test_rope_coordinates_are_never_truncated(query, keys):
    """They are 64 against the latent's 512 and carry the term that cannot
    absorb, so step 3 keeps them whatever the budget."""
    content, rope = query[:, :512], query[:, 512:]
    latent, k_rope = keys[:, :512], keys[:, 512:]
    with_rope = select_latent(content, rope, latent, k_rope, 20, 6.2)
    zeroed = select_latent(
        content, torch.zeros_like(rope), latent, k_rope, 20, 6.2
    )
    assert set(with_rope.tolist()) != set(zeroed.tolist())


def test_selection_respects_the_token_budget(query, keys):
    content, rope = query[:, :512], query[:, 512:]
    latent, k_rope = keys[:, :512], keys[:, 512:]
    assert len(select_latent(content, rope, latent, k_rope, 7, 6.2)) == 7
    assert len(select_latent(content, rope, latent, k_rope, 10**6, 6.2)) == TOKENS


def test_mismatched_widths_are_rejected(query, keys):
    content, rope = query[:, :512], query[:, 512:]
    latent, k_rope = keys[:, :512], keys[:, 512:]
    with pytest.raises(ValueError, match="latent"):
        select_latent(content[:, :-1], rope, latent, k_rope, 20, 6.2)
    with pytest.raises(ValueError, match="k_rope"):
        select_latent(content, rope[:, :-1], latent, k_rope, 20, 6.2)


def test_forcing_the_scale_changes_nothing_when_it_matches(query, keys):
    """The control used in the measurement: overriding the temperature with
    the value the run would have computed must leave the result alone."""
    content, rope = query[:, :512], query[:, 512:]
    latent, k_rope = keys[:, :512], keys[:, 512:]
    full = torch.cat([content, rope], -1)
    sign = torch.where(full.sum(0) > 0, 1.0, -1.0)
    scale = torch.sqrt(
        WIDTH * (full * sign).abs().sum(-1, keepdim=True)
        / full.abs().sum(-1, keepdim=True)
    )
    plain = select_latent(content, rope, latent, k_rope, 20, 6.2, False)
    forced = select_latent(content, rope, latent, k_rope, 20, 6.2, False, scale)
    assert set(plain.tolist()) == set(forced.tolist())


# --------------------------------------------------------------------------
# the overrides used to test the traffic remedy
# --------------------------------------------------------------------------


@pytest.mark.parametrize("truncate_rope", (True, False))
def test_keeping_every_coordinate_matches_no_reduction(query, keys, truncate_rope):
    """``force_kept=1.0`` keeps the whole width, so it must agree with the
    reduction switched off. Coordinate order differs -- ``topk`` returns
    them by value -- and both the score and the rescaling are sums over
    coordinates, so the permutation cannot matter."""
    content, rope = query[:, :512], query[:, 512:]
    latent, k_rope = keys[:, :512], keys[:, 512:]
    unreduced = select_latent(content, rope, latent, k_rope, 20, 6.2, False)
    everything = select_latent(
        content, rope, latent, k_rope, 20, 6.2, True,
        force_kept=1.0, truncate_rope=truncate_rope,
    )
    assert set(unreduced.tolist()) == set(everything.tolist())


def test_forcing_their_own_kept_fraction_changes_nothing(query, keys):
    content, rope = query[:, :512], query[:, 512:]
    latent, k_rope = keys[:, :512], keys[:, 512:]
    budget = Budget.of(64)
    plain = select_latent(content, rope, latent, k_rope, 20, budget.sparse_ratio)
    forced = select_latent(
        content, rope, latent, k_rope, 20, budget.sparse_ratio,
        force_kept=budget.kept_fraction,
    )
    assert set(plain.tolist()) == set(forced.tolist())


def test_truncating_rope_can_drop_rope_coordinates(query, keys):
    """The point of the variant: at a low kept fraction the RoPE
    coordinates stop being guaranteed, so zeroing them may now be a no-op
    where :func:`test_rope_coordinates_are_never_truncated` showed it is
    not under the default."""
    content, rope = query[:, :512], query[:, 512:]
    kept = sum(
        1
        for i in torch.cat([content, rope], -1).abs().mean(0).topk(
            int(WIDTH * 0.24)
        ).indices.tolist()
        if i >= 512
    )
    assert kept < 64
