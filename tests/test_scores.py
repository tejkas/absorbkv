# SPDX-License-Identifier: Apache-2.0
"""Tests for scoring the latent directly.

The load-bearing test is ``test_scores_match_the_models_own_attention``:
everything the relevance measurement reports is a comparison against
these scores, so if they are wrong, every number downstream is wrong in a
way no later check would catch. A randomly initialized model is enough,
since the claim is about algebra, not weights.
"""

# Standard
import pytest

# Third Party
import torch
from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Model

# First Party
from absorbkv import scores as scores_module
from absorbkv.capture import capture_context, capture_queries
from absorbkv.scores import captured_mass, exact_scores
from absorbkv.signals import extract_w_uk

VOCAB = 100
HIDDEN = 64
LAYERS = 2
HEADS = 2
KV_LORA = 16
QK_NOPE = 8
QK_ROPE = 4
CONTEXT = 12
QUESTION = 3


@pytest.fixture(scope="module")
def config() -> DeepseekV2Config:
    config = DeepseekV2Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=HIDDEN,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        kv_lora_rank=KV_LORA,
        q_lora_rank=None,
        qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=QK_ROPE,
        v_head_dim=QK_NOPE,
        first_k_dense_replace=LAYERS + 1,
        max_position_embeddings=512,
    )
    config._attn_implementation = "eager"
    return config


@pytest.fixture(scope="module")
def captured(config) -> dict:
    """One forward, with the model's own attention kept for comparison."""
    torch.manual_seed(0)
    model = DeepseekV2Model(config).to(torch.float32).eval()
    ids = torch.randint(0, VOCAB, (1, CONTEXT + QUESTION))
    positions = torch.arange(CONTEXT + QUESTION).unsqueeze(0)

    with torch.inference_mode():
        out = model(
            input_ids=ids,
            position_ids=positions,
            use_cache=False,
            output_attentions=True,
        )
    attention = out.attentions[0][0].clone()

    context = capture_context(model, ids[:, :CONTEXT], positions[:, :CONTEXT])
    queries = capture_queries(model, ids, positions, QUESTION, HEADS, QK_NOPE + QK_ROPE)
    w_uk = extract_w_uk(
        model.layers[0].self_attn.kv_b_proj.weight.data, HEADS, QK_NOPE, QK_NOPE
    )
    return {
        "attention": attention,
        "context": context,
        "queries": queries,
        "w_uk": w_uk,
    }


def model_weights_over_context(attention: torch.Tensor) -> torch.Tensor:
    """The model's attention from question tokens, renormalized to the context."""
    weights = attention[:, CONTEXT:, :CONTEXT]
    return weights / weights.sum(-1, keepdim=True)


def ours(captured, config) -> torch.Tensor:
    scored = exact_scores(
        captured["queries"].query[0],
        captured["queries"].position_ids,
        captured["context"].latent[0],
        captured["context"].k_rope[0],
        captured["w_uk"],
        config,
    )
    return scored.weights.permute(1, 0, 2)


# --------------------------------------------------------------------------
# the scorer
# --------------------------------------------------------------------------


def test_scores_match_the_models_own_attention(captured, config):
    """Scoring the latent reproduces attention over materialized keys.

    Both halves are exercised: ``q' . c`` for the absorbed part and the
    rotated ``q^R . k^R`` for the part that cannot absorb.
    """
    theirs = model_weights_over_context(captured["attention"])
    assert float((ours(captured, config) - theirs).abs().max()) < 1e-6


def test_forgetting_to_rotate_the_query_is_caught(captured, config, monkeypatch):
    """The trap: ``q`` is captured pre-RoPE, ``k^R`` is cached post-RoPE.

    Scoring them together silently computes the score for a query at
    position 0. Here the query sits at position 14 and the error is
    already visible; at a realistic 2400 it is far larger.
    """
    monkeypatch.setattr(
        scores_module,
        "rotate_query",
        lambda query_rope, position_ids, config: query_rope.to(torch.float32),
    )
    theirs = model_weights_over_context(captured["attention"])
    assert float((ours(captured, config) - theirs).abs().max()) > 1e-4


def test_exact_scores_rejects_a_mismatched_head_dim(captured, config):
    bad = captured["queries"].query[0][..., :-1]
    with pytest.raises(ValueError, match="expected"):
        exact_scores(
            bad,
            captured["queries"].position_ids,
            captured["context"].latent[0],
            captured["context"].k_rope[0],
            captured["w_uk"],
            config,
        )


def test_weights_sum_to_one_over_the_context(captured, config):
    weights = ours(captured, config)
    torch.testing.assert_close(
        weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-5, rtol=0
    )


# --------------------------------------------------------------------------
# captured_mass
# --------------------------------------------------------------------------


def test_selecting_everything_captures_all_the_mass():
    weights = torch.rand(2, 3, 10).softmax(-1)
    everything = torch.arange(10)
    torch.testing.assert_close(
        captured_mass(weights, everything), torch.ones(2, 3), atol=1e-6, rtol=0
    )


def test_mass_matches_a_hand_computed_case():
    weights = torch.tensor([[[0.5, 0.3, 0.15, 0.05]]])
    mass = float(captured_mass(weights, torch.tensor([0, 2]))[0, 0])
    assert mass == pytest.approx(0.65)


def test_the_ideal_selection_beats_a_random_one():
    """What the metric exists to distinguish."""
    torch.manual_seed(0)
    weights = (torch.rand(4, 2, 50) * 5).softmax(-1)
    best = weights.mean(dim=(0, 1)).topk(5).indices
    worst = weights.mean(dim=(0, 1)).topk(5, largest=False).indices
    assert float(captured_mass(weights, best).mean()) > float(
        captured_mass(weights, worst).mean()
    )


def test_per_query_selections_are_supported():
    """Each query token may select its own set, as the method does."""
    weights = torch.tensor([[[0.6, 0.4, 0.0]], [[0.0, 0.3, 0.7]]])
    selected = torch.tensor([[0], [2]])
    mass = captured_mass(weights, selected)
    assert float(mass[0, 0]) == pytest.approx(0.6)
    assert float(mass[1, 0]) == pytest.approx(0.7)


def test_captured_mass_rejects_a_bad_shape():
    with pytest.raises(ValueError, match="num_selected"):
        captured_mass(torch.rand(2, 2, 5), torch.zeros(2, 2, 2, dtype=torch.long))


def _tiny_mla_config():
    """A small MLA model with the real structure, YaRN included.

    Same shape of thing as DeepSeek-V2-Lite -- a latent, a decoupled RoPE
    key, no ``q_lora_rank`` so ``q_proj`` exists for the capture hook --
    at dimensions that run in milliseconds on a laptop.
    """
    # Third Party
    from transformers.models.deepseek_v2.configuration_deepseek_v2 import (
        DeepseekV2Config,
    )

    return DeepseekV2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        kv_lora_rank=16,
        q_lora_rank=None,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=2,
        max_position_embeddings=32,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 8,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        attn_implementation="eager",
    )


def _reconstruct_against_model(query_offset: int = 0):
    """Score from the latent and return (model's attention, ours, peak).

    Mirrors the capture exactly: the latent and the RoPE key come out of
    the model's own cache, the query off a forward hook on ``q_proj``
    before any rotation. ``query_offset`` shifts the position the query is
    rotated at, which is the negative control.
    """
    # Third Party
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
        DeepseekV2ForCausalLM,
    )

    # First Party
    from absorbkv.scores import exact_scores
    from absorbkv.signals import extract_w_uk

    config = _tiny_mla_config()
    layer, tokens = 1, 12
    torch.manual_seed(0)
    model = DeepseekV2ForCausalLM(config).eval().float()
    attention = model.model.layers[layer].self_attn

    # Random init leaves every logit near zero, so the softmax comes out
    # flat at 1/tokens and the comparison would hold for a wrong formula
    # too. Scaling the two projections that feed q.k spreads it.
    with torch.no_grad():
        attention.q_proj.weight.mul_(16)
        attention.kv_a_proj_with_mqa.weight.mul_(16)

    head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    captured = {}

    def hook(module, args, output):
        captured["query"] = (
            output[0, -1:].unflatten(-1, (config.num_attention_heads, head_dim)).clone()
        )

    handle = attention.q_proj.register_forward_hook(hook)
    ids = torch.randint(0, config.vocab_size, (1, tokens))
    positions = torch.arange(tokens).unsqueeze(0)
    with torch.no_grad():
        out = model(
            input_ids=ids,
            position_ids=positions,
            use_cache=True,
            output_attentions=True,
        )
    handle.remove()

    cache = out.past_key_values
    w_uk = extract_w_uk(
        attention.kv_b_proj.weight.float(),
        config.num_attention_heads,
        config.qk_nope_head_dim,
        config.v_head_dim,
    )
    scored = exact_scores(
        captured["query"],
        positions[0, -1:] + query_offset,
        cache.layers[layer].keys[0, 0],
        cache.layers[layer].values[0, 0],
        w_uk,
        config,
    )
    theirs = out.attentions[layer][0, :, -1, :].detach()
    return theirs, scored.weights[0].detach(), float(theirs.max())


def test_latent_scores_match_the_models_own_attention():
    """The end-to-end check: scoring from the latent reproduces the model.

    Everything the relevance measurements rest on is in this one
    comparison -- absorption, the rotation applied at analysis time, the
    YaRN softmax scale, and reading the latent out of the cache. The
    model's eager path computes the same attention the slow way, by
    expanding per-head keys.
    """
    theirs, mine, peak = _reconstruct_against_model()
    assert peak > 0.5, f"attention is too flat to test against: peak {peak:.3f}"
    assert torch.allclose(mine, theirs, atol=1e-6), (
        f"max difference {float((mine - theirs).abs().max()):.2e}"
    )


def test_the_comparison_detects_a_wrong_query_position():
    """Proof the test above can fail: rotate the query five positions off."""
    theirs, mine, _ = _reconstruct_against_model(query_offset=-5)
    assert float((mine - theirs).abs().max()) > 0.1
