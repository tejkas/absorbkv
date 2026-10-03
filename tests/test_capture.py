# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the capture path, on a randomly initialized model.

No weights are downloaded and no GPU is used. Everything here is
structural -- shapes, guards, and the two properties the chunked pass
rests on -- so a random model proves it as well as a trained one:

* the latent does not depend on the absolute position a chunk is
  forwarded at, only on relative positions within it (``test_position_
  shift_leaves_the_latent_alone``);
* therefore the first chunk, which has nothing before it in either pass,
  has zero deviation up to floating-point noise
  (``test_first_chunk_has_no_deviation``).

If the second fails on real hardware, the capture is wrong and no amount
of GPU time will produce a usable measurement.

Tolerances come from measuring this model rather than from theory: the
position shift lands at 5e-08 and the first chunk at 2e-07 in float32.
"""

# Standard
import pytest

# Third Party
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Model
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

# First Party
from absorbkv.capture import (
    ContextCapture,
    GqaContextCapture,
    QueryCapture,
    capture_context,
    capture_document,
    capture_gqa_context,
    capture_queries,
)

VOCAB = 100
HIDDEN = 64
LAYERS = 2
HEADS = 2
KV_LORA = 16
QK_NOPE = 8
QK_ROPE = 4
V_HEAD = 8
QK_HEAD = QK_NOPE + QK_ROPE
GQA_HEADS = 4
GQA_KV_HEADS = 2
GQA_HEAD_DIM = HIDDEN // GQA_HEADS


def relative_residual(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Max absolute deviation, scaled by the largest expected magnitude."""
    scale = expected.abs().max()
    if scale == 0:
        return float(actual.abs().max())
    return float((actual - expected).abs().max() / scale)


@pytest.fixture(scope="module")
def model() -> DeepseekV2Model:
    """A two-layer MLA model with random weights, dense MLPs only.

    ``first_k_dense_replace`` above the layer count keeps the mixture-of-
    experts path out of the way: this suite tests capture, not routing.
    """
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
        v_head_dim=V_HEAD,
        first_k_dense_replace=LAYERS + 1,
        max_position_embeddings=512,
    )
    torch.manual_seed(0)
    return DeepseekV2Model(config).to(torch.float32).eval()


@pytest.fixture(scope="module")
def gqa_model() -> LlamaModel:
    """An ordinary attention model: real KV heads, no latent."""
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=HIDDEN,
        num_hidden_layers=LAYERS,
        num_attention_heads=GQA_HEADS,
        num_key_value_heads=GQA_KV_HEADS,
    )
    torch.manual_seed(0)
    return LlamaModel(config).eval()


@pytest.fixture
def ids() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, VOCAB, (1, 12))


@pytest.fixture
def positions(ids) -> torch.Tensor:
    return torch.arange(ids.shape[1]).unsqueeze(0)


# --------------------------------------------------------------------------
# capture_context
# --------------------------------------------------------------------------


def test_capture_context_shapes(model, ids, positions):
    capture = capture_context(model, ids, positions)
    assert isinstance(capture, ContextCapture)
    assert capture.latent.shape == (LAYERS, ids.shape[1], KV_LORA)
    assert capture.k_rope.shape == (LAYERS, ids.shape[1], QK_ROPE)
    assert capture.position_ids.shape == (ids.shape[1],)


def test_capture_context_keeps_the_positions_it_was_given(model, ids):
    offset = torch.arange(ids.shape[1]).unsqueeze(0) + 40
    capture = capture_context(model, ids, offset)
    assert torch.equal(capture.position_ids, offset[0])


def test_capture_context_returns_cpu_tensors(model, ids, positions):
    capture = capture_context(model, ids, positions)
    assert capture.latent.device.type == "cpu"
    assert not capture.latent.is_inference()


def test_position_shift_leaves_the_latent_alone(model, ids, positions):
    """The property the whole chunked pass depends on.

    RoPE scores depend on the difference between positions, so shifting a
    span to a later offset leaves every score inside it unchanged, and
    with it the hidden states and the latent they produce. ``k_rope`` is
    rotated by the shift and must therefore differ.
    """
    here = capture_context(model, ids, positions)
    there = capture_context(model, ids, positions + 40)

    assert relative_residual(there.latent, here.latent) < 1e-6
    assert relative_residual(there.k_rope, here.k_rope) > 1e-2


def test_capture_context_rejects_a_non_mla_cache(gqa_model, ids, positions):
    """A model with real KV heads stores per-head keys in the same slot."""
    with pytest.raises(ValueError, match="singleton head axis"):
        capture_context(gqa_model, ids, positions)


def test_capture_context_rejects_training_mode(model, ids, positions):
    model.train()
    try:
        with pytest.raises(ValueError, match="eval mode"):
            capture_context(model, ids, positions)
    finally:
        model.eval()


def test_capture_context_rejects_a_batch(model, positions):
    torch.manual_seed(2)
    batched = torch.randint(0, VOCAB, (2, positions.shape[1]))
    with pytest.raises(ValueError, match="one sequence at a time"):
        capture_context(model, batched, positions.expand(2, -1))


def test_capture_context_rejects_mismatched_positions(model, ids, positions):
    with pytest.raises(ValueError, match="same shape"):
        capture_context(model, ids, positions[:, :-1])


# --------------------------------------------------------------------------
# capture_queries
# --------------------------------------------------------------------------


def test_capture_queries_keeps_only_the_tail(model):
    """Checked against q_proj's own output, captured independently."""
    torch.manual_seed(3)
    full_ids = torch.randint(0, VOCAB, (1, 16))
    full_positions = torch.arange(16).unsqueeze(0)
    num_question = 4

    seen = {}

    def spy(index):
        def hook(module, args, output):
            seen[index] = output.detach().clone()

        return hook

    handles = [
        layer.self_attn.q_proj.register_forward_hook(spy(index))
        for index, layer in enumerate(model.layers)
    ]
    try:
        with torch.inference_mode():
            model(input_ids=full_ids, position_ids=full_positions, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    capture = capture_queries(
        model, full_ids, full_positions, num_question, HEADS, QK_HEAD
    )
    assert isinstance(capture, QueryCapture)
    assert capture.query.shape == (LAYERS, num_question, HEADS, QK_HEAD)

    for index in range(LAYERS):
        expected = seen[index][0, -num_question:].unflatten(-1, (HEADS, QK_HEAD))
        assert relative_residual(capture.query[index], expected) == 0.0


def test_capture_queries_does_not_leave_hooks_attached(model, ids, positions):
    """A leaked hook would fire on every later pass, including the other one."""
    before = sum(len(layer.self_attn.q_proj._forward_hooks) for layer in model.layers)
    capture_queries(model, ids, positions, 3, HEADS, QK_HEAD)
    after = sum(len(layer.self_attn.q_proj._forward_hooks) for layer in model.layers)
    assert before == after == 0


def test_capture_queries_rejects_an_empty_tail(model, ids, positions):
    """``output[0, -0:]`` is the whole tensor, not nothing."""
    with pytest.raises(ValueError, match="num_question_tokens"):
        capture_queries(model, ids, positions, 0, HEADS, QK_HEAD)


def test_capture_queries_rejects_an_oversized_tail(model, ids, positions):
    with pytest.raises(ValueError, match="num_question_tokens"):
        capture_queries(model, ids, positions, ids.shape[1] + 1, HEADS, QK_HEAD)


# --------------------------------------------------------------------------
# capture_document
# --------------------------------------------------------------------------


@pytest.fixture
def document(model) -> tuple:
    torch.manual_seed(4)
    chunks = [torch.randint(0, VOCAB, (1, width)) for width in (5, 7, 4)]
    question = torch.randint(0, VOCAB, (1, 3))
    return capture_document(model, chunks, question), chunks, question


def test_capture_document_assembles_both_passes(document):
    capture, chunks, question = document
    total = sum(chunk.shape[1] for chunk in chunks)
    assert capture.chunked.latent.shape == (LAYERS, total, KV_LORA)
    assert capture.truth.latent.shape == (LAYERS, total, KV_LORA)
    assert capture.queries.query.shape == (
        LAYERS,
        question.shape[1],
        HEADS,
        QK_HEAD,
    )
    assert capture.chunk_starts.tolist() == [0, 5, 12]


def test_first_chunk_has_no_deviation(document):
    """Nothing precedes chunk 1 in either pass, so its deviation is noise.

    This is the end-to-end check that the position discipline works. A
    real deviation here means the chunked pass and the truth pass
    disagree about where the tokens are, and every later number is
    measuring a rotation instead of staleness.
    """
    capture, chunks, _ = document
    width = chunks[0].shape[1]
    residual = relative_residual(
        capture.chunked.latent[:, :width], capture.truth.latent[:, :width]
    )
    assert residual < 1e-5


def test_later_chunks_do_deviate(document):
    """The complement: without it, a broken capture returning zeros passes."""
    capture, chunks, _ = document
    width = chunks[0].shape[1]
    residual = relative_residual(
        capture.chunked.latent[:, width:], capture.truth.latent[:, width:]
    )
    assert residual > 1e-2


def test_capture_document_rejects_no_chunks(model):
    with pytest.raises(ValueError, match="at least one chunk"):
        capture_document(model, [], torch.randint(0, VOCAB, (1, 3)))


# --------------------------------------------------------------------------
# capture_gqa_context -- the baseline path
# --------------------------------------------------------------------------


def test_capture_gqa_context_shapes(gqa_model, ids, positions):
    capture = capture_gqa_context(gqa_model, ids, positions)
    assert isinstance(capture, GqaContextCapture)
    assert capture.keys.shape == (
        LAYERS,
        ids.shape[1],
        GQA_KV_HEADS,
        GQA_HEAD_DIM,
    )
    assert capture.position_ids.shape == (ids.shape[1],)


def test_capture_gqa_context_is_token_major(gqa_model, ids, positions):
    """The cache is head-major; every capture in this module is token-major."""
    cache = DynamicCache()
    with torch.inference_mode():
        gqa_model(
            input_ids=ids,
            position_ids=positions,
            past_key_values=cache,
            use_cache=True,
        )
    expected = cache.layers[0].keys[0].transpose(0, 1)

    capture = capture_gqa_context(gqa_model, ids, positions)
    assert relative_residual(capture.keys[0], expected.cpu()) == 0.0


def test_capture_gqa_context_rejects_an_mla_cache(model, ids, positions):
    """The mirror of the guard in capture_context."""
    with pytest.raises(ValueError, match="use capture_context"):
        capture_gqa_context(model, ids, positions)


def test_capture_queries_works_on_the_baseline_model(gqa_model, ids, positions):
    """What parametrizing num_heads and head_dim bought."""
    capture = capture_queries(
        gqa_model, ids, positions, 3, GQA_HEADS, GQA_HEAD_DIM
    )
    assert capture.query.shape == (LAYERS, 3, GQA_HEADS, GQA_HEAD_DIM)
