# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-document capture functions the runner drives.

``main`` itself is argument parsing and a loop; what matters is that the
two model paths produce the right structure and that the result survives
a save and load without this package being importable -- a capture that
can only be read by the code that wrote it is a capture that stops being
readable the moment those classes change.
"""

# Standard
import pytest

# Third Party
import torch
from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Model
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

# First Party
from absorbkv.data import Document
from absorbkv.runner import CHUNK_SIZES, capture_gqa_document, capture_mla_document

VOCAB = 100
HIDDEN = 64
LAYERS = 2
HEADS = 2
KV_LORA = 16
QK_NOPE = 8
QK_ROPE = 4
QK_HEAD = QK_NOPE + QK_ROPE
GQA_HEADS = 4
GQA_KV_HEADS = 2
GQA_HEAD_DIM = HIDDEN // GQA_HEADS
PARAGRAPH_WIDTHS = (5, 7, 4, 6, 8, 3)


@pytest.fixture(scope="module")
def mla_model() -> DeepseekV2Model:
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
    torch.manual_seed(0)
    return DeepseekV2Model(config).to(torch.float32).eval()


@pytest.fixture(scope="module")
def gqa_model() -> LlamaModel:
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
def document() -> Document:
    torch.manual_seed(1)
    return Document(
        record_id="2hop__1_2",
        hops="2hop",
        paragraphs=[
            torch.randint(0, VOCAB, (1, width)) for width in PARAGRAPH_WIDTHS
        ],
        question_plain=torch.randint(0, VOCAB, (1, 3)),
        question_chat=torch.randint(0, VOCAB, (1, 4)),
    )


def total_tokens(document: Document) -> int:
    return sum(ids.shape[1] for ids in document.paragraphs)


# --------------------------------------------------------------------------
# the MLA path
# --------------------------------------------------------------------------


def test_mla_capture_covers_every_chunk_size_and_framing(mla_model, document):
    captured = capture_mla_document(mla_model, document)
    tokens = total_tokens(document)

    assert captured["truth"]["latent"].shape == (LAYERS, tokens, KV_LORA)
    assert captured["truth"]["k_rope"].shape == (LAYERS, tokens, QK_ROPE)
    assert set(captured["chunked"]) == set(CHUNK_SIZES)
    assert set(captured["queries"]) == {"plain", "chat"}

    for per_chunk, chunked in captured["chunked"].items():
        assert chunked["capture"]["latent"].shape == (LAYERS, tokens, KV_LORA)
        expected_chunks = -(-len(document.paragraphs) // per_chunk)
        assert len(chunked["chunk_starts"]) == expected_chunks


def test_mla_capture_first_chunk_is_noise(mla_model, document):
    """The end-to-end position check, through the runner's own entry point."""
    captured = capture_mla_document(mla_model, document)
    truth = captured["truth"]["latent"]
    for per_chunk, chunked in captured["chunked"].items():
        width = sum(
            ids.shape[1] for ids in document.paragraphs[:per_chunk]
        )
        latent = chunked["capture"]["latent"]
        residual = float(
            (latent[:, :width] - truth[:, :width]).abs().max()
            / truth[:, :width].abs().max()
        )
        assert residual < 1e-5, f"chunking {per_chunk}"


# --------------------------------------------------------------------------
# the baseline path
# --------------------------------------------------------------------------


def test_gqa_capture_has_keys_and_queries(gqa_model, document):
    captured = capture_gqa_document(gqa_model, document)
    tokens = total_tokens(document)

    assert captured["keys"].shape == (
        LAYERS,
        tokens,
        GQA_KV_HEADS,
        GQA_HEAD_DIM,
    )
    assert set(captured["queries"]) == {"plain", "chat"}
    assert captured["queries"]["plain"]["query"].shape == (
        LAYERS,
        document.question_plain.shape[1],
        GQA_HEADS,
        GQA_HEAD_DIM,
    )


def test_gqa_capture_has_no_chunked_pass(gqa_model, document):
    """Relevance only: there is no staleness to measure on the baseline."""
    captured = capture_gqa_document(gqa_model, document)
    assert "chunked" not in captured
    assert "truth" not in captured


# --------------------------------------------------------------------------
# what lands on disk
# --------------------------------------------------------------------------


def test_saved_capture_loads_without_this_package(mla_model, document, tmp_path):
    """``weights_only=True`` refuses to unpickle custom classes.

    Passing it proves the file holds plain containers and tensors, so the
    analysis -- or anyone the data is shared with -- can read a capture
    without absorbkv installed, and a later change to the dataclasses
    cannot orphan an old run.
    """
    path = tmp_path / "doc.pt"
    torch.save(capture_mla_document(mla_model, document), path)

    reloaded = torch.load(path, weights_only=True)
    assert set(reloaded) == {"truth", "chunked", "queries"}
    assert isinstance(reloaded["truth"]["latent"], torch.Tensor)
    assert isinstance(reloaded["chunked"][CHUNK_SIZES[0]]["chunk_starts"], torch.Tensor)
