# SPDX-License-Identifier: Apache-2.0
"""Tests for the parts of the loader that do not need the network.

``load_documents`` reaches MuSiQue and is left to a manual check; the
tokenization and grouping it delegates to are pure, and those are what a
mistake would silently corrupt.
"""

# Standard
import pytest

# Third Party
import torch

# First Party
from absorbkv.data import Document, group_paragraphs, tokenize_record

BOS = 1


class StubTokenizer:
    """One token per character, so token counts are readable by eye."""

    bos_token_id = BOS

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        assert not add_special_tokens, "the loader tokenizes pieces, not prompts"
        return {"input_ids": [ord(character) % 97 for character in text]}


@pytest.fixture
def record() -> dict:
    return {
        "id": "3hop1__12345_678",
        "question": "who?",
        "paragraphs": [
            {"title": "A", "paragraph_text": "alpha", "is_supporting": True},
            {"title": "B", "paragraph_text": "beta", "is_supporting": False},
            {"title": "C", "paragraph_text": "gamma gamma", "is_supporting": False},
        ],
    }


# --------------------------------------------------------------------------
# group_paragraphs
# --------------------------------------------------------------------------


@pytest.fixture
def paragraphs() -> list[torch.Tensor]:
    return [torch.arange(width).unsqueeze(0) for width in (5, 7, 4, 6)]


def test_grouping_tiles_the_input_exactly(paragraphs):
    """Chunks must cover every token once: the two passes are aligned by
    position, so a dropped or duplicated token shifts everything after it."""
    for per_chunk in (1, 2, 3, 4, 10):
        chunks = group_paragraphs(paragraphs, per_chunk)
        assert torch.equal(
            torch.cat(chunks, dim=1), torch.cat(paragraphs, dim=1)
        )


def test_grouping_counts(paragraphs):
    assert len(group_paragraphs(paragraphs, 1)) == 4
    assert len(group_paragraphs(paragraphs, 2)) == 2
    assert len(group_paragraphs(paragraphs, 3)) == 2  # last chunk is short
    assert len(group_paragraphs(paragraphs, 99)) == 1


def test_grouping_widths_add_up(paragraphs):
    chunks = group_paragraphs(paragraphs, 2)
    assert [chunk.shape[1] for chunk in chunks] == [12, 10]


def test_grouping_rejects_non_positive(paragraphs):
    with pytest.raises(ValueError, match="per_chunk must be positive"):
        group_paragraphs(paragraphs, 0)


# --------------------------------------------------------------------------
# tokenize_record
# --------------------------------------------------------------------------


def test_tokenize_record_keeps_every_paragraph(record):
    document = tokenize_record(record, StubTokenizer())
    assert isinstance(document, Document)
    assert len(document.paragraphs) == len(record["paragraphs"])


def test_distractors_are_kept(record):
    """A cache holds what retrieval returned, relevant or not."""
    document = tokenize_record(record, StubTokenizer())
    assert len(document.paragraphs) == 3
    assert sum(p["is_supporting"] for p in record["paragraphs"]) == 1


def test_only_the_first_paragraph_carries_bos(record):
    document = tokenize_record(record, StubTokenizer())
    assert document.paragraphs[0][0, 0] == BOS
    for paragraph in document.paragraphs[1:]:
        assert paragraph[0, 0] != BOS


def test_hops_come_from_the_record_id(record):
    document = tokenize_record(record, StubTokenizer())
    assert document.hops == "3hop1"
    assert document.record_id == "3hop1__12345_678"


def test_both_question_framings_are_produced(record):
    """Same question, two framings, so Phase 3 (does query concentration
    survive absorption) can report whether the format matters."""
    document = tokenize_record(record, StubTokenizer())
    assert document.question_plain.shape[1] > 0
    assert document.question_chat.shape[1] > 0
    assert not torch.equal(document.question_plain, document.question_chat)


def test_paragraph_shapes_are_batch_first(record):
    document = tokenize_record(record, StubTokenizer())
    for paragraph in document.paragraphs:
        assert paragraph.ndim == 2
        assert paragraph.shape[0] == 1
    assert document.question_plain.shape[0] == 1
