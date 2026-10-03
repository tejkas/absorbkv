# SPDX-License-Identifier: Apache-2.0
"""MuSiQue documents, tokenized into the shape the capture expects.

Each MuSiQue record is a question plus 20 paragraphs, two to four of which
support the answer and the rest of which are distractors. Distractors are
kept: a cache holds whatever retrieval returned, and relevance is only an
interesting question when some of it is irrelevant.

Paragraphs are tokenized *separately*, which is what a cache does -- each
passage is encoded once, on its own, and reused wherever it lands. They
therefore tile the context exactly, and grouping them into larger chunks
is a pure function of the token ids (:func:`group_paragraphs`), so chunk
size is a parameter of the capture rather than of the data.

The question is produced in two framings, plain and chat-templated, to
measure whether prompt format changes the query vectors. Both sit after
an identical context, so the context capture is shared between them. That
is a deliberate simplification: a real chat prompt would also wrap the
context in role markers, which would change the context tokens too.
"""

# Standard
import random
from dataclasses import dataclass

# Third Party
import torch

DATASET = "dgslibisey/MuSiQue"


@dataclass(frozen=True)
class Document:
    """One tokenized record.

    Attributes:
        record_id: MuSiQue's own id, e.g. ``2hop__128801_205185``.
        hops: The reasoning depth encoded in that id, e.g. ``2hop``.
            Worth carrying: more hops plausibly means more of the answer
            depends on information spread across paragraphs.
        paragraphs: Token ids per paragraph, each ``[1, paragraph_tokens]``,
            in dataset order. The first carries the beginning-of-sequence
            token, so concatenating them is a well-formed prompt.
        question_plain: ``Question: ... Answer:``, ``[1, question_tokens]``.
        question_chat: The same question in the model's chat framing,
            ``User: ... Assistant:``.
    """

    record_id: str
    hops: str
    paragraphs: list[torch.Tensor]
    question_plain: torch.Tensor
    question_chat: torch.Tensor


def group_paragraphs(
    paragraphs: list[torch.Tensor], per_chunk: int
) -> list[torch.Tensor]:
    """Concatenate paragraphs into chunks of ``per_chunk`` paragraphs.

    The last chunk is short when the count does not divide evenly. Chunk
    size decides how much context a token already had when it was cached,
    so it decides the size of the deviations the whole measurement sees:
    one paragraph per chunk is the most fragmented cache and the largest
    deviations, and grouping shrinks both.

    Args:
        paragraphs: Token ids per paragraph, each ``[1, tokens]``.
        per_chunk: How many paragraphs go into one chunk.

    Returns:
        Token ids per chunk, each ``[1, chunk_tokens]``, tiling the input
        exactly and in order.

    Raises:
        ValueError: If ``per_chunk`` is not positive.
    """
    if per_chunk <= 0:
        raise ValueError(f"per_chunk must be positive, got {per_chunk}")

    return [
        torch.cat(paragraphs[start : start + per_chunk], dim=1)
        for start in range(0, len(paragraphs), per_chunk)
    ]


def tokenize_record(record: dict, tokenizer) -> Document:
    """Turn one raw MuSiQue record into a :class:`Document`.

    Separated from loading so it can be tested without the network.

    Args:
        record: A MuSiQue record, with ``id``, ``question`` and
            ``paragraphs`` (each having ``title`` and ``paragraph_text``).
        tokenizer: The model's tokenizer.

    Returns:
        The tokenized document.
    """

    def encode(text: str) -> torch.Tensor:
        return torch.tensor(
            [tokenizer(text, add_special_tokens=False)["input_ids"]],
            dtype=torch.long,
        )

    paragraphs = [
        encode(f"{item['title']}\n{item['paragraph_text']}\n\n")
        for item in record["paragraphs"]
    ]
    bos = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long)
    paragraphs[0] = torch.cat([bos, paragraphs[0]], dim=1)

    question = record["question"]
    return Document(
        record_id=record["id"],
        hops=record["id"].split("__")[0],
        paragraphs=paragraphs,
        question_plain=encode(f"Question: {question}\nAnswer:"),
        question_chat=encode(f"User: {question}\n\nAssistant:"),
    )


def load_documents(
    tokenizer, count: int, seed: int, split: str = "validation"
) -> list[Document]:
    """Draw ``count`` MuSiQue records at random and tokenize them.

    One draw with a stated seed, rather than several runs at different
    seeds: resampling this one sample gives the same stability check from
    more data, and pools instead of fragmenting it.

    Args:
        tokenizer: The model's tokenizer.
        count: How many records to draw.
        seed: Seed for the draw. Record it with the results.
        split: Which MuSiQue split to draw from.

    Returns:
        Tokenized documents, in the order drawn.

    Raises:
        ValueError: If ``count`` exceeds the split's size.
    """
    # Third Party
    from datasets import load_dataset

    records = load_dataset(DATASET, split=split)
    if count > len(records):
        raise ValueError(
            f"asked for {count} records but {split} holds {len(records)}"
        )

    chosen = random.Random(seed).sample(range(len(records)), count)
    return [tokenize_record(records[index], tokenizer) for index in chosen]
