# SPDX-License-Identifier: Apache-2.0
"""Drive the capture: one file per document, one rental for everything.

Run from the repo root on the GPU box::

    PYTHONPATH=src python -m absorbkv.runner --out captures/mla
    PYTHONPATH=src python -m absorbkv.runner --out captures/gqa \
        --model mistralai/Mistral-7B-Instruct-v0.2

``PYTHONPATH=src`` rather than relying on ``pip install -e .``: the editable
install writes a ``.pth`` file, and a venv built on top of a conda
interpreter was observed not to process those at all, leaving the package
unimportable while ``pip list`` still showed it. ``pip install .`` without
``-e`` also works, since that copies the package in.

Everything both halves of the measurement need is captured in one session,
because the expensive thing is the rental, not the compute (D13). A
document is a few seconds of prefill; a second booking is an afternoon.

Files are written per document and skipped if present, so an interrupted
run resumes by re-invoking the same command. Tensors are saved as plain
nested dicts rather than as the dataclasses, so a later change to those
classes cannot make an old capture unreadable.
"""

# Standard
import argparse
import json
import time
from pathlib import Path

# Third Party
import torch
from transformers import AutoModel, AutoTokenizer

# First Party
from absorbkv.capture import (
    ContextCapture,
    DocumentCapture,
    GqaContextCapture,
    QueryCapture,
    capture_document,
    capture_gqa_context,
    capture_queries,
)
from absorbkv.data import Document, group_paragraphs, load_documents

MLA_MODEL = "deepseek-ai/DeepSeek-V2-Lite"
CHUNK_SIZES = (1, 5)
DOCUMENTS = 60
SEED = 0


def context_to_dict(capture: ContextCapture) -> dict:
    return {
        "latent": capture.latent,
        "k_rope": capture.k_rope,
        "position_ids": capture.position_ids,
    }


def queries_to_dict(capture: QueryCapture) -> dict:
    return {"query": capture.query, "position_ids": capture.position_ids}


def document_to_dict(capture: DocumentCapture) -> dict:
    return {
        "truth": context_to_dict(capture.truth),
        "chunked": {
            per_chunk: {
                "capture": context_to_dict(chunked.capture),
                "chunk_starts": chunked.chunk_starts,
            }
            for per_chunk, chunked in capture.chunked.items()
        },
        "queries": {
            framing: queries_to_dict(query)
            for framing, query in capture.queries.items()
        },
    }


def gqa_to_dict(
    context: GqaContextCapture, queries: dict[str, QueryCapture]
) -> dict:
    return {
        "keys": context.keys,
        "position_ids": context.position_ids,
        "queries": {
            framing: queries_to_dict(query) for framing, query in queries.items()
        },
    }


def capture_mla_document(model, document: Document) -> dict:
    """One MLA document: one truth pass, one chunked pass per chunk size."""
    chunkings = {
        per_chunk: group_paragraphs(document.paragraphs, per_chunk)
        for per_chunk in CHUNK_SIZES
    }
    questions = {
        "plain": document.question_plain,
        "chat": document.question_chat,
    }
    return document_to_dict(capture_document(model, chunkings, questions))


def capture_gqa_document(model, document: Document) -> dict:
    """One baseline document: keys for the context, queries for the question.

    No chunked pass. The baseline measures relevance, not staleness, so
    there is nothing to compare a fragmented cache against.
    """
    device = next(model.parameters()).device
    context_ids = torch.cat(document.paragraphs, dim=1).to(device)
    positions = torch.arange(context_ids.shape[1], device=device).unsqueeze(0)
    context = capture_gqa_context(model, context_ids, positions)

    num_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // num_heads
    )
    queries = {}
    for framing, question_ids in (
        ("plain", document.question_plain),
        ("chat", document.question_chat),
    ):
        question = question_ids.to(device)
        full_ids = torch.cat([context_ids, question], dim=1)
        full_positions = torch.arange(
            full_ids.shape[1], device=device
        ).unsqueeze(0)
        queries[framing] = capture_queries(
            model,
            full_ids,
            full_positions,
            question.shape[1],
            num_heads,
            head_dim,
        )
    return gqa_to_dict(context, queries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=MLA_MODEL)
    parser.add_argument("--documents", type=int, default=DOCUMENTS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    is_mla = args.model == MLA_MODEL

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    documents = load_documents(tokenizer, args.documents, args.seed)
    print(f"{len(documents)} documents, seed {args.seed}", flush=True)

    model = (
        AutoModel.from_pretrained(args.model, dtype=torch.bfloat16)
        .to(args.device)
        .eval()
    )
    print(f"loaded {args.model} on {args.device}", flush=True)

    manifest = {
        "model": args.model,
        "documents": args.documents,
        "seed": args.seed,
        "chunk_sizes": list(CHUNK_SIZES) if is_mla else [],
        "framings": ["plain", "chat"],
        "dtype": "bfloat16",
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    started = time.monotonic()
    for index, document in enumerate(documents):
        path = args.out / f"{index:03d}_{document.record_id}.pt"
        if path.exists():
            print(f"[{index:3d}] exists, skipping", flush=True)
            continue

        capture = (
            capture_mla_document(model, document)
            if is_mla
            else capture_gqa_document(model, document)
        )
        torch.save(capture, path)

        tokens = sum(ids.shape[1] for ids in document.paragraphs)
        size = path.stat().st_size / 1e6
        elapsed = time.monotonic() - started
        print(
            f"[{index:3d}] {document.hops:6s} {tokens:5d} tokens "
            f"-> {size:7.1f} MB  ({elapsed:6.1f}s elapsed)",
            flush=True,
        )

    total = sum(f.stat().st_size for f in args.out.glob("*.pt")) / 1e9
    print(f"done: {total:.1f} GB in {args.out}", flush=True)


if __name__ == "__main__":
    main()
