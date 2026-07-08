"""Split long contract text into overlapping chunks.

Contracts are far too long to embed whole — the embedding model has a token
limit and, more importantly, a single vector for a 50-page contract is
meaningless. We slide a window over the text with overlap so that a clause
sitting on a boundary still appears intact in at least one chunk.

Chunking on paragraph/sentence boundaries (not mid-word) keeps chunks
readable, which matters when they get shown to the LLM as context.
"""

from __future__ import annotations

from dataclasses import dataclass

from lexisgraph.config import get_settings


@dataclass
class Chunk:
    text: str
    source: str        # filename the chunk came from
    chunk_index: int   # position within that document


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def chunk_text(
    text: str,
    source: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    paragraphs = _split_paragraphs(text)
    chunks: list[Chunk] = []
    buffer = ""
    idx = 0

    for para in paragraphs:
        # If adding this paragraph would overflow, flush the buffer first.
        if buffer and len(buffer) + len(para) + 2 > chunk_size:
            chunks.append(Chunk(text=buffer.strip(), source=source, chunk_index=idx))
            idx += 1
            # Start next buffer with a tail of the previous one (overlap).
            buffer = buffer[-overlap:] if overlap else ""
        buffer = f"{buffer}\n\n{para}" if buffer else para

    if buffer.strip():
        chunks.append(Chunk(text=buffer.strip(), source=source, chunk_index=idx))

    return chunks
