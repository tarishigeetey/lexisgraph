"""Sparse (keyword) retrieval via BM25.

Dense retrieval matches meaning but blurs exact tokens; BM25 matches exact
terms -- clause numbers, defined terms, dollar figures -- that dense fumbles.
They fail on opposite query types, which is why we fuse them (RRF) rather
than pick one. BM25 is pure-Python and needs no server or GPU.

Unlike Qdrant, BM25 holds no external index: it's built in memory from the
corpus at construction, so this retriever owns the chunks it searches.
"""

from __future__ import annotations

import logging

from rank_bm25 import BM25Okapi

from lexisgraph.ingest.chunker import Chunk
from lexisgraph.retrieval.base import RetrievedChunk

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenizer.

    Must be applied identically at index-time and query-time so tokens line
    up. Deliberately simple and predictable; a smarter tokenizer is a later
    optimization, not needed to prove the pattern.
    """
    return text.lower().split()


class SparseRetriever:
    """BM25 keyword retriever over an in-memory corpus.

    Satisfies the `Retriever` protocol: `retrieve(query, limit)` returns
    `RetrievedChunk`s, same shape as the dense retriever, so fusion treats
    them interchangeably.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("SparseRetriever needs a non-empty corpus")
        self._chunks = chunks
        # Tokenize every chunk once and build the BM25 index.
        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built over %d chunks", len(chunks))

    def retrieve(self, query: str, limit: int = 5) -> list[RetrievedChunk]:
        """Return up to `limit` chunks scored by BM25, best first."""
        scores = self._bm25.get_scores(_tokenize(query))
        # Pair each chunk with its score, sort desc, take top `limit`.
        ranked = sorted(
            zip(self._chunks, scores), key=lambda pair: pair[1], reverse=True
        )
        return [
            RetrievedChunk(text=chunk.text, source=chunk.source, score=float(score))
            for chunk, score in ranked[:limit]
        ]
