"""Shared retrieval contract.

Every retriever (dense, sparse/BM25, graph) returns the same type and
exposes the same method, so fusion and orchestration code can treat them
interchangeably. High-level code depends on this interface, not on any
concrete backend -- swap BM25 for Elasticsearch by writing one new class
that fits `Retriever`; nothing downstream changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class RetrievedChunk:
    """One retrieved piece of text plus where it came from and how it scored.

    `score` is retriever-specific (cosine for dense, BM25 weight for sparse,
    fused score after RRF). It's only meaningful for ranking within one
    retriever's output -- don't compare raw scores across retrievers.
    """

    text: str
    source: str
    score: float


@runtime_checkable
class Retriever(Protocol):
    """Anything with this shape is a Retriever -- no inheritance required.

    `runtime_checkable` lets us assert isinstance(x, Retriever) in tests.
    """

    def retrieve(self, query: str, limit: int) -> list[RetrievedChunk]:
        """Return up to `limit` chunks most relevant to `query`, best first."""
        ...
