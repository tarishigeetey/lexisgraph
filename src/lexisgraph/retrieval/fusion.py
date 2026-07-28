"""Reciprocal Rank Fusion (RRF): combine multiple ranked lists into one.

Dense and sparse retrievers score on incompatible scales -- cosine (~0.7)
vs BM25 weights (unbounded). You cannot add or average those. RRF sidesteps
this entirely by ignoring scores and using only RANK POSITION: a chunk's
fused score is the sum over each list of 1/(k+rank). Scale-free, and chunks
that BOTH retrievers rank highly get contributions from both -- so consensus
results rise to the top.

k=60 is the standard constant from the original RRF paper. It dampens how
much rank-1 dominates rank-2; larger k = flatter weighting.
"""

from __future__ import annotations

from lexisgraph.retrieval.base import RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int = 60,
    limit: int = 10,
) -> list[RetrievedChunk]:
    """Fuse several ranked lists of chunks into one ranked list.

    Args:
        ranked_lists: each inner list is one retriever's output, best first.
        k: RRF damping constant (60 = paper default).
        limit: how many fused results to return.

    Returns:
        Chunks sorted by fused score, highest first, with `score` set to
        the RRF score (NOT the original retriever score).
    """
    fused_scores: dict[tuple[str, str], float] = {}
    chunk_by_key: dict[tuple[str, str], RetrievedChunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            # Identity of "the same chunk" across lists: (source, text).
            key = (chunk.source, chunk.text)
            # THE RRF FORMULA. rank is 0-based here, so the top result
            # contributes 1/(k+0); adding across lists rewards agreement.
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank)
            chunk_by_key[key] = chunk

    ranked_keys = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)

    return [
        RetrievedChunk(
            text=chunk_by_key[key].text,
            source=chunk_by_key[key].source,
            score=fused_score,  # overwrite with the fused score, honestly labeled
        )
        for key, fused_score in ranked_keys[:limit]
    ]
