"""Dense (semantic) retrieval backed by Qdrant.

This is the first of several retrievers. It owns the Qdrant collection:
creating it, indexing chunks, and searching by vector similarity. BM25 and
Neo4j retrievers will sit alongside it and get fused later (RRF).
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from lexisgraph.config import get_settings
from lexisgraph.ingest.chunker import Chunk
from lexisgraph.retrieval.embedder import Embedder


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


class DenseRetriever:
    def __init__(self, embedder: Embedder, client: QdrantClient | None = None) -> None:
        self.settings = get_settings()
        self.embedder = embedder
        self.client = client or QdrantClient(
            host=self.settings.qdrant_host, port=self.settings.qdrant_port
        )
        self.collection = self.settings.qdrant_collection

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and not recreate:
            return
        if exists:
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=self.embedder.dim, distance=Distance.COSINE
            ),
        )

    def index(self, chunks: list[Chunk], batch_size: int = 128) -> int:
        """Embed and upsert chunks. Returns the number indexed."""
        texts = [c.text for c in chunks]
        vectors = self.embedder.encode_many(texts)
        points = [
            PointStruct(
                id=i,
                vector=vectors[i],
                payload={
                    "text": chunks[i].text,
                    "source": chunks[i].source,
                    "chunk_index": chunks[i].chunk_index,
                },
            )
            for i in range(len(chunks))
        ]
        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=points[start : start + batch_size],
            )
        return len(points)

    def search(self, query: str, limit: int = 5) -> list[RetrievedChunk]:
        qvec = self.embedder.encode_one(query)
        hits = self.client.query_points(
            collection_name=self.collection, query=qvec, limit=limit
        ).points
        return [
            RetrievedChunk(
                text=h.payload["text"], source=h.payload["source"], score=h.score
            )
            for h in hits
        ]
