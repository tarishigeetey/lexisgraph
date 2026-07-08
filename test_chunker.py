"""Thin wrapper around the sentence-transformers embedding model.

Loading the model is expensive (~seconds + memory), so we load it once and
reuse. Encapsulating it here means the rest of the code never imports
sentence-transformers directly — if we later swap models or move to an API,
only this file changes.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from lexisgraph.config import get_settings


class Embedder:
    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self._model = SentenceTransformer(self.model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode_one(self, text: str) -> list[float]:
        """Embed a single string into a plain Python list (Qdrant-ready)."""
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def encode_many(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed many strings efficiently in batches."""
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return [v.tolist() for v in vectors]
