"""Cross-encoder reranker: the precision stage of retrieve-and-rerank.

The hybrid retriever (dense + sparse -> RRF) is a fast, recall-oriented
first stage using bi-encoders (query and doc embedded separately). This
reranker is the slow, precision-oriented second stage: a cross-encoder
that scores [query, chunk] jointly, letting every query token attend to
every document token. We only run it over the ~20 candidates the first
stage already shortlisted, so the cost stays bounded.
"""

from __future__ import annotations

import logging
from functools import cached_property

from sentence_transformers import CrossEncoder

from lexisgraph.config import get_settings

logger = logging.getLogger(__name__)


class Reranker:
    """Wraps a cross-encoder and scores query/document pairs.

    The model is loaded lazily and exactly once (see `_model`). Loading a
    transformer is expensive, so we never want it happening per-request.
    """

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        # Config is the single source of truth; allow an override for tests.
        self.model_name = model_name or settings.reranker_model
        logger.info("Reranker configured with model=%s", self.model_name)

    @cached_property
    def _model(self) -> CrossEncoder:
        """Load the cross-encoder once, on first use.

        cached_property means the first access loads and caches; every
        access after returns the same instance. Lazy loading keeps import
        cheap and startup fast until we actually rerank.
        """
        logger.info("Loading cross-encoder (first use): %s", self.model_name)
        return CrossEncoder(self.model_name)

    def score_pair(self, query: str, document: str) -> float:
        """Score a single (query, document) pair.

        Higher = more relevant. The raw logit from ms-marco models is
        unbounded (not a 0-1 probability); we only ever use it to *rank*,
        so the absolute value doesn't matter -- ordering does.
        """
        # CrossEncoder.predict expects a list of pairs and returns an array.
        score = self._model.predict([(query, document)])[0]
        return float(score)
