
"""
Embedding-Provider — BGE-small-en-v1.5.

Wichtig fuer BGE-Modelle:
    Fuer RETRIEVAL-Queries empfiehlt BAAI, einen Instruction-Prefix
    voranzustellen. Fuer Dokumente (Passages) wird KEIN Prefix verwendet.
    Das verbessert die Retrieval-Qualitaet messbar.
"""
import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingProvider:
    """Kapselt ein SentenceTransformer-Modell."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._dimension = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            logger.info("Lade Embedding-Modell: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info("Modell geladen - Dimension: %d", self._dimension)

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        return self._dimension

    def encode_documents(self, texts, batch_size=32, show_progress=True):
        self._ensure_loaded()
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    def encode_query(self, query: str):
        self._ensure_loaded()
        text = BGE_QUERY_PREFIX + query
        return self._model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    return EmbeddingProvider(model_name=settings.embedding_model)
