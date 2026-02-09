from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from sentence_transformers import SentenceTransformer


@dataclass
class Embedder:
    model_name: str

    _model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        model = self._load()
        # normalize_embeddings improves cosine similarity behavior.
        embs = model.encode(list(texts), normalize_embeddings=True)
        return embs.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

