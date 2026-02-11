from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional

from sentence_transformers import SentenceTransformer


EmbeddingDevice = Literal["auto", "cpu", "cuda"]


@dataclass
class Embedder:
    model_name: str
    device: EmbeddingDevice = "auto"

    _model: SentenceTransformer | None = None

    def _resolve_device(self) -> str:
        """
        Resolve which device to run embeddings on.

        - cpu: always CPU
        - cuda: require CUDA if available, otherwise fall back to CPU
        - auto: use CUDA if available, else CPU
        """
        dev = (self.device or "auto").strip().lower()
        if dev == "cpu":
            return "cpu"

        # Try to detect CUDA availability. Keep this import local so we don't
        # make torch a hard import at module import time.
        cuda_ok = False
        try:
            import torch  # noqa: WPS433

            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False

        if dev == "cuda":
            return "cuda" if cuda_ok else "cpu"
        # auto
        return "cuda" if cuda_ok else "cpu"

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            device = self._resolve_device()
            try:
                self._model = SentenceTransformer(self.model_name, device=device)
            except Exception:
                # Safety fallback: if CUDA init fails for any reason, fall back to CPU
                # rather than breaking the working system.
                self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        model = self._load()
        # normalize_embeddings improves cosine similarity behavior.
        embs = model.encode(list(texts), normalize_embeddings=True)
        return embs.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

