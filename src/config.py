from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
import os


LLMProvider = Literal["none", "openai", "gemini"]
VLMMode = Literal["off", "auto", "force"]
EmbeddingDevice = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True)
class Settings:
    llm_provider: LLMProvider
    openai_api_key: str
    openai_model: str
    gemini_api_key: str
    gemini_model: str

    embedding_model: str
    embedding_device: EmbeddingDevice

    data_dir: Path
    chroma_dir: Path

    tesseract_cmd: Optional[str]
    tessdata_prefix: Optional[str]

    # VLM (multimodal extract-only) controls
    vlm_mode: VLMMode
    vlm_max_pages: int


def load_settings() -> Settings:
    # Load .env if present (dev-friendly)
    load_dotenv(override=False)

    def _cuda_available() -> bool:
        try:
            import torch  # noqa: WPS433

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    llm_provider: LLMProvider = os.getenv("LLM_PROVIDER", "none").strip().lower()  # type: ignore
    if llm_provider not in ("none", "openai", "gemini"):
        llm_provider = "none"

    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    chroma_dir = Path(os.getenv("CHROMA_DIR", str(data_dir / "chroma")))

    tesseract_cmd = os.getenv("TESSERACT_CMD", "").strip() or None
    tessdata_prefix = os.getenv("TESSDATA_PREFIX", "").strip() or None

    # VLM: keep UI behavior by default (force, 25 pages), but allow env override.
    vlm_mode_raw = os.getenv("VLM_MODE", "force").strip().lower()
    vlm_mode: VLMMode = "force"
    if vlm_mode_raw in ("off", "auto", "force"):
        vlm_mode = vlm_mode_raw  # type: ignore[assignment]

    try:
        vlm_max_pages = int(os.getenv("VLM_MAX_PAGES", "25").strip())
    except Exception:
        vlm_max_pages = 25
    # Safety clamp (avoid accidental huge costs)
    vlm_max_pages = max(0, min(200, vlm_max_pages))

    embedding_device_raw = (os.getenv("EMBEDDING_DEVICE", "auto") or "auto").strip().lower()
    embedding_device: EmbeddingDevice = (
        embedding_device_raw  # type: ignore[assignment]
        if embedding_device_raw in ("auto", "cpu", "cuda")
        else "auto"
    )

    # Embedding model selection:
    # - If EMBEDDING_MODEL is explicitly set to a model name, use it.
    # - If EMBEDDING_MODEL is missing or set to "auto", choose:
    #     - CUDA available (and not forced cpu) -> multilingual-e5-base
    #     - otherwise -> multilingual-e5-small
    embedding_model_raw = (os.getenv("EMBEDDING_MODEL", "auto") or "auto").strip()
    if embedding_model_raw.lower() == "auto":
        cuda_ok = _cuda_available() and embedding_device != "cpu"
        embedding_model = "intfloat/multilingual-e5-base" if cuda_ok else "intfloat/multilingual-e5-small"
    else:
        embedding_model = embedding_model_raw

    return Settings(
        llm_provider=llm_provider,
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        embedding_model=embedding_model,
        embedding_device=embedding_device,
        data_dir=data_dir,
        chroma_dir=chroma_dir,
        tesseract_cmd=tesseract_cmd,
        tessdata_prefix=tessdata_prefix,
        vlm_mode=vlm_mode,
        vlm_max_pages=vlm_max_pages,
    )

