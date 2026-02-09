from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
import os


LLMProvider = Literal["none", "openai", "gemini"]


@dataclass(frozen=True)
class Settings:
    llm_provider: LLMProvider
    openai_api_key: str
    openai_model: str
    gemini_api_key: str
    gemini_model: str

    embedding_model: str

    data_dir: Path
    chroma_dir: Path

    tesseract_cmd: Optional[str]


def load_settings() -> Settings:
    # Load .env if present (dev-friendly)
    load_dotenv(override=False)

    llm_provider: LLMProvider = os.getenv("LLM_PROVIDER", "none").strip().lower()  # type: ignore
    if llm_provider not in ("none", "openai", "gemini"):
        llm_provider = "none"

    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    chroma_dir = Path(os.getenv("CHROMA_DIR", str(data_dir / "chroma")))

    tesseract_cmd = os.getenv("TESSERACT_CMD", "").strip() or None

    return Settings(
        llm_provider=llm_provider,
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
        data_dir=data_dir,
        chroma_dir=chroma_dir,
        tesseract_cmd=tesseract_cmd,
    )

