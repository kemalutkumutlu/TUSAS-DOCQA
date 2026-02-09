from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


TextSource = Literal["pdf_text", "ocr", "image_ocr"]


@dataclass(frozen=True)
class PageText:
    doc_id: str
    file_name: str
    page_number: int  # 1-based
    text: str
    source: TextSource


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    file_name: str
    pages: list[PageText]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    file_name: str
    section_id: str
    parent_id: Optional[str]
    heading_path: str
    page_start: int
    page_end: int
    text: str
    kind: Literal["parent", "child"]

