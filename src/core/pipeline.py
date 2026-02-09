"""
Full RAG pipeline — single entry point that wires:
  ingest → structure → chunk → index → retrieve → generate

Used by:
  - Chainlit UI (app.py)
  - CLI scripts
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .generation import GenerationResult, generate_answer, generate_chat_answer
from .indexing import LocalIndex
from .ingestion import OCRConfig, ingest_any, IngestResult
from .vlm_extract import VLMConfig
from .models import Chunk
from .retrieval import RetrievalResult, retrieve
from .structure import build_section_tree, section_tree_to_chunks


@dataclass
class DocumentState:
    """Holds parsed state for one uploaded document."""
    doc_id: str
    file_name: str
    chunks: List[Chunk]
    page_count: int
    warnings: List[str]


@dataclass
class RAGPipeline:
    """
    Stateful RAG pipeline that manages multiple documents in one session.
    """
    # Config
    embedding_model: str
    chroma_dir: Path
    gemini_api_key: str
    gemini_model: str
    ocr_config: OCRConfig
    vlm_config: Optional[VLMConfig] = None

    # State
    _documents: Dict[str, DocumentState] = field(default_factory=dict)
    _index: Optional[LocalIndex] = None
    _all_chunks: List[Chunk] = field(default_factory=list)
    _active_doc_id: Optional[str] = None

    def add_document(self, file_path: Path, display_name: Optional[str] = None) -> DocumentState:
        """
        Ingest a document, build structure, create chunks, and rebuild index.
        Returns the document state.
        """
        ingest = ingest_any(
            file_path,
            ocr=self.ocr_config,
            display_name=display_name,
            vlm=self.vlm_config,
        )
        root = build_section_tree(ingest)
        chunks = section_tree_to_chunks(ingest, root)

        state = DocumentState(
            doc_id=ingest.doc_id,
            file_name=ingest.file_name,
            chunks=chunks,
            page_count=len(ingest.pages),
            warnings=ingest.warnings,
        )
        self._documents[ingest.doc_id] = state
        self._active_doc_id = ingest.doc_id

        # Rebuild full chunk list and index
        self._all_chunks = []
        for doc_state in self._documents.values():
            self._all_chunks.extend(doc_state.chunks)

        self._index = LocalIndex.build(
            chunks=self._all_chunks,
            chroma_dir=self.chroma_dir,
            embedding_model=self.embedding_model,
        )

        return state

    def list_documents(self) -> List[str]:
        """User-facing filenames currently loaded in this session."""
        return [st.file_name for st in self._documents.values()]

    @staticmethod
    def _normalize_doc_ref(text: str) -> str:
        """
        Normalize user text / filenames for fuzzy matching.
        Keeps only letters+digits, collapses separators, strips extension-like suffix.
        """
        s = (text or "").strip().lower()
        # Remove a common extension mention (user might omit it anyway)
        s = re.sub(r"\.(pdf|png|jpg|jpeg)\b", "", s)
        # Normalize separators to spaces
        s = re.sub(r"[_\-\.\(\)\[\]\{\}]+", " ", s)
        # Remove everything else except letters/digits (keep Turkish letters)
        s = re.sub(r"[^0-9a-zçğıöşüâîû\s]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @classmethod
    def _doc_match_score(cls, query: str, file_name: str) -> float:
        """
        Return a 0..1 confidence score that `query` refers to `file_name`.
        Conservative: exact equality/stem equality > substring > token overlap.
        """
        qn = cls._normalize_doc_ref(query)
        fn = cls._normalize_doc_ref(file_name)
        if not qn or not fn:
            return 0.0

        stem = cls._normalize_doc_ref(Path(file_name).stem)

        # Strong signals
        if qn == fn or qn == stem:
            return 1.0

        # Medium: query contains full filename/stem (common when user writes a partial phrase)
        if stem and len(stem) >= 3 and stem in qn:
            # Longer stems yield higher confidence
            return min(0.95, 0.60 + 0.35 * min(1.0, len(stem) / 18.0))
        if fn and len(fn) >= 6 and fn in qn:
            return 0.85

        # Weak: token overlap (require at least one meaningful token)
        q_tokens = {t for t in qn.split() if len(t) >= 3}
        f_tokens = {t for t in stem.split() if len(t) >= 3} or {t for t in fn.split() if len(t) >= 3}
        if not q_tokens or not f_tokens:
            return 0.0
        inter = q_tokens & f_tokens
        if not inter:
            return 0.0
        # Guard against overly-generic overlaps (e.g. only "doc").
        # If we only match a single short token (<=3) with no digits, treat as no match.
        if len(inter) == 1:
            t = next(iter(inter))
            if len(t) <= 3 and not any(ch.isdigit() for ch in t):
                return 0.0
        # Prefer higher overlap and longer tokens
        overlap = len(inter) / max(1, len(f_tokens))
        longest = max((len(t) for t in inter), default=0)
        return min(0.80, 0.25 + 0.55 * overlap + 0.05 * min(10, longest) / 10.0)

    def set_active_document(self, file_name: str) -> bool:
        """
        Set active document by (case-insensitive) filename match.
        Also supports unique partial matches (e.g., "case_study" matches "Case_Study_20260205.pdf").
        Returns True if matched, False otherwise.
        """
        target = (file_name or "").strip().lower()
        if not target:
            return False

        # 1) Exact (case-insensitive) match
        for did, st in self._documents.items():
            if (st.file_name or "").lower() == target:
                self._active_doc_id = did
                return True

        # 2) Unique fuzzy match (avoid surprising switches when ambiguous)
        scored: list[tuple[float, str]] = []
        for did, st in self._documents.items():
            sc = self._doc_match_score(target, st.file_name or "")
            if sc > 0:
                scored.append((sc, did))
        scored.sort(reverse=True, key=lambda x: x[0])
        if scored and scored[0][0] >= 0.55:
            # If top is clearly better than second, accept it.
            if len(scored) == 1 or (scored[0][0] - scored[1][0]) >= 0.12:
                self._active_doc_id = scored[0][1]
                return True
        return False

    def _resolve_doc_id_hint(self, query: str) -> Optional[str]:
        """
        Document-agnostic routing for multi-doc sessions.

        Rules:
        - If only one document is loaded → use it.
        - If query mentions a known filename (exact or partial) → route to that doc.
        - Else → route to the last uploaded / active doc (if any).
        """
        if not self._documents:
            return None
        if len(self._documents) == 1:
            return next(iter(self._documents.keys()))

        # Prefer explicit mention in the user's message.
        scored: list[tuple[float, str]] = []
        for did, st in self._documents.items():
            sc = self._doc_match_score(query, st.file_name or "")
            if sc > 0:
                scored.append((sc, did))
        scored.sort(reverse=True, key=lambda x: x[0])
        if scored and scored[0][0] >= 0.55:
            # Avoid wrong routing when ambiguous; fall back to active doc.
            if len(scored) == 1 or (scored[0][0] - scored[1][0]) >= 0.12:
                return scored[0][1]
        return self._active_doc_id

    @property
    def has_documents(self) -> bool:
        return bool(self._documents)

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def total_chunks(self) -> int:
        return len(self._all_chunks)

    def ask(self, query: str) -> GenerationResult:
        """
        Full pipeline: retrieve → generate answer.
        Raises ValueError if no documents have been indexed.
        """
        if self._index is None:
            raise ValueError("No documents indexed. Upload a document first.")

        doc_hint = self._resolve_doc_id_hint(query)
        ret = retrieve(self._index, query, doc_id=doc_hint)

        return generate_answer(
            retrieval=ret,
            query=query,
            gemini_api_key=self.gemini_api_key,
            gemini_model=self.gemini_model,
        )

    def chat(self, query: str) -> str:
        """
        Chat-only mode (no retrieval).
        """
        return generate_chat_answer(
            query=query,
            gemini_api_key=self.gemini_api_key,
            gemini_model=self.gemini_model,
        )

    def get_retrieval(self, query: str) -> RetrievalResult:
        """
        Retrieve evidence without generation (useful for debugging).
        """
        if self._index is None:
            raise ValueError("No documents indexed.")
        doc_hint = self._resolve_doc_id_hint(query)
        return retrieve(self._index, query, doc_id=doc_hint)
