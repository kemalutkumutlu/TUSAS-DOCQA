"""
Full RAG pipeline — single entry point that wires:
  ingest → structure → chunk → index → retrieve → generate

Used by:
  - Chainlit UI (app.py)
  - CLI scripts
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from .generation import (
    GenerationResult,
    generate_answer,
    generate_answer_local,
    generate_chat_answer,
    generate_chat_answer_local,
    generate_extractive_answer,
)
from .local_llm import OllamaConfig
from .eventlog import JsonlEventLogger
from .indexing import LocalIndex
from .ingestion import OCRConfig, ingest_any, IngestResult
from .vlm_extract import VLMConfig
from .models import Chunk
from .retrieval import RetrievalResult, retrieve, classify_query
from .structure import build_section_tree, section_tree_to_chunks
from .utils import sha256_file


@dataclass
class DocumentState:
    """Holds parsed state for one uploaded document."""
    doc_id: str
    file_name: str
    chunks: List[Chunk]
    page_count: int
    warnings: List[str]
    build_fingerprint: str = ""


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
    embedding_device: str = "auto"
    vlm_config: Optional[VLMConfig] = None
    llm_provider: str = "gemini"  # "gemini" | "openai" | "local" | "none"
    ollama_config: Optional[OllamaConfig] = None

    # State
    _documents: Dict[str, DocumentState] = field(default_factory=dict)
    _index: Optional[LocalIndex] = None
    _all_chunks: List[Chunk] = field(default_factory=list)
    _active_doc_id: Optional[str] = None
    _session_id: str = field(default_factory=lambda: uuid4().hex)
    _logger: Optional[JsonlEventLogger] = None

    def _get_logger(self) -> Optional[JsonlEventLogger]:
        """
        Lazy-create logger from env. Logging is OFF by default.
        """
        if self._logger is not None:
            return self._logger
        self._logger = JsonlEventLogger.from_env()
        return self._logger

    @property
    def session_id(self) -> str:
        return self._session_id

    def add_document(self, file_path: Path, display_name: Optional[str] = None) -> DocumentState:
        """
        Ingest a document, build structure, create chunks, and rebuild index.
        Returns the document state.
        """
        # If the file bytes are identical AND ingestion/index settings are identical,
        # skip re-processing entirely to avoid redundant OCR/VLM + embedding work.
        #
        # doc_id is sha256(file_bytes) (see ingestion.py). Computing it here is cheap
        # compared to OCR/VLM and embeddings, and allows a fast early-exit.
        doc_id = sha256_file(file_path)

        def _fingerprint() -> str:
            o = self.ocr_config
            v = self.vlm_config
            # Keep fingerprint stable and conservative: include only settings that can
            # affect extracted text/chunking or embedding consistency.
            parts = [
                f"emb_model={self.embedding_model}",
                f"emb_dev={self.embedding_device}",
                f"ocr_enabled={bool(o.enabled)}",
                f"ocr_lang={o.lang}",
                f"tess_cmd={o.tesseract_cmd or ''}",
                f"tessdata={o.tessdata_prefix or ''}",
                f"tess_cfg={getattr(o, 'tesseract_config', None) or ''}",
            ]
            if v is None:
                parts.append("vlm=none")
            else:
                parts.extend(
                    [
                        f"vlm_provider={getattr(v, 'provider', 'gemini')}",
                        f"vlm_mode={v.mode}",
                        f"vlm_model={v.model}",
                        f"vlm_max_pages={v.max_pages}",
                        f"vlm_has_key={bool(v.api_key)}",
                    ]
                )
            return "|".join(parts)

        fp = _fingerprint()
        existing = self._documents.get(doc_id)
        if existing and (existing.build_fingerprint == fp):
            # Treat this upload as a "select active document" action.
            self._active_doc_id = doc_id
            return existing

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
            build_fingerprint=fp,
        )
        self._documents[ingest.doc_id] = state
        self._active_doc_id = ingest.doc_id

        # Update full chunk list.
        self._all_chunks = []
        for doc_state in self._documents.values():
            self._all_chunks.extend(doc_state.chunks)

        # If we couldn't extract any chunks (empty PDF, OCR missing, etc.), keep the
        # document state but avoid building an empty index (some backends reject empty upserts).
        if not self._all_chunks:
            state.warnings = list(state.warnings) + [
                "Bu belgeden metin/bolum cikarilamadi (bos veya OCR gerektiriyor olabilir)."
            ]
            self._index = None
            lg = self._get_logger()
            if lg:
                lg.log(
                    session_id=self._session_id,
                    event="doc_added",
                    payload={
                        "doc_name": state.file_name,
                        "doc_id": state.doc_id,
                        "page_count": state.page_count,
                        "chunk_count": len(state.chunks),
                        "warnings": state.warnings,
                    },
                )
            return state

        # Incremental indexing: if an index already exists AND the new doc has
        # chunks, add only the new doc's chunks (avoid re-embedding all previous).
        _t0 = time.perf_counter()
        if self._index is not None and chunks:
            self._index.add_chunks(chunks)
        else:
            self._index = LocalIndex.build(
                chunks=self._all_chunks,
                chroma_dir=self.chroma_dir,
                embedding_model=self.embedding_model,
                embedding_device=self.embedding_device,
            )
        _index_ms = (time.perf_counter() - _t0) * 1000

        lg = self._get_logger()
        if lg:
            lg.log(
                session_id=self._session_id,
                event="doc_added",
                payload={
                    "doc_name": state.file_name,
                    "doc_id": state.doc_id,
                    "page_count": state.page_count,
                    "chunk_count": len(state.chunks),
                    "index_time_ms": round(_index_ms, 1),
                    "incremental": self._index is not None and len(self._documents) > 1,
                    "warnings": state.warnings,
                },
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
    def has_index(self) -> bool:
        """True if at least one chunk is indexed and retrieval can run."""
        return self._index is not None and bool(self._all_chunks)

    @property
    def active_document_name(self) -> Optional[str]:
        """User-facing active document filename (if any)."""
        if not self._active_doc_id:
            return None
        st = self._documents.get(self._active_doc_id)
        return st.file_name if st else None

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
            # No chunks indexed (empty/scanned doc without OCR, etc.). Return safe "not found".
            empty = RetrievalResult(intent=classify_query(query), evidences=[], section_complete=False, coverage=None)
            if self.llm_provider == "none":
                result = generate_extractive_answer(retrieval=empty, query=query)
            elif self.llm_provider == "local" and self.ollama_config:
                result = generate_answer_local(
                    retrieval=empty,
                    query=query,
                    ollama_cfg=self.ollama_config,
                )
            else:
                result = generate_answer(
                    retrieval=empty,
                    query=query,
                    gemini_api_key=self.gemini_api_key,
                    gemini_model=self.gemini_model,
                )
            lg = self._get_logger()
            if lg:
                lg.log(
                    session_id=self._session_id,
                    event="qa",
                    payload={
                        "query": query,
                        "intent": empty.intent,
                        "active_doc_name": self.active_document_name,
                        "active_doc_id": self._active_doc_id,
                        "documents": self.list_documents(),
                        "doc_count": self.document_count,
                        "evidence_count": 0,
                        "section_complete": False,
                        "coverage_expected": None,
                        "coverage_actual": None,
                        "coverage_ok": None,
                        "citations_found": result.citations_found,
                        "answer": result.answer,
                        **(
                            {"context_preview": result.context_preview}
                            if (lg.include_context_preview and result.context_preview)
                            else {}
                        ),
                    },
                )
            return result

        _t_ret = time.perf_counter()
        doc_hint = self._resolve_doc_id_hint(query)
        ret = retrieve(self._index, query, doc_id=doc_hint)
        _retrieval_ms = (time.perf_counter() - _t_ret) * 1000

        _t_gen = time.perf_counter()
        if self.llm_provider == "none":
            result = generate_extractive_answer(retrieval=ret, query=query)
        elif self.llm_provider == "local" and self.ollama_config:
            result = generate_answer_local(
                retrieval=ret,
                query=query,
                ollama_cfg=self.ollama_config,
            )
        else:
            result = generate_answer(
                retrieval=ret,
                query=query,
                gemini_api_key=self.gemini_api_key,
                gemini_model=self.gemini_model,
            )
        _gen_ms = (time.perf_counter() - _t_gen) * 1000
        lg = self._get_logger()
        if lg:
            lg.log(
                session_id=self._session_id,
                event="qa",
                payload={
                    "query": query,
                    "intent": ret.intent,
                    "active_doc_name": self.active_document_name,
                    "active_doc_id": doc_hint,
                    "documents": self.list_documents(),
                    "doc_count": self.document_count,
                    "evidence_count": len(ret.evidences),
                    "section_complete": bool(ret.section_complete),
                    "coverage_expected": ret.coverage.expected_items if ret.coverage else None,
                    "coverage_actual": result.coverage_actual,
                    "coverage_ok": result.coverage_ok,
                    "citations_found": result.citations_found,
                    "answer": result.answer,
                    "retrieval_ms": round(_retrieval_ms, 1),
                    "generation_ms": round(_gen_ms, 1),
                    **(
                        {"context_preview": result.context_preview}
                        if (lg.include_context_preview and result.context_preview)
                        else {}
                    ),
                },
            )

        return result
    def chat(self, query: str) -> str:
        """
        Chat-only mode (no retrieval).
        """
        if self.llm_provider == "local" and self.ollama_config:
            return generate_chat_answer_local(
                query=query,
                ollama_cfg=self.ollama_config,
            )
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
            return RetrievalResult(intent=classify_query(query), evidences=[], section_complete=False, coverage=None)
        doc_hint = self._resolve_doc_id_hint(query)
        return retrieve(self._index, query, doc_id=doc_hint)
