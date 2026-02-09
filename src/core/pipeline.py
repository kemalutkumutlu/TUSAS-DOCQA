"""
Full RAG pipeline — single entry point that wires:
  ingest → structure → chunk → index → retrieve → generate

Used by:
  - Chainlit UI (app.py)
  - CLI scripts
"""
from __future__ import annotations

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

    def set_active_document(self, file_name: str) -> bool:
        """
        Set active document by (case-insensitive) filename match.
        Returns True if matched, False otherwise.
        """
        target = (file_name or "").strip().lower()
        if not target:
            return False
        for did, st in self._documents.items():
            if (st.file_name or "").lower() == target:
                self._active_doc_id = did
                return True
        return False

    def _resolve_doc_id_hint(self, query: str) -> Optional[str]:
        """
        Document-agnostic routing for multi-doc sessions.

        Rules:
        - If only one document is loaded → use it.
        - If query mentions a known filename → route to that doc.
        - Else → route to the last uploaded / active doc (if any).
        """
        if not self._documents:
            return None
        if len(self._documents) == 1:
            return next(iter(self._documents.keys()))

        q = query.lower()
        for did, st in self._documents.items():
            fname = (st.file_name or "").lower()
            if fname and fname in q:
                return did
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
