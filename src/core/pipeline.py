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

from .generation import GenerationResult, generate_answer
from .indexing import LocalIndex
from .ingestion import OCRConfig, ingest_any, IngestResult
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

    # State
    _documents: Dict[str, DocumentState] = field(default_factory=dict)
    _index: Optional[LocalIndex] = None
    _all_chunks: List[Chunk] = field(default_factory=list)

    def add_document(self, file_path: Path) -> DocumentState:
        """
        Ingest a document, build structure, create chunks, and rebuild index.
        Returns the document state.
        """
        ingest = ingest_any(file_path, ocr=self.ocr_config)
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

        ret = retrieve(self._index, query)

        return generate_answer(
            retrieval=ret,
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
        return retrieve(self._index, query)
