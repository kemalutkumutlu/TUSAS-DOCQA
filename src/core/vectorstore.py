from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.models.Collection import Collection

from .models import Chunk


def _chunk_metadata(c: Chunk) -> Dict[str, Any]:
    return {
        "doc_id": c.doc_id,
        "file_name": c.file_name,
        "section_id": c.section_id,
        "parent_id": c.parent_id or "",
        "heading_path": c.heading_path,
        "page_start": c.page_start,
        "page_end": c.page_end,
        "kind": c.kind,
    }


@dataclass
class ChromaStore:
    persist_dir: str
    collection_name: str = "chunks"

    _client: chromadb.PersistentClient | None = None
    _collection: Collection | None = None

    def _get_collection(self) -> Collection:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(name=self.collection_name)
        return self._collection

    def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have same length")
        col = self._get_collection()
        col.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[_chunk_metadata(c) for c in chunks],
        )

    def query(
        self,
        query_embedding: List[float],
        top_k: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> dict:
        col = self._get_collection()
        return col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def get(self, ids: List[str]) -> dict:
        col = self._get_collection()
        return col.get(ids=ids, include=["documents", "metadatas"])

