from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

from .embedding import Embedder
from .hybrid import HybridResult, rrf_fuse
from .models import Chunk
from .sparse import BM25Index
from .vectorstore import ChromaStore


@dataclass
class LocalIndex:
    """
    MVP index: Chroma (dense) + in-memory BM25 (sparse).

    - Store all chunks in Chroma (parents + children)
    - Build BM25 over child chunks only
    """

    embedder: Embedder
    store: ChromaStore
    bm25: BM25Index
    allowed_doc_ids: Set[str]

    @classmethod
    def build(
        cls,
        chunks: List[Chunk],
        chroma_dir: Path,
        embedding_model: str,
        collection_name: str = "chunks",
    ) -> "LocalIndex":
        embedder = Embedder(model_name=embedding_model)
        store = ChromaStore(persist_dir=str(chroma_dir), collection_name=collection_name)

        # Prevent stale chunks accumulating for the same doc_id(s) in the persistent store.
        doc_ids = sorted({c.doc_id for c in chunks})
        if doc_ids:
            if len(doc_ids) == 1:
                store.delete_where(where={"doc_id": doc_ids[0]})
            else:
                store.delete_where(where={"$or": [{"doc_id": did} for did in doc_ids]})

        # Embed + upsert all chunks (parents + children)
        embeddings = embedder.embed_texts([c.text for c in chunks])
        store.upsert_chunks(chunks, embeddings=embeddings)

        bm25 = BM25Index.build(chunks)
        allowed_doc_ids = {c.doc_id for c in chunks}
        return cls(embedder=embedder, store=store, bm25=bm25, allowed_doc_ids=allowed_doc_ids)

    def _where_allowed_docs(self) -> Optional[dict]:
        """
        Build a Chroma `where` filter to avoid cross-document contamination.
        We never rely on clearing the persistent store; instead we restrict queries
        to doc_ids that were used to build this index instance.
        """
        if not self.allowed_doc_ids:
            return None
        if len(self.allowed_doc_ids) == 1:
            return {"doc_id": next(iter(self.allowed_doc_ids))}
        return {"$or": [{"doc_id": did} for did in sorted(self.allowed_doc_ids)]}

    def dense_search(self, query: str, top_k: int, where: Optional[dict] = None) -> List[str]:
        qemb = self.embedder.embed_query(query)
        # Always restrict dense search to the documents in this index (unless caller supplies where)
        where_final = where or self._where_allowed_docs()
        res = self.store.query(qemb, top_k=top_k, where=where_final)
        # Chroma returns list-of-lists
        return list(res.get("ids", [[]])[0])

    @staticmethod
    def _where_doc_ids(doc_ids: Set[str]) -> Optional[dict]:
        if not doc_ids:
            return None
        if len(doc_ids) == 1:
            return {"doc_id": next(iter(doc_ids))}
        return {"$or": [{"doc_id": did} for did in sorted(doc_ids)]}

    def sparse_search(self, query: str, top_k: int, *, doc_ids: Optional[Set[str]] = None) -> List[str]:
        return [cid for cid, _ in self.bm25.search(query, top_k=top_k, doc_ids=doc_ids)]

    def hybrid_search(
        self,
        query: str,
        dense_k: int = 10,
        sparse_k: int = 10,
        final_k: int = 10,
        *,
        doc_ids: Optional[Set[str]] = None,
    ) -> HybridResult:
        doc_ids_use: Optional[Set[str]] = None
        if doc_ids:
            # Only allow doc_ids that belong to this index instance.
            filtered = set(doc_ids) & set(self.allowed_doc_ids)
            if not filtered:
                # IMPORTANT: if caller asked to restrict to doc_ids but none are allowed,
                # we must NOT fall back to an unrestricted search (that would leak other docs).
                return HybridResult(ids=[], scores={})
            doc_ids_use = filtered

        where = self._where_doc_ids(doc_ids_use) if doc_ids_use else None
        dense_ids = self.dense_search(query, top_k=dense_k, where=where)
        sparse_ids = self.sparse_search(query, top_k=sparse_k, doc_ids=doc_ids_use)
        fused = rrf_fuse(dense_ids=dense_ids, sparse_ids=sparse_ids)
        top = fused[:final_k]
        ids = [cid for cid, _ in top]
        return HybridResult(ids=ids, scores=dict(top))

