from __future__ import annotations

import dataclasses
from typing import Iterable, TYPE_CHECKING

from src.retrieval.bm25 import BM25Document, BM25Retriever
from .schemas import TableSchema

if TYPE_CHECKING:  # pragma: no cover
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.hybrid import HybridRetriever


@dataclasses.dataclass
class RetrievedTable:
    table: TableSchema
    score: float


class TableRetriever:
    """Lexical, dense, or hybrid retriever over table schemas."""

    def __init__(
        self,
        tables: Iterable[TableSchema],
        mode: str = "bm25",
        dense_model_name: str = "BAAI/bge-base-en",
        top_k_sparse: int = 20,
        top_k_dense: int = 20,
        hybrid_weight_sparse: float = 0.5,
        hybrid_weight_dense: float = 0.5,
        hybrid_strategy: str = "weighted",
        rrf_k: int = 60,
        device: str | None = None,
    ) -> None:
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError(f"Unsupported mode '{mode}', choose from bm25, dense, hybrid")
        self.tables = list(tables)
        self.mode = mode
        self._lookup: dict[str, TableSchema] = {}
        self._bm25 = BM25Retriever()
        self._dense: "DenseRetriever | None" = None
        self._hybrid: "HybridRetriever | None" = None
        self._fit_bm25()

        if mode in {"dense", "hybrid"}:
            from src.retrieval.dense import DenseDocument, DenseEncoder, DenseRetriever

            encoder = DenseEncoder(dense_model_name, device=device)
            dense_docs: list[DenseDocument] = []
            for idx, table in enumerate(self.tables):
                dense_docs.append(
                    DenseDocument(
                        document_id=table.identifier,
                        chunk_id=idx,
                        text=table.as_retrieval_text(),
                    )
                )
            dense_retriever = DenseRetriever(encoder)
            dense_retriever.fit(dense_docs)
            self._dense = dense_retriever

        if mode == "hybrid":
            from src.retrieval.hybrid import HybridRetriever

            assert self._dense is not None
            self._hybrid = HybridRetriever(
                sparse_retriever=self._bm25,
                dense_retriever=self._dense,
                top_k_sparse=top_k_sparse,
                top_k_dense=top_k_dense,
                weight_sparse=hybrid_weight_sparse,
                weight_dense=hybrid_weight_dense,
                rrf_k=rrf_k,
                strategy=hybrid_strategy,
            )

    def _fit_bm25(self) -> None:
        documents: list[BM25Document] = []
        for idx, table in enumerate(self.tables):
            document_id = table.identifier
            chunk_id = idx  # stable unique integer so the identifier is unique
            text = table.as_retrieval_text()
            doc = BM25Document(document_id=document_id, chunk_id=chunk_id, text=text)
            documents.append(doc)
            self._lookup[doc.doc_identifier] = table
        self._bm25.fit(documents)

    def _resolve_table(self, doc_identifier: str) -> TableSchema | None:
        return self._lookup.get(doc_identifier)

    def _filter_hits(
        self, hits: Iterable[tuple[str, float]], top_k: int, limit_to_db: str | None
    ) -> list[RetrievedTable]:
        results: list[RetrievedTable] = []
        for doc_identifier, score in hits:
            table = self._resolve_table(doc_identifier)
            if table is None:
                continue
            if limit_to_db and table.database != limit_to_db:
                continue
            results.append(RetrievedTable(table=table, score=score))
            if len(results) >= top_k:
                break
        return results

    def search(self, query: str, top_k: int = 5, limit_to_db: str | None = None) -> list[RetrievedTable]:
        """Return the most relevant tables for a natural language query."""
        if self.mode == "bm25":
            raw_hits = self._bm25.search(query, top_k=max(top_k * 3, 10))
            return self._filter_hits(raw_hits, top_k=top_k, limit_to_db=limit_to_db)

        if self.mode == "dense":
            if self._dense is None:
                raise RuntimeError("Dense retriever not initialized")
            raw_hits = self._dense.search(query, top_k=max(top_k * 3, 10))
            return self._filter_hits(raw_hits, top_k=top_k, limit_to_db=limit_to_db)

        # hybrid
        if self._hybrid is None:
            raise RuntimeError("Hybrid retriever not initialized")
        hybrid_hits = self._hybrid.search(query, top_k=max(top_k * 2, 10))
        return self._filter_hits(
            [(hit.doc_identifier, hit.score) for hit in hybrid_hits],
            top_k=top_k,
            limit_to_db=limit_to_db,
        )
