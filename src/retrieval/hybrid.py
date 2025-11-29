from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class RetrievedChunk:
    doc_identifier: str
    score: float


class HybridRetriever:
    def __init__(
        self,
        sparse_retriever,
        dense_retriever,
        top_k_sparse: int = 20,
        top_k_dense: int = 20,
        weight_sparse: float = 0.5,
        weight_dense: float = 0.5,
        rrf_k: int = 60,
        strategy: str = "weighted",
    ) -> None:
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.top_k_sparse = top_k_sparse
        self.top_k_dense = top_k_dense
        self.weight_sparse = weight_sparse
        self.weight_dense = weight_dense
        self.rrf_k = rrf_k
        self.strategy = strategy

    @staticmethod
    def _normalize(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        max_score = max(scores.values())
        if max_score == 0.0:
            return {key: 0.0 for key in scores}
        return {key: value / max_score for key, value in scores.items()}

    def _combine_weighted(self, sparse_scores: dict[str, float], dense_scores: dict[str, float]) -> dict[str, float]:
        combined: dict[str, float] = {}
        normalized_sparse = self._normalize(sparse_scores)
        normalized_dense = self._normalize(dense_scores)
        keys = set(normalized_sparse) | set(normalized_dense)
        for key in keys:
            score_sparse = normalized_sparse.get(key, 0.0)
            score_dense = normalized_dense.get(key, 0.0)
            combined[key] = self.weight_sparse * score_sparse + self.weight_dense * score_dense
        return combined

    def _combine_rrf(self, sparse_ranking: list[str], dense_ranking: list[str]) -> dict[str, float]:
        combined: dict[str, float] = {}
        for rank, doc_id in enumerate(sparse_ranking, start=1):
            combined[doc_id] = combined.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, doc_id in enumerate(dense_ranking, start=1):
            combined[doc_id] = combined.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
        return combined

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        sparse_hits = self.sparse_retriever.search(query, top_k=self.top_k_sparse)
        dense_hits = self.dense_retriever.search(query, top_k=self.top_k_dense)
        sparse_scores = {doc_id: score for doc_id, score in sparse_hits}
        dense_scores = {doc_id: score for doc_id, score in dense_hits}

        if self.strategy == "rrf":
            combined_scores = self._combine_rrf(
                [doc_id for doc_id, _ in sparse_hits],
                [doc_id for doc_id, _ in dense_hits],
            )
        else:
            combined_scores = self._combine_weighted(sparse_scores, dense_scores)

        ranked = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        return [RetrievedChunk(doc_identifier=doc_id, score=score) for doc_id, score in ranked[:top_k]]
