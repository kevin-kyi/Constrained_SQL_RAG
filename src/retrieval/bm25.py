"""Sparse BM25 retrieval implementation without external RAG frameworks."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rank_bm25 import BM25Okapi


@dataclass
class BM25Document:
    document_id: str
    chunk_id: int
    text: str

    @property
    def doc_identifier(self) -> str:
        return f"{self.document_id}:{self.chunk_id}"


class BM25Retriever:
    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._documents: list[BM25Document] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def fit(self, documents: Iterable[BM25Document]) -> None:
        self._documents = list(documents)
        tokenized = [self._tokenize(doc.text) for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized)

    def _ensure_fitted(self) -> None:
        if self._bm25 is None:
            raise RuntimeError("BM25 retriever is not fitted")

    def save(self, path: Path) -> None:
        self._ensure_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": [doc.__dict__ for doc in self._documents],
            "bm25": self._bm25,
        }
        with path.open("wb") as fp:
            pickle.dump(payload, fp)

    @classmethod
    def load(cls, path: Path) -> "BM25Retriever":
        with path.open("rb") as fp:
            payload = pickle.load(fp)
        retriever = cls()
        retriever._documents = [BM25Document(**doc) for doc in payload["documents"]]
        retriever._bm25 = payload["bm25"]
        return retriever

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        self._ensure_fitted()
        assert self._bm25 is not None
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
        results: list[tuple[str, float]] = []
        for idx in ranked_indices:
            doc = self._documents[idx]
            results.append((doc.doc_identifier, float(scores[idx])))
        return results

    def dump_lookup_table(self, path: Path) -> None:
        payload = {
            doc.doc_identifier: json.dumps({
                "document_id": doc.document_id,
                "chunk_id": doc.chunk_id,
                "text": doc.text,
            }, ensure_ascii=False)
            for doc in self._documents
        }
        with path.open("w", encoding="utf-8") as fp:
            for key, value in payload.items():
                fp.write(json.dumps({"id": key, "value": value}) + "\n")


def load_documents_from_jsonl(path: Path) -> list[BM25Document]:
    documents: list[BM25Document] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            payload = json.loads(line)
            documents.append(
                BM25Document(
                    document_id=payload["document_id"],
                    chunk_id=int(payload["chunk_id"]),
                    text=payload["text"],
                )
            )
    return documents
