"""Dense retrieval using sentence embeddings computed with Transformers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class DenseDocument:
    document_id: str
    chunk_id: int
    text: str

    @property
    def doc_identifier(self) -> str:
        return f"{self.document_id}:{self.chunk_id}"


class DenseEncoder:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: Sequence[str], batch_size: int = 8) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)
            model_output = self.model(**encoded)
            hidden_state = (
                model_output.last_hidden_state if hasattr(model_output, "last_hidden_state") else model_output[0]
            )
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            masked_hidden = hidden_state * attention_mask
            sentence_embeddings = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
            normalized = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            embeddings.append(normalized.cpu().numpy())
        return np.vstack(embeddings)


class DenseRetriever:
    def __init__(self, encoder: DenseEncoder) -> None:
        self.encoder = encoder
        self.documents: list[DenseDocument] = []
        self._embeddings: np.ndarray | None = None

    def fit(self, documents: Iterable[DenseDocument]) -> None:
        self.documents = list(documents)
        texts = [doc.text for doc in self.documents]
        if not texts:
            raise ValueError("No documents provided to DenseRetriever.fit")
        embeddings = self.encoder.encode(texts)
        self._embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

    def _ensure_embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            raise RuntimeError("Dense retriever has not been fitted")
        return self._embeddings

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        embeddings = self._ensure_embeddings()
        if not self.documents:
            return []

        query_emb = self.encoder.encode([query])
        query_f32 = np.ascontiguousarray(query_emb.astype(np.float32))[0]
        scores = embeddings @ query_f32

        k = min(top_k, len(scores))
        if k <= 0:
            return []

        top_indices = np.argpartition(-scores, k - 1)[:k]
        ordered = top_indices[np.argsort(-scores[top_indices])]

        results: list[tuple[str, float]] = []
        for idx in ordered:
            doc = self.documents[int(idx)]
            results.append((doc.doc_identifier, float(scores[idx])))
        return results

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        embeddings = self._ensure_embeddings()
        np.save(directory / "embeddings.npy", embeddings)
        with (directory / "metadata.json").open("w", encoding="utf-8") as fp:
            json.dump([doc.__dict__ for doc in self.documents], fp, ensure_ascii=False, indent=2)
        self.encoder.model.save_pretrained(directory / "encoder")
        self.encoder.tokenizer.save_pretrained(directory / "encoder")

    @classmethod
    def load(cls, directory: Path, device: str | None = None) -> "DenseRetriever":
        metadata_path = directory / "metadata.json"
        with metadata_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        documents = [DenseDocument(**doc) for doc in payload]
        embeddings_path = directory / "embeddings.npy"
        embeddings = np.load(embeddings_path)
        encoder = DenseEncoder(str(directory / "encoder"), device=device)
        retriever = cls(encoder)
        retriever.documents = documents
        retriever._embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        return retriever


def load_documents_from_jsonl(path: Path) -> list[DenseDocument]:
    documents: list[DenseDocument] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            payload = json.loads(line)
            documents.append(
                DenseDocument(
                    document_id=payload["document_id"],
                    chunk_id=int(payload["chunk_id"]),
                    text=payload["text"],
                )
            )
    return documents
