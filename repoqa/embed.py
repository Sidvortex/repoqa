"""Embedding backends. All return L2-normalized float32 vectors so dot product == cosine similarity.

- fastembed:<model>  ONNX on CPU, no PyTorch. Default: jinaai/jina-embeddings-v2-base-code (trained on code).
- ollama:<model>     e.g. ollama:nomic-embed-text, if you already run Ollama.
- hash               Feature-hashed bag of code terms. No download, no semantics; for tests and CI only.
"""
from __future__ import annotations

import hashlib

import httpx
import numpy as np

from .terms import code_terms


def _norm(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class Embedder:
    name: str

    def embed_docs(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedEmbedder(Embedder):
    def __init__(self, model: str, batch: int = 32):
        from fastembed import TextEmbedding
        self.name, self.batch = f"fastembed:{model}", batch
        self.model = TextEmbedding(model_name=model)
        self._nomic = "nomic" in model

    def embed_docs(self, texts):
        if self._nomic:
            texts = [f"search_document: {t}" for t in texts]
        return _norm(np.vstack(list(self.model.embed(texts, batch_size=self.batch))))

    def embed_query(self, text):
        if self._nomic:
            return _norm(np.vstack(list(self.model.embed([f"search_query: {text}"]))))[0]
        return _norm(np.vstack(list(self.model.query_embed(text))))[0]


class OllamaEmbedder(Embedder):
    def __init__(self, model: str, url: str, batch: int = 32):
        self.name, self.url, self.model, self.batch = f"ollama:{model}", url.rstrip("/"), model, batch
        self._nomic = "nomic" in model
        self.client = httpx.Client(timeout=300)

    def _call(self, texts):
        out = []
        for i in range(0, len(texts), self.batch):
            r = self.client.post(f"{self.url}/api/embed", json={"model": self.model, "input": texts[i:i + self.batch]})
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
        return _norm(np.array(out))

    def embed_docs(self, texts):
        return self._call([f"search_document: {t}" if self._nomic else t for t in texts])

    def embed_query(self, text):
        return self._call([f"search_query: {text}" if self._nomic else text])[0]


class HashEmbedder(Embedder):
    def __init__(self, dim: int = 1024):
        self.name, self.dim = "hash", dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        terms = code_terms(text, drop_stopwords=True)
        for t in terms + [f"{a}_{b}" for a, b in zip(terms, terms[1:])]:
            h = int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        return np.sign(v) * np.log1p(np.abs(v))

    def embed_docs(self, texts):
        return _norm(np.vstack([self._vec(t) for t in texts])) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text):
        return _norm(self._vec(text))[0]


def get_embedder(spec: str, ollama_url: str = "http://localhost:11434", batch: int = 32) -> Embedder:
    kind, _, model = spec.partition(":")
    if kind == "hash":
        return HashEmbedder()
    if kind == "fastembed":
        return FastEmbedEmbedder(model, batch)
    if kind == "ollama":
        return OllamaEmbedder(model, ollama_url, batch)
    raise ValueError(f"unknown embedder {spec!r}; use fastembed:<model>, ollama:<model> or hash")


class Reranker:
    """Cross-encoder reranking via fastembed (ONNX, CPU)."""

    def __init__(self, model: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self.name = model
        self.model = TextCrossEncoder(model_name=model)

    def scores(self, query: str, docs: list[str]) -> list[float]:
        return [float(s) for s in self.model.rerank(query, docs)]


def get_reranker(spec: str | None) -> Reranker | None:
    if not spec or spec.lower() in ("none", "off", "false", "0"):
        return None
    return Reranker(spec)
