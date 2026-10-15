"""Wires a repo's index to the embedder it was built with, the reranker and the LLM."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .embed import get_embedder, get_reranker
from .indexer import index_dir, open_store
from .llm import LLM, get_llm
from .retrieve import Retriever
from .store import Store


class NotIndexed(RuntimeError):
    pass


@dataclass
class Engine:
    root: Path
    settings: Settings
    store: Store
    retriever: Retriever
    _llm: LLM | None = None

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = get_llm(self.settings.llm, self.settings)
        return self._llm


def load_engine(root: str | Path, settings: Settings | None = None, *, rerank: bool = True) -> Engine:
    settings = settings or Settings()
    root = Path(root).resolve()
    if not (index_dir(root) / "index.db").exists():
        raise NotIndexed(f"{root} has no index yet. Run: repoqa index {root}")
    store = open_store(root)
    built_with = store.get_meta("embedder")
    if not built_with:
        raise NotIndexed(f"{root} index is empty. Run: repoqa index {root}")
    # always query with the model the vectors were made with, whatever the current setting says
    embedder = get_embedder(built_with, settings.ollama_url, settings.embed_batch)
    reranker = get_reranker(settings.reranker) if rerank else None
    retriever = Retriever(store, embedder, reranker, settings.candidates, settings.rerank_pool)
    return Engine(root, settings, store, retriever)
