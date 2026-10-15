"""Hybrid retrieval: BM25 + dense + exact symbol lookup, fused with RRF, optionally reranked,
then expanded along the call graph so the LLM sees the functions a hit calls (and who calls it)."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

from .embed import Embedder, Reranker
from .store import Store
from .terms import identifiers_in, query_terms

MODES = ("bm25", "dense", "hybrid")


@dataclass
class Hit:
    chunk_id: int
    path: str
    start: int
    end: int
    lang: str
    kind: str
    symbol: str | None
    text: str
    score: float = 0.0
    via: list[str] = field(default_factory=list)   # which retrievers found it: bm25/dense/symbol/graph/rerank

    @property
    def ref(self) -> str:
        return f"{self.path}:{self.start}-{self.end}"

    def to_dict(self, with_text: bool = True) -> dict:
        d = {"path": self.path, "start": self.start, "end": self.end, "symbol": self.symbol, "kind": self.kind,
             "lang": self.lang, "score": round(self.score, 4), "via": self.via, "ref": self.ref}
        if with_text:
            d["text"] = self.text
        return d


def rrf(ranked_lists: list[tuple[str, list[int], float]], k: int = 60) -> tuple[dict[int, float], dict[int, list[str]]]:
    """Reciprocal rank fusion. Each list is (name, ids best-first, weight)."""
    scores: dict[int, float] = {}
    via: dict[int, list[str]] = {}
    for name, ids, weight in ranked_lists:
        for rank, cid in enumerate(ids):
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank + 1)
            via.setdefault(cid, []).append(name)
    return scores, via


_PATHISH = re.compile(r"[\w\-./]+\.[A-Za-z]{1,6}\b")
_TEST_PATH = re.compile(r"(^|/)(tests?|__tests__|spec|testdata)/|(^|/)test_[^/]*$|_test\.\w+$|\.(test|spec)\.\w+$|"
                        r"(^|/)conftest\.py$")


class Retriever:
    def __init__(self, store: Store, embedder: Embedder | None, reranker: Reranker | None = None,
                 candidates: int = 50, rerank_pool: int = 30):
        self.store, self.embedder, self.reranker = store, embedder, reranker
        self.candidates, self.rerank_pool = candidates, rerank_pool
        self.last_timings: dict[str, float] = {}

    # -- individual retrievers ---------------------------------------------------------------
    def bm25(self, q: str, n: int) -> list[int]:
        return self.store.fts_search(query_terms(q), n)

    def dense(self, q: str, n: int) -> list[int]:
        if self.embedder is None:
            return []
        ids, mat = self.store.matrix()
        if len(ids) == 0:
            return []
        sims = mat @ self.embedder.embed_query(q)
        n = min(n, len(ids))
        top = np.argpartition(-sims, n - 1)[:n]
        return [int(ids[i]) for i in top[np.argsort(-sims[top])]]

    def symbols(self, q: str) -> list[int]:
        """Exact matches for identifiers and file names written in the question."""
        names = identifiers_in(q)
        out = self.store.defs_for(names) or self.store.defs_for(names, exact_case=False)
        for frag in _PATHISH.findall(q):
            if "/" in frag or frag.count(".") == 1:
                out += self.store.chunks_in_path(frag.lstrip("./"), limit=3)
        return list(dict.fromkeys(out))

    # -- pipeline ----------------------------------------------------------------------------
    def search(self, q: str, k: int = 8, mode: str = "hybrid", graph: bool = True,
               rerank: bool = True) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        t = {}
        t0 = time.perf_counter()
        lists: list[tuple[str, list[int], float]] = []
        if mode in ("bm25", "hybrid"):
            lists.append(("bm25", self.bm25(q, self.candidates), 1.0))
        t["bm25"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        if mode in ("dense", "hybrid"):
            lists.append(("dense", self.dense(q, self.candidates), 1.0))
        t["dense"] = time.perf_counter() - t1
        if mode == "hybrid":
            lists.append(("symbol", self.symbols(q), 2.0))   # a named identifier is a strong signal
        scores, via = rrf(lists)
        self._apply_priors(q, scores, via)
        order = sorted(scores, key=scores.get, reverse=True)

        t2 = time.perf_counter()
        if rerank and self.reranker is not None and order:
            pool = order[: self.rerank_pool]
            rows = self.store.chunks_by_ids(pool)
            docs = [f"{rows[i]['path']} :: {rows[i]['symbol'] or ''}\n{rows[i]['text'][:2000]}" for i in pool]
            rs = self.reranker.scores(q, docs)
            # keep exact symbol hits pinned: the cross-encoder can underrate a short definition
            pinned = [i for i in pool if "symbol" in via.get(i, [])][:2]
            reranked = [i for _, i in sorted(zip(rs, pool), key=lambda x: -x[0]) if i not in pinned]
            order = pinned + reranked
            scores = {i: s for i, s in zip(pool, rs)} | {i: max(rs, default=0) + 1 for i in pinned}
            for i in pool:
                via.setdefault(i, []).append("rerank")
        t["rerank"] = time.perf_counter() - t2

        rows = self.store.chunks_by_ids(order[: k * 3])
        hits = self._dedupe([self._hit(rows[i], scores.get(i, 0.0), via.get(i, [])) for i in order[: k * 3]
                             if i in rows], k)
        if graph and hits:
            hits += self._expand(hits, budget=max(2, k // 2))
        t["total"] = time.perf_counter() - t0
        self.last_timings = t
        return hits

    def _apply_priors(self, q: str, scores: dict[int, float], via: dict[int, list[str]]) -> None:
        """Most questions are about how the code works, so tests and changelogs shouldn't crowd out source.
        Skipped when the question is itself about tests / history, or the chunk was named exactly."""
        ql = q.lower()
        wants_tests = any(w in ql for w in ("test", "spec", "fixture", "mock"))
        wants_history = any(w in ql for w in ("changelog", "history", "release", "version", "changed"))
        rows = self.store.chunks_by_ids(list(scores))
        for cid, row in rows.items():
            if "symbol" in via.get(cid, []):
                continue
            p = row["path"].lower()
            name = p.rsplit("/", 1)[-1]
            if not wants_tests and _TEST_PATH.search(p):
                scores[cid] *= 0.6
            elif not wants_history and name.split(".")[0] in ("changelog", "history", "changes", "news", "releases"):
                scores[cid] *= 0.4
            elif row["kind"] in ("section", "window") and row["lang"] in ("markdown", "text"):
                scores[cid] *= 0.85   # prose docs: useful, but below the code they describe

    def _expand(self, hits: list[Hit], budget: int) -> list[Hit]:
        have = {h.chunk_id for h in hits}
        extra: list[int] = []
        for h in hits[:3]:
            for cid in self.store.callees(h.chunk_id)[:3] + self.store.callers(h.chunk_id, limit=2):
                if cid not in have and cid not in extra:
                    extra.append(cid)
        rows = self.store.chunks_by_ids(extra[:budget])
        return [self._hit(rows[i], 0.0, ["graph"]) for i in extra[:budget] if i in rows]

    @staticmethod
    def _hit(row, score: float, via: list[str]) -> Hit:
        return Hit(row["id"], row["path"], row["start_line"], row["end_line"], row["lang"], row["kind"],
                   row["symbol"], row["text"], score, list(dict.fromkeys(via)))

    @staticmethod
    def _dedupe(hits: list[Hit], k: int) -> list[Hit]:
        out: list[Hit] = []
        for h in hits:
            if any(o.path == h.path and o.start <= h.start and h.end <= o.end for o in out):
                continue   # fully inside a chunk we already have (overlapping windows)
            out.append(h)
            if len(out) == k:
                break
        return out
