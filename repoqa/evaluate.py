"""Retrieval evaluation against a golden set, with per-configuration ablation.

Golden file: JSONL, one question per line:
  {"q": "Where are retries configured?", "expected": ["src/http/retry.py", "src/http/session.py::Session.send"]}

An expected item is either a path (any chunk from that file counts) or path::symbol (the chunk's symbol must
equal it or end with it). Metrics per configuration:
  hit@k     share of questions with at least one expected item in the top k
  recall@k  share of expected items found in the top k (averaged over questions)
  MRR       mean of 1/rank of the first relevant chunk
  p50/p95   retrieval latency
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .answer import ask
from .retrieve import Hit, Retriever

CONFIGS = {
    "bm25": dict(mode="bm25", graph=False, rerank=False),
    "dense": dict(mode="dense", graph=False, rerank=False),
    "hybrid": dict(mode="hybrid", graph=False, rerank=False),
    "hybrid+rerank": dict(mode="hybrid", graph=False, rerank=True),
    "hybrid+rerank+graph": dict(mode="hybrid", graph=True, rerank=True),
}


def load_golden(path: str | Path) -> list[dict]:
    items = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        obj = json.loads(line)
        if "q" not in obj or not obj.get("expected"):
            raise ValueError(f"{path}:{n} needs 'q' and a non-empty 'expected' list")
        items.append(obj)
    return items


def matches(hit: Hit, expected: str) -> bool:
    path, _, sym = expected.partition("::")
    if hit.path != path.lstrip("./"):
        return False
    if not sym:
        return True
    hs = (hit.symbol or "").split(" (part")[0]
    if hs == sym or hs.endswith("." + sym):
        return True
    # a small class kept as one chunk contains its methods: Class chunk matches Class.method
    short = sym.rsplit(".", 1)[-1]
    return hit.kind == "class" and bool(hs) and sym.startswith(hs + ".") and f"{short}(" in hit.text.replace(" (", "(")


@dataclass
class ConfigResult:
    name: str
    hit_at_k: float
    recall_at_k: float
    mrr: float
    p50_ms: float
    p95_ms: float
    misses: list[str] = field(default_factory=list)


def evaluate(retriever: Retriever, golden: list[dict], k: int = 8, configs: list[str] | None = None) -> list[ConfigResult]:
    results = []
    for name in configs or list(CONFIGS):
        cfg = CONFIGS[name]
        if cfg["rerank"] and retriever.reranker is None:
            continue
        if cfg["mode"] in ("dense", "hybrid") and retriever.embedder is None:
            continue
        hits_n, recalls, rrs, lat, misses = 0, [], [], [], []
        for item in golden:
            t0 = time.perf_counter()
            hits = retriever.search(item["q"], k=k, **cfg)
            lat.append((time.perf_counter() - t0) * 1000)
            exp = item["expected"]
            found = [e for e in exp if any(matches(h, e) for h in hits)]
            recalls.append(len(found) / len(exp))
            rank = next((i for i, h in enumerate(hits, 1) if any(matches(h, e) for e in exp)), None)
            rrs.append(1 / rank if rank else 0.0)
            if found:
                hits_n += 1
            else:
                misses.append(item["q"])
        lat.sort()
        results.append(ConfigResult(
            name, hits_n / len(golden), statistics.mean(recalls), statistics.mean(rrs),
            statistics.median(lat), lat[min(len(lat) - 1, int(0.95 * len(lat)))], misses))
    return results


def evaluate_answers(retriever: Retriever, llm, golden: list[dict], k: int = 8, context_chars: int = 24_000) -> dict:
    """End-to-end: how often answers cite at all, and how many citations point at real retrieved lines."""
    rows = []
    for item in golden:
        res = None
        for kind, payload in ask(item["q"], retriever, llm, k=k, context_chars=context_chars):
            if kind == "done":
                res = payload
        cites = res.citations
        rows.append({"q": item["q"], "n_citations": len(cites), "invalid": [c.raw for c in res.invalid_citations],
                     "cites_expected": any(any(c.path == e.partition("::")[0] for e in item["expected"]) for c in cites),
                     "seconds": res.timings.get("total", 0.0)})
    total = sum(r["n_citations"] for r in rows)
    return {
        "answered_with_citations": sum(1 for r in rows if r["n_citations"]) / len(rows),
        "citation_validity": (total - sum(len(r["invalid"]) for r in rows)) / total if total else 0.0,
        "cites_expected_file": sum(r["cites_expected"] for r in rows) / len(rows),
        "mean_seconds": statistics.mean(r["seconds"] for r in rows),
        "rows": rows,
    }


def to_json(results: list[ConfigResult]) -> list[dict]:
    return [asdict(r) for r in results]
