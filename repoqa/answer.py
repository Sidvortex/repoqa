"""Grounded answering: numbered code context in, cited answer out, citations checked afterwards."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Iterator

from .llm import LLM, Message
from .retrieve import Hit, Retriever

SYSTEM_PROMPT = """You are a senior engineer answering questions about a codebase. You are given numbered excerpts \
retrieved from the repository; they are your only source of truth.

Rules:
- Cite every claim about the code as [path:start-end], using the file paths and line numbers shown in the excerpts. \
Narrow the range to the lines that matter.
- If the excerpts don't contain enough to answer, say so plainly, say what is missing, and name the files or \
symbols that probably hold it. Never invent functions, files, or behaviour.
- Name concrete functions, classes and files. Quote at most a few lines of code when it helps.
- Be concise: lead with the direct answer, then the supporting detail."""

_CITE = re.compile(r"\[([\w@+\-./]+?):(\d+)(?:\s*[-–]\s*(\d+))?\]")


def format_context(hits: list[Hit], budget_chars: int) -> tuple[str, list[Hit]]:
    blocks, used, total = [], [], 0
    for i, h in enumerate(hits, 1):
        numbered = "\n".join(f"{n:>5}| {line}" for n, line in enumerate(h.text.splitlines(), h.start))
        label = f" {h.kind} `{h.symbol}`" if h.symbol else f" {h.kind}"
        note = " (related via call graph)" if h.via == ["graph"] else ""
        block = f"### [{i}] {h.path}:{h.start}-{h.end}{label}{note}\n```{h.lang}\n{numbered}\n```"
        if total + len(block) > budget_chars and used:
            break
        blocks.append(block)
        used.append(h)
        total += len(block)
    return "\n\n".join(blocks), used


@dataclass
class Citation:
    raw: str
    path: str
    start: int
    end: int
    valid: bool


def check_citations(answer: str, hits: list[Hit]) -> list[Citation]:
    """A citation is valid if it points at a file and line range that was actually in the context."""
    out = []
    for m in _CITE.finditer(answer):
        path, a = m.group(1).lstrip("./"), int(m.group(2))
        b = int(m.group(3) or a)
        ok = any(h.path == path and a <= h.end and b >= h.start for h in hits)
        out.append(Citation(m.group(0), path, a, b, ok))
    return out


@dataclass
class AnswerResult:
    question: str
    answer: str = ""
    hits: list[Hit] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def invalid_citations(self) -> list[Citation]:
        return [c for c in self.citations if not c.valid]


def retrieval_query(question: str, history: list[Message] | None) -> str:
    """Follow-ups like "and who calls it?" retrieve badly alone, so prepend the previous user question."""
    if history:
        prev = [m["content"] for m in history if m["role"] == "user"]
        if prev and len(question.split()) < 12:
            return f"{prev[-1]}\n{question}"
    return question


def ask(question: str, retriever: Retriever, llm: LLM, *, k: int = 8, context_chars: int = 24_000,
        history: list[Message] | None = None, mode: str = "hybrid", rerank: bool = True,
        graph: bool = True) -> Iterator[tuple[str, object]]:
    """Yields ("sources", [Hit]), then ("token", str)..., then ("done", AnswerResult)."""
    res = AnswerResult(question)
    t0 = time.perf_counter()
    hits = retriever.search(retrieval_query(question, history), k=k, mode=mode, rerank=rerank, graph=graph)
    res.timings["retrieve"] = time.perf_counter() - t0
    context, used = format_context(hits, context_chars)
    res.hits = used
    yield "sources", used

    if not used:
        res.answer = "Nothing in the index matched this question. Is the repo indexed (`repoqa index`)?"
        yield "token", res.answer
        res.timings["total"] = time.perf_counter() - t0
        yield "done", res
        return

    msgs: list[Message] = list(history or [])[-6:]
    msgs.append({"role": "user", "content": f"Repository excerpts:\n\n{context}\n\n---\nQuestion: {question}"})
    t1 = time.perf_counter()
    first = None
    parts: list[str] = []
    for piece in llm.stream(SYSTEM_PROMPT, msgs):
        if first is None:
            first = time.perf_counter() - t1
        parts.append(piece)
        yield "token", piece
    res.answer = "".join(parts)
    res.timings["first_token"] = first or 0.0
    res.timings["generate"] = time.perf_counter() - t1
    res.timings["total"] = time.perf_counter() - t0
    res.citations = check_citations(res.answer, used)
    yield "done", res
