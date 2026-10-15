"""Runtime settings. Everything can be overridden with environment variables or CLI flags."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    # "fastembed:<model>", "ollama:<model>" or "hash" (offline, no model download; for tests)
    embedder: str = field(default_factory=lambda: _env("REPOQA_EMBEDDER", "fastembed:jinaai/jina-embeddings-v2-base-code"))
    # fastembed cross-encoder name, or "none" to disable reranking
    reranker: str = field(default_factory=lambda: _env("REPOQA_RERANKER", "jinaai/jina-reranker-v1-turbo-en"))
    # "ollama:<model>", "openai:<model>" (any OpenAI-compatible server) or "anthropic:<model>"
    llm: str = field(default_factory=lambda: _env("REPOQA_LLM", "ollama:qwen2.5-coder:7b"))

    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))

    # chunking
    max_chunk_lines: int = 120     # definitions longer than this get split
    window_lines: int = 60         # window size for non-AST text and oversized bodies
    window_overlap: int = 10
    max_file_bytes: int = 1_000_000

    # retrieval / generation
    top_k: int = 8
    candidates: int = 50           # per-retriever candidate pool before fusion
    rerank_pool: int = 30          # how many fused candidates go to the cross-encoder
    context_chars: int = 24_000    # ~6-7k tokens of code context sent to the LLM
    embed_batch: int = 32
