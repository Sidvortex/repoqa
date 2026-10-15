"""Incremental indexing: hash every file, re-chunk/re-embed only what changed, drop what was deleted."""
from __future__ import annotations

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .chunker import Chunk, chunk_file, detect_lang
from .config import Settings
from .embed import Embedder, get_embedder
from .store import Store

SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env", "__pycache__", ".mypy_cache",
             ".pytest_cache", ".ruff_cache", "dist", "build", "target", ".next", ".nuxt", ".cache", "vendor",
             ".repoqa", ".idea", ".vscode", "coverage", ".tox", "site-packages", ".gradle", "out"}
SKIP_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Cargo.lock", "go.sum",
              "composer.lock", "Pipfile.lock", "uv.lock", "bun.lockb"}
SKIP_SUFFIXES = (".min.js", ".min.css", ".map", ".pb.go", "_pb2.py", ".snap", ".svg")


def index_dir(root: Path) -> Path:
    return root / ".repoqa"


def open_store(root: Path) -> Store:
    return Store(index_dir(root) / "index.db")


def _git_files(root: Path) -> list[str] | None:
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "-co", "--exclude-standard"],
                             capture_output=True, check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return [p for p in out.decode(errors="replace").split("\0") if p]


def _walk_files(root: Path) -> list[str]:
    spec = None
    gi = root / ".gitignore"
    if gi.exists():
        import pathspec
        spec = pathspec.PathSpec.from_lines("gitwildmatch", gi.read_text(errors="replace").splitlines())
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.startswith(".")]
        for f in files:
            rel = os.path.relpath(os.path.join(d, f), root)
            if spec is None or not spec.match_file(rel):
                out.append(rel)
    return out


def list_source_files(root: Path, max_bytes: int) -> list[str]:
    files = _git_files(root)
    if files is None:
        files = _walk_files(root)
    keep = []
    for rel in files:
        parts = Path(rel).parts
        if any(p in SKIP_DIRS for p in parts[:-1]) or parts[-1] in SKIP_FILES or rel.endswith(SKIP_SUFFIXES):
            continue
        if detect_lang(rel) is None:
            continue
        full = root / rel
        try:
            if not full.is_file() or full.stat().st_size > max_bytes or full.stat().st_size == 0:
                continue
        except OSError:
            continue
        keep.append(rel.replace(os.sep, "/"))
    return sorted(keep)


def _read_text(path: Path) -> tuple[str, str] | None:
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        return None
    return hashlib.blake2b(data, digest_size=16).hexdigest(), data.decode("utf-8", errors="replace")


@dataclass
class IndexStats:
    scanned: int = 0
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    chunks: int = 0
    seconds: float = 0.0
    rebuilt: bool = False
    errors: list[str] = field(default_factory=list)


def index_repo(root: str | Path, settings: Settings | None = None, *, full: bool = False,
               embedder: Embedder | None = None,
               progress: Callable[[int, int, str], None] | None = None) -> IndexStats:
    settings = settings or Settings()
    root = Path(root).resolve()
    t0 = time.perf_counter()
    stats = IndexStats()
    store = open_store(root)
    embedder = embedder or get_embedder(settings.embedder, settings.ollama_url, settings.embed_batch)

    previous = store.get_meta("embedder")
    if full or (previous and previous != embedder.name):
        store.clear()     # vectors from different models aren't comparable
        stats.rebuilt = True
    store.set_meta("embedder", embedder.name)
    store.set_meta("root", str(root))

    files = list_source_files(root, settings.max_file_bytes)
    stats.scanned = len(files)
    old = store.file_hashes()
    current = set(files)
    for p in sorted(set(old) - current):
        store.delete_file(p)
        stats.removed += 1

    pending: list[tuple[int, Chunk]] = []
    pending_files: list[tuple[str, str, str | None, int]] = []

    def flush():
        if pending:
            vecs = embedder.embed_docs([c.embed_text() for _, c in pending])
            store.add_vectors([i for i, _ in pending], vecs)
            pending.clear()
        now = time.time()
        for path, digest, lang, n in pending_files:   # only marked done once its vectors are stored
            store.mark_file(path, digest, lang, n, now)
        pending_files.clear()

    for n, rel in enumerate(files, 1):
        if progress:
            progress(n, len(files), rel)
        try:
            res = _read_text(root / rel)
        except OSError as e:
            stats.errors.append(f"{rel}: {e}")
            continue
        if res is None:
            continue
        digest, text = res
        if old.get(rel) == digest:
            stats.unchanged += 1
            continue
        store.delete_file(rel)          # also clears half-written chunks from an interrupted run
        lang = detect_lang(rel)
        chunks = chunk_file(rel, text, lang, max_lines=settings.max_chunk_lines,
                            window=settings.window_lines, overlap=settings.window_overlap)
        ids = store.add_chunks(chunks)
        pending.extend(zip(ids, chunks))
        pending_files.append((rel, digest, lang, len(chunks)))
        stats.chunks += len(chunks)
        if rel in old:
            stats.updated += 1
        else:
            stats.added += 1
        if len(pending) >= settings.embed_batch * 4:
            flush()
    flush()
    store.bump_generation()
    store.set_meta("indexed_at", time.time())
    stats.seconds = time.perf_counter() - t0
    return stats
