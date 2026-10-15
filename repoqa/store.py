"""Single-file SQLite store: chunks, a weighted FTS5 lexical index, a symbol/call graph and embeddings.

One file per indexed repo (<repo>/.repoqa/index.db). No server to run, safe to delete, and it rebuilds
incrementally: only files whose content hash changed are re-chunked and re-embedded.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np

from .chunker import Chunk
from .terms import code_terms

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, hash TEXT NOT NULL, lang TEXT, n_chunks INT, indexed_at REAL);
CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY, path TEXT NOT NULL, start_line INT, end_line INT,
    lang TEXT, kind TEXT, symbol TEXT, text TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_chunks_path ON chunks(path);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(path_terms, symbol_terms, body_terms, tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS defs(short TEXT NOT NULL, symbol TEXT NOT NULL, chunk_id INT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_defs_short ON defs(short);
CREATE INDEX IF NOT EXISTS ix_defs_chunk ON defs(chunk_id);
CREATE TABLE IF NOT EXISTS calls(chunk_id INT NOT NULL, name TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_calls_name ON calls(name);
CREATE INDEX IF NOT EXISTS ix_calls_chunk ON calls(chunk_id);
CREATE TABLE IF NOT EXISTS vecs(chunk_id INTEGER PRIMARY KEY, v BLOB NOT NULL);
"""

# bm25 column weights: path (+ enclosing class), own symbol name, body.
# Matching a definition's own name matters more than a stray mention in a body.
BM25_WEIGHTS = (2.0, 5.0, 1.0)
SCHEMA_VERSION = "2"   # bump when tokenization or table layout changes; forces a rebuild


def short_name(symbol: str) -> str:
    return symbol.split(" (part")[0].rsplit(".", 1)[-1].split(" for ")[-1]


def _qualifier(symbol: str | None) -> str:
    """`Session.send` -> `Session` (enclosing class goes in the lower-weight column)."""
    if not symbol:
        return ""
    base = symbol.split(" (part")[0]
    return base.rsplit(".", 1)[0] if "." in base else ""


class Store:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        ver = None
        try:
            ver = self.db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
        except sqlite3.OperationalError:
            pass
        if ver is not None and ver[0] != SCHEMA_VERSION:
            for t in ("meta", "files", "chunks", "chunks_fts", "defs", "calls", "vecs"):
                self.db.execute(f"DROP TABLE IF EXISTS {t}")
        self.db.executescript(SCHEMA)
        self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('schema', ?)", (SCHEMA_VERSION,))
        self._mat: tuple[int, np.ndarray, np.ndarray] | None = None   # (generation, ids, matrix)

    # ---------------------------------------------------------------- meta
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, str(value)))

    def generation(self) -> int:
        return int(self.get_meta("generation", "0"))

    def bump_generation(self) -> None:
        self.set_meta("generation", self.generation() + 1)

    # ---------------------------------------------------------------- writes
    def file_hashes(self) -> dict[str, str]:
        return {r[0]: r[1] for r in self.db.execute("SELECT path, hash FROM files")}

    def delete_file(self, path: str) -> None:
        with self.lock:
            ids = [r[0] for r in self.db.execute("SELECT id FROM chunks WHERE path=?", (path,))]
            self.db.execute("BEGIN")
            if ids:
                q = ",".join("?" * len(ids))
                for sql in (f"DELETE FROM chunks_fts WHERE rowid IN ({q})", f"DELETE FROM defs WHERE chunk_id IN ({q})",
                            f"DELETE FROM calls WHERE chunk_id IN ({q})", f"DELETE FROM vecs WHERE chunk_id IN ({q})",
                            f"DELETE FROM chunks WHERE id IN ({q})"):
                    self.db.execute(sql, ids)
            self.db.execute("DELETE FROM files WHERE path=?", (path,))
            self.db.execute("COMMIT")

    def add_chunks(self, chunks: list[Chunk]) -> list[int]:
        ids: list[int] = []
        with self.lock:
            self.db.execute("BEGIN")
            for c in chunks:
                cur = self.db.execute(
                    "INSERT INTO chunks(path, start_line, end_line, lang, kind, symbol, text) VALUES(?,?,?,?,?,?,?)",
                    (c.path, c.start, c.end, c.lang, c.kind, c.symbol, c.text))
                cid = cur.lastrowid
                ids.append(cid)
                self.db.execute(
                    "INSERT INTO chunks_fts(rowid, path_terms, symbol_terms, body_terms) VALUES(?,?,?,?)",
                    (cid, " ".join(code_terms(c.path + " " + _qualifier(c.symbol))),
                     " ".join(code_terms(short_name(c.symbol) if c.symbol else "")),
                     " ".join(code_terms(c.text))))
                if c.symbol and c.kind in ("function", "class"):
                    self.db.executemany("INSERT INTO defs(short, symbol, chunk_id) VALUES(?,?,?)",
                                        [(short_name(x), x, cid) for x in [c.symbol, *c.inner]])
                if c.calls:
                    self.db.executemany("INSERT INTO calls(chunk_id, name) VALUES(?,?)",
                                        [(cid, n) for n in c.calls])
            self.db.execute("COMMIT")
        return ids

    def add_vectors(self, ids: list[int], vecs: np.ndarray) -> None:
        vecs = np.asarray(vecs, dtype=np.float32)
        with self.lock:
            self.db.execute("BEGIN")
            self.db.executemany("INSERT OR REPLACE INTO vecs(chunk_id, v) VALUES(?,?)",
                                [(i, v.tobytes()) for i, v in zip(ids, vecs)])
            self.db.execute("COMMIT")

    def mark_file(self, path: str, digest: str, lang: str | None, n_chunks: int, ts: float) -> None:
        self.db.execute("INSERT OR REPLACE INTO files(path, hash, lang, n_chunks, indexed_at) VALUES(?,?,?,?,?)",
                        (path, digest, lang, n_chunks, ts))

    def clear(self) -> None:
        with self.lock:
            self.db.executescript("""DELETE FROM chunks; DELETE FROM chunks_fts; DELETE FROM defs;
                                     DELETE FROM calls; DELETE FROM vecs; DELETE FROM files;""")
            self.bump_generation()

    # ---------------------------------------------------------------- reads
    def chunks_by_ids(self, ids: list[int]) -> dict[int, sqlite3.Row]:
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        return {r["id"]: r for r in self.db.execute(f"SELECT * FROM chunks WHERE id IN ({q})", ids)}

    def fts_search(self, terms: list[str], limit: int) -> list[int]:
        if not terms:
            return []
        query = " OR ".join(f'"{t}"' for t in terms)
        w = ", ".join(str(x) for x in BM25_WEIGHTS)
        sql = f"SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts, {w}) LIMIT ?"
        try:
            return [r[0] for r in self.db.execute(sql, (query, limit))]
        except sqlite3.OperationalError:
            return []

    def matrix(self) -> tuple[np.ndarray, np.ndarray]:
        gen = self.generation()
        if self._mat is None or self._mat[0] != gen:
            rows = self.db.execute("SELECT chunk_id, v FROM vecs ORDER BY chunk_id").fetchall()
            if rows:
                ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
                mat = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
            else:
                ids, mat = np.zeros(0, dtype=np.int64), np.zeros((0, 1), dtype=np.float32)
            self._mat = (gen, ids, mat)
        return self._mat[1], self._mat[2]

    def defs_for(self, names: list[str], exact_case: bool = True) -> list[int]:
        if not names:
            return []
        q = ",".join("?" * len(names))
        if exact_case:
            sql = f"SELECT chunk_id FROM defs WHERE short IN ({q}) OR symbol IN ({q})"
            return [r[0] for r in self.db.execute(sql, names + names)]
        low = [n.lower() for n in names]
        sql = f"SELECT chunk_id FROM defs WHERE lower(short) IN ({q}) OR lower(symbol) IN ({q})"
        return [r[0] for r in self.db.execute(sql, low + low)]

    def callees(self, chunk_id: int) -> list[int]:
        """Chunks defining functions that `chunk_id` calls."""
        # Skip ambiguous names (defined in many places, e.g. `get`) and order by how specific the callee is:
        # a helper called from 40 places tells the LLM little; one called from 2 places is likely the point.
        return [r[0] for r in self.db.execute(
            """SELECT d.chunk_id FROM calls c JOIN defs d ON d.short = c.name
               WHERE c.chunk_id = ? AND d.chunk_id != ?
                 AND (SELECT count(*) FROM defs d2 WHERE d2.short = c.name) <= 3
               GROUP BY d.chunk_id
               ORDER BY (SELECT count(*) FROM calls c2 WHERE c2.name = c.name) ASC""",
            (chunk_id, chunk_id))]

    def callers(self, chunk_id: int, limit: int = 5) -> list[int]:
        """Chunks that call the function/class defined in `chunk_id`."""
        return [r[0] for r in self.db.execute(
            "SELECT DISTINCT c.chunk_id FROM defs d JOIN calls c ON c.name = d.short "
            "WHERE d.chunk_id=? AND c.chunk_id != ? LIMIT ?", (chunk_id, chunk_id, limit))]

    def chunks_in_path(self, path_fragment: str, limit: int = 5) -> list[int]:
        return [r[0] for r in self.db.execute(
            "SELECT id FROM chunks WHERE path LIKE ? ORDER BY length(path), start_line LIMIT ?",
            (f"%{path_fragment}", limit))]

    def all_paths(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT path FROM files ORDER BY path")]

    def stats(self) -> dict:
        one = lambda sql: self.db.execute(sql).fetchone()[0]
        langs = dict(self.db.execute("SELECT lang, count(*) FROM chunks GROUP BY lang ORDER BY 2 DESC").fetchall())
        return {"files": one("SELECT count(*) FROM files"), "chunks": one("SELECT count(*) FROM chunks"),
                "symbols": one("SELECT count(*) FROM defs"), "call_edges": one("SELECT count(*) FROM calls"),
                "vectors": one("SELECT count(*) FROM vecs"), "embedder": self.get_meta("embedder"),
                "chunks_by_lang": langs}
