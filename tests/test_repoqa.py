"""Offline tests: hash embedder, no reranker, fake LLM. Run with `pytest -q`."""
from __future__ import annotations

import json
import subprocess
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from repoqa.answer import ask, check_citations, format_context
from repoqa.chunker import chunk_file
from repoqa.config import Settings
from repoqa.engine import load_engine
from repoqa.evaluate import evaluate, matches
from repoqa.indexer import index_repo
from repoqa.llm import LLM, Ollama
from repoqa.retrieve import Hit, rrf
from repoqa.terms import called_names, code_terms, identifiers_in


# ------------------------------------------------------------------ terms
def test_code_terms_split_camel_and_snake():
    t = code_terms("getUserName parse_http_header")
    assert {"getusername", "get", "user", "name", "parse_http_header", "parse", "http", "header"} <= set(t)


def test_identifiers_in_question():
    ids = identifiers_in("Why does `Session.send` call resolve_redirects and HTTPAdapter?")
    assert "Session.send" in ids and "resolve_redirects" in ids and "HTTPAdapter" in ids
    assert "Why" not in ids


def test_called_names_excludes_definitions():
    src = "def foo(a):\n    return bar(a) + baz (1)\nif (x): pass\nfunc (s *S) Start() { run() }"
    assert called_names(src) == {"bar", "baz", "run"}


# ------------------------------------------------------------------ chunking
PY = textwrap.dedent('''\
    import os

    CONST = 1
    OTHER = 2

    @decorator
    def top(a):
        """Top-level function."""
        return helper(a)

    class Store:
        """A store."""

        def get(self, key):
            return self._db.fetch(key)
    ''')


def test_python_chunks_and_symbols():
    chunks = chunk_file("pkg/store.py", PY)
    syms = {c.symbol: c for c in chunks if c.symbol}
    assert "top" in syms and syms["top"].start == 6  # decorator line included
    assert "Store" in syms
    assert any(c.kind == "module" for c in chunks)  # imports/constants kept
    assert "helper" in syms["top"].calls


def test_big_class_split_into_methods():
    body = "".join(f"    def m{i}(self):\n" + "        x = 1\n" * 30 for i in range(6))
    chunks = chunk_file("big.py", "class Big:\n    attr = 1\n" + body, max_lines=50)
    names = [c.symbol for c in chunks]
    assert "Big" in names and "Big.m0" in names and "Big.m5" in names


def test_go_doc_comment_attached():
    src = "package x\n\n// Start runs the server.\n// It blocks.\nfunc Start() error { return nil }\n"
    fn = next(c for c in chunk_file("s.go", src) if c.symbol == "Start")
    assert fn.start == 3 and "Start runs" in fn.text


def test_typescript_arrow_and_class():
    src = "export const add = (a: number, b: number) => a + b;\nexport class Svc { run() { return 1 } }\n"
    syms = {c.symbol for c in chunk_file("a.ts", src)}
    assert {"add", "Svc"} <= syms


def test_markdown_sections():
    chunks = chunk_file("README.md", "# Title\nintro\n\n## Install\npip install x\n")
    assert [c.symbol for c in chunks] == ["Title", "Install"]


def test_oversized_function_windows():
    src = "def long():\n" + "    x = 1\n" * 300
    parts = [c for c in chunk_file("l.py", src, max_lines=100, window=60, overlap=10) if c.symbol]
    assert len(parts) > 1 and parts[0].symbol == "long (part 1)"


# ------------------------------------------------------------------ fusion / citations
def test_rrf_rewards_agreement():
    scores, via = rrf([("a", [1, 2, 3], 1.0), ("b", [2, 1, 4], 1.0)])
    assert max(scores, key=scores.get) in (1, 2) and scores[4] < scores[2]
    assert via[2] == ["a", "b"]


def _hit(path="a.py", s=10, e=20):
    return Hit(1, path, s, e, "python", "function", "f", "x = 1\n" * (e - s + 1))


def test_check_citations():
    cites = check_citations("See [a.py:12-14] and [b.py:3] and [a.py:50-60].", [_hit()])
    assert [c.valid for c in cites] == [True, False, False]


def test_format_context_respects_budget():
    hits = [_hit(s=1, e=100), _hit(s=200, e=300)]
    ctx, used = format_context(hits, budget_chars=1500)
    assert len(used) == 1 and "    1| x = 1" in ctx


# ------------------------------------------------------------------ end to end on a fixture repo
@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "auth.py").write_text(textwrap.dedent('''\
        def hash_password(raw, salt):
            """Hash a password with PBKDF2."""
            return pbkdf2(raw, salt, iterations=600_000)

        def verify_login(user, raw):
            return hash_password(raw, user.salt) == user.password_hash
        '''))
    (tmp_path / "app" / "billing.py").write_text(textwrap.dedent('''\
        class InvoiceService:
            def create_invoice(self, customer, items):
                total = sum(i.price for i in items)
                return self.repo.save(customer, total)
        '''))
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("function junk() {}")
    (tmp_path / "README.md").write_text("# Demo\nA demo app.\n")
    return tmp_path


def _settings() -> Settings:
    s = Settings()
    s.embedder, s.reranker = "hash", "none"
    return s


def test_index_is_incremental(repo: Path):
    st = index_repo(repo, _settings())
    assert st.added == 3 and st.chunks > 0          # node_modules skipped
    assert index_repo(repo, _settings()).unchanged == 3
    (repo / "app" / "billing.py").write_text("def refund(x):\n    return x\n")
    st = index_repo(repo, _settings())
    assert st.updated == 1 and st.unchanged == 2
    (repo / "README.md").unlink()
    assert index_repo(repo, _settings()).removed == 1


def test_git_ignored_files_skipped(repo: Path):
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("secret.py\n")
    (repo / "secret.py").write_text("def leak():\n    return 'key'\n")
    index_repo(repo, _settings())
    eng = load_engine(repo, _settings(), rerank=False)
    assert "secret.py" not in eng.store.all_paths()


def test_search_finds_definition_and_graph_neighbour(repo: Path):
    index_repo(repo, _settings())
    eng = load_engine(repo, _settings(), rerank=False)
    hits = eng.retriever.search("how is the login password verified?", k=3)
    assert hits[0].symbol in ("verify_login", "hash_password")
    syms = {h.symbol for h in hits}
    assert {"verify_login", "hash_password"} <= syms     # one direct, one via call graph or bm25
    exact = eng.retriever.search("what does `create_invoice` do", k=3)
    # small class kept whole: its methods resolve to the class chunk
    assert exact[0].symbol == "InvoiceService" and "symbol" in exact[0].via
    assert matches(exact[0], "app/billing.py::InvoiceService.create_invoice")


def test_eval_harness(repo: Path):
    index_repo(repo, _settings())
    eng = load_engine(repo, _settings(), rerank=False)
    golden = [{"q": "password hashing", "expected": ["app/auth.py::hash_password"]},
              {"q": "invoice totals", "expected": ["app/billing.py"]}]
    res = {r.name: r for r in evaluate(eng.retriever, golden, k=3)}
    assert res["bm25"].hit_at_k == 1.0 and "hybrid+rerank" not in res   # no reranker -> skipped
    assert matches(_hit("app/x.py"), "app/x.py") and not matches(_hit("app/x.py"), "app/x.py::g")


class FakeLLM(LLM):
    name = "fake"

    def __init__(self):
        self.seen = None

    def stream(self, system, messages):
        self.seen = messages
        yield "It hashes with PBKDF2 "
        yield "[app/auth.py:1-3] and checks in [app/auth.py:5-6]. Also [nope.py:1]."


def test_ask_streams_and_checks_citations(repo: Path):
    index_repo(repo, _settings())
    eng = load_engine(repo, _settings(), rerank=False)
    llm = FakeLLM()
    events = list(ask("how are passwords hashed?", eng.retriever, llm, k=4))
    kinds = [k for k, _ in events]
    assert kinds[0] == "sources" and kinds[-1] == "done" and kinds.count("token") == 2
    res = events[-1][1]
    assert [c.valid for c in res.citations] == [True, True, False]
    assert "   1| def hash_password" in llm.seen[-1]["content"]   # line-numbered context


def test_followup_uses_previous_question(repo: Path):
    from repoqa.answer import retrieval_query
    hist = [{"role": "user", "content": "how are passwords hashed?"}, {"role": "assistant", "content": "..."}]
    assert "passwords" in retrieval_query("who calls it?", hist)


# ------------------------------------------------------------------ backends against fake servers
class _FakeOllama(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200)
        self.end_headers()
        if self.path == "/api/chat":
            assert body["stream"] and body["messages"][0]["role"] == "system"
            for piece in ("Hello", " world"):
                self.wfile.write((json.dumps({"message": {"content": piece}, "done": False}) + "\n").encode())
            self.wfile.write(b'{"message":{"content":""},"done":true}\n')
        elif self.path == "/api/embed":
            self.wfile.write(json.dumps({"embeddings": [[1.0, 0.0]] * len(body["input"])}).encode())

    def log_message(self, *a):
        pass


def test_ollama_streaming_against_fake_server():
    srv = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        llm = Ollama("fake", f"http://127.0.0.1:{srv.server_port}")
        assert "".join(llm.stream("sys", [{"role": "user", "content": "hi"}])) == "Hello world"
    finally:
        srv.shutdown()


def test_server_endpoints(repo: Path, monkeypatch):
    from fastapi.testclient import TestClient
    import repoqa.engine as engine_mod
    from repoqa.server import create_app

    index_repo(repo, _settings())
    monkeypatch.setattr(engine_mod, "get_llm", lambda spec, s: FakeLLM())
    client = TestClient(create_app(repo, _settings()))
    assert client.get("/health").json()["files"] == 3
    hits = client.get("/search", params={"q": "invoice", "k": 2}).json()["hits"]
    assert hits and hits[0]["path"] == "app/billing.py"
    with client.stream("POST", "/ask", json={"question": "how are passwords hashed?"}) as r:
        raw = "".join(r.iter_text())
    events = [blk.split("\n")[0][7:] for blk in raw.strip().split("\n\n")]
    assert events[0] == "sources" and events[-1] == "done" and "token" in events
    assert client.post("/reindex").json()["unchanged"] == 3
    assert "<title>repoqa</title>" in client.get("/").text
