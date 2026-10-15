# repoqa

Ask questions about a codebase and get answers that cite exact files and lines.

```
# illustrative session; wording and line numbers depend on the model and repo version
$ repoqa ask "When is the Authorization header stripped on redirect?"
sources: src/requests/sessions.py:127-158  src/requests/sessions.py:282-306  …

The header is dropped in `rebuild_auth` whenever `should_strip_auth` says the redirect
changes host, or downgrades from https to http [src/requests/sessions.py:282-306].
`should_strip_auth` allows http→https upgrades on default ports [src/requests/sessions.py:127-158] …

3 citations · retrieve 0.04s · first token 1.9s · total 6.2s
```

Runs fully local on a CPU-only laptop (ONNX embeddings, Ollama), or against any OpenAI-compatible API or Anthropic.

## How it works

```
                 ┌─────────────── index (incremental) ───────────────┐
 git ls-files →  tree-sitter AST chunks → SQLite: FTS5 (BM25, porter) │
 (respects       one chunk per function/      + symbol table          │
  .gitignore)    class, doc comments kept     + call graph            │
                 oversized → split            + embeddings            │
                 └──────────────────────────────────────────────────────┘
 question ─┬─ BM25 (path / symbol / body columns, weighted) ──┐
           ├─ dense (code embedding model) ───────────────────┼─ RRF fusion → priors → cross-encoder
           └─ exact symbol + filename lookup (weight 2×) ─────┘     rerank → top-k → call-graph expansion
                                                                    (callees + callers of the top hits)
 top-k chunks, line-numbered ─→ LLM ─→ streamed answer ─→ every [path:a-b] checked against what was retrieved
```

What makes it more than a PDF chatbot pointed at a repo:

| Problem | What repoqa does |
|---|---|
| Fixed-size chunks cut functions in half | **AST chunking** with tree-sitter for 17 languages. Each function, class, impl or struct is one chunk. Big classes split into a header plus one chunk per method, and doc comments stay attached. Markdown splits on headings. |
| Embeddings miss exact identifiers | **Hybrid retrieval**: BM25 with camelCase/snake_case splitting and stemming, plus dense vectors, plus exact symbol lookup, combined with reciprocal rank fusion. |
| The answer needs the function the hit *calls* | **Call-graph expansion** adds the callees and callers of the top hits. Ambiguous names and widely used helpers are deprioritised. |
| Tests and changelogs drown out source | **Path priors** downweight tests, changelogs and prose, unless the question asks about them or names the symbol. |
| Hallucinated references | **Citation checking**: every `[path:a-b]` in the answer is verified against the retrieved line ranges, and invalid ones are flagged in the CLI, the API and the UI. |
| Re-indexing a big repo on every change | **Incremental indexing** by content hash, plus optional git hooks that refresh after commit, merge or checkout. |
| "Is it actually good?" | **Eval harness** reporting hit@k, recall@k, MRR and latency, with side-by-side ablations (BM25 vs dense vs hybrid vs +rerank vs +graph) and an optional end-to-end citation score. |

## Install (EndeavourOS / Arch)

```bash
sudo pacman -S --needed python python-pip git
cd ~/Documents/github && git clone <your repo url> repoqa && cd repoqa
pip install --break-system-packages -e '.[dev]'       # global install; puts `repoqa` on PATH
```

If `repoqa` isn't found afterwards, add `~/.local/bin` to your `PATH`.

The embedding model (~0.6 GB) and reranker (~0.15 GB) download from Hugging Face on first use and are cached.

### Pick an LLM

**Local (default): Ollama**
```bash
sudo pacman -S ollama
sudo systemctl enable --now ollama
ollama pull qwen2.5-coder:7b          # ~4.7 GB; usable on CPU, a few tokens/sec
# lighter: ollama pull qwen2.5-coder:3b && export REPOQA_LLM=ollama:qwen2.5-coder:3b
```

**Any OpenAI-compatible API** (OpenAI, Groq, OpenRouter, vLLM, LM Studio, llama.cpp server)
```bash
export REPOQA_LLM=openai:<model-name>
export OPENAI_BASE_URL=https://api.groq.com/openai/v1   # or your provider's base URL
export OPENAI_API_KEY=...
```

**Anthropic**
```bash
export REPOQA_LLM=anthropic:claude-sonnet-5-5
export ANTHROPIC_API_KEY=...
```

## Use

```bash
repoqa index ~/Documents/github/some-repo          # first run embeds everything; later runs only changed files
cd ~/Documents/github/some-repo

repoqa ask "how does auth token refresh work?"      # one question, streamed
repoqa ask                                          # interactive; follow-ups like "who calls it?" work
repoqa search "retry backoff" --mode bm25           # retrieval only: see which chunks come back and why (via)
repoqa serve                                        # web UI + API at http://127.0.0.1:8765
repoqa hook install                                 # keep the index fresh on commit/merge/checkout
repoqa stats
```

Add `.repoqa/` to the indexed repo's `.gitignore`. The index is one SQLite file in that folder.

### API

| | |
|---|---|
| `GET /health` | index stats |
| `GET /search?q=…&k=8&mode=hybrid&rerank=true&graph=true` | ranked chunks as JSON |
| `POST /ask` `{"question", "k", "history"}` | Server-Sent Events: `sources`, `token`…, `done` (timings + citation validity), `error` |
| `POST /reindex` | incremental re-index |

## Evaluate

The golden set is JSONL. Each `expected` entry is a file path, or `path::Symbol` for a specific definition:

```json
{"q": "When is the Authorization header stripped during a redirect?", "expected": ["src/requests/sessions.py::SessionRedirectMixin.should_strip_auth"]}
```

```bash
git clone https://github.com/psf/requests /tmp/requests && repoqa index /tmp/requests
repoqa eval examples/requests_golden.jsonl -r /tmp/requests              # retrieval ablation table
repoqa eval examples/requests_golden.jsonl -r /tmp/requests --answers --out results.json   # + LLM citation scores
```

`examples/requests_golden.jsonl` has 24 questions, most phrased the way a newcomer would ask, without naming the function.

Development baseline on that set (k=8). This used the offline `hash` embedder, which has no semantics, so the "dense" row is a floor, not a result. Run the same command with the real embedder to get your actual numbers:

| config | hit@k | recall@k | MRR |
|---|---|---|---|
| bm25 | 0.96 | 0.94 | 0.81 |
| dense (hash placeholder) | 0.75 | 0.69 | 0.49 |
| hybrid | 0.96 | 0.90 | 0.66 |

Two changes found through this harness during development: Porter stemming, and putting the enclosing class name in a low-weight column. Together they raised BM25 MRR from 0.66 to 0.81. Run the eval before and after any chunking or prompt change.

## Configuration

Set these as environment variables or via CLI flags.

| variable | default |
|---|---|
| `REPOQA_EMBEDDER` | `fastembed:jinaai/jina-embeddings-v2-base-code`; also `fastembed:BAAI/bge-small-en-v1.5` (fast), `ollama:nomic-embed-text`, `hash` (tests) |
| `REPOQA_RERANKER` | `jinaai/jina-reranker-v1-turbo-en`, or `none` |
| `REPOQA_LLM` | `ollama:qwen2.5-coder:7b` |
| `OLLAMA_HOST` | `http://localhost:11434` |

Changing the embedder triggers a full rebuild on the next `index`, because vectors from different models aren't comparable. Queries always use the model the index was built with.

## Tests

```bash
pytest -q        # 20 tests, offline: hash embedder, fake LLM, fake Ollama server, FastAPI TestClient
```

## Layout

```
repoqa/
  chunker.py    tree-sitter chunking, symbols, call extraction
  terms.py      code-aware tokenization (camel/snake split, stopwords, identifier detection)
  store.py      SQLite: chunks, FTS5, defs, calls, vectors; schema versioning
  indexer.py    file discovery (git ls-files / .gitignore), hashing, incremental updates
  embed.py      fastembed / Ollama / hash embedders, cross-encoder reranker
  retrieve.py   BM25 + dense + symbol → RRF → priors → rerank → graph expansion
  llm.py        streaming Ollama, OpenAI-compatible, Anthropic over plain httpx
  answer.py     context building (line-numbered), prompting, citation checking, follow-ups
  evaluate.py   golden-set metrics and ablations
  server.py     FastAPI + SSE + minimal web UI
  cli.py        typer CLI
```

## Limitations and next steps

- The call graph matches names, not types. `obj.save()` links to every `save` defined three or fewer times. LSP or SCIP data would make it exact.
- Vectors are brute-force numpy in memory, which is fine up to roughly 200k chunks. Beyond that, swap the `matrix()` path for an approximate-nearest-neighbour index such as sqlite-vec or hnswlib.
- Questions about the repo as a whole ("give me an architecture overview") need a summary layer, e.g. per-directory summaries generated at index time.
- No auth on the server. Bind to localhost, as it does by default.
