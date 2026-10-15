"""repoqa command line."""
from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from .config import Settings

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Ask questions about a codebase. Hybrid retrieval over AST chunks, answers with line citations.")
hook_app = typer.Typer(help="Keep the index fresh with a git hook.")
app.add_typer(hook_app, name="hook")
con = Console()
err = Console(stderr=True)

RepoOpt = typer.Option(".", "--repo", "-r", help="Repository root (must be indexed).")


def _settings(embedder=None, llm=None, reranker=None) -> Settings:
    s = Settings()
    if embedder:
        s.embedder = embedder
    if llm:
        s.llm = llm
    if reranker:
        s.reranker = reranker
    return s


def _engine(repo: str, settings: Settings, rerank: bool = True):
    from .engine import NotIndexed, load_engine
    try:
        return load_engine(repo, settings, rerank=rerank)
    except NotIndexed as e:
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def index(path: str = typer.Argument(".", help="Repository to index."),
          embedder: str = typer.Option(None, help="fastembed:<model> | ollama:<model> | hash"),
          full: bool = typer.Option(False, "--full", help="Drop the index and rebuild from scratch."),
          quiet: bool = typer.Option(False, "--quiet", "-q")):
    """Index (or incrementally update) a repository."""
    from .indexer import index_repo
    s = _settings(embedder=embedder)
    if quiet:
        st = index_repo(path, s, full=full)
    else:
        note = "  (first run downloads the model)" if s.embedder.startswith("fastembed") else ""
        con.print(f"[dim]embedder: {s.embedder}{note}[/dim]")
        with Progress(TextColumn("[bold]indexing"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                      TextColumn("[dim]{task.description}"), console=con, transient=True) as prog:
            task = prog.add_task("", total=None)
            st = index_repo(path, s, full=full,
                            progress=lambda n, t, f: prog.update(task, completed=n, total=t, description=f[-60:]))
    if not quiet:
        con.print(f"[green]done[/green] in {st.seconds:.1f}s: {st.scanned} files scanned, {st.added} added, "
                  f"{st.updated} updated, {st.removed} removed, {st.unchanged} unchanged, {st.chunks} new chunks"
                  + (" [yellow](full rebuild)[/yellow]" if st.rebuilt else ""))
        for e in st.errors[:10]:
            err.print(f"[yellow]skipped[/yellow] {e}")


@app.command()
def stats(repo: str = RepoOpt):
    """Show what's in the index."""
    from .indexer import open_store
    con.print_json(json.dumps(open_store(Path(repo).resolve()).stats()))


@app.command()
def search(query: str, repo: str = RepoOpt, k: int = typer.Option(8, "-k"),
           mode: str = typer.Option("hybrid", help="bm25 | dense | hybrid"),
           rerank: bool = typer.Option(True, help="Cross-encoder reranking."),
           graph: bool = typer.Option(True, help="Add call-graph neighbours."),
           as_json: bool = typer.Option(False, "--json")):
    """Retrieval only: show the chunks a question would pull in."""
    e = _engine(repo, _settings(), rerank=rerank)
    hits = e.retriever.search(query, k=k, mode=mode, rerank=rerank, graph=graph)
    if as_json:
        print(json.dumps({"hits": [h.to_dict(with_text=False) for h in hits], "timings": e.retriever.last_timings}))
        return
    t = Table(show_lines=False, header_style="bold")
    for col in ("#", "location", "symbol", "via", "score"):
        t.add_column(col)
    for i, h in enumerate(hits, 1):
        t.add_row(str(i), h.ref, h.symbol or f"[dim]{h.kind}[/dim]", ", ".join(h.via), f"{h.score:.3f}")
    con.print(t)
    con.print(f"[dim]{e.retriever.last_timings.get('total', 0) * 1000:.0f} ms[/dim]")


def _answer_once(e, question: str, k: int, history: list, show_sources: bool) -> str:
    from .answer import ask
    from .llm import LLMError
    text = ""
    try:
        for kind, payload in ask(question, e.retriever, e.llm, k=k, context_chars=e.settings.context_chars,
                                 history=history):
            if kind == "sources" and show_sources:
                con.print("[dim]sources: " + escape("  ".join(h.ref for h in payload)) + "[/dim]\n")
            elif kind == "token":
                text += payload
                sys.stdout.write(payload)
                sys.stdout.flush()
            elif kind == "done":
                print()
                bad = payload.invalid_citations
                t = payload.timings
                con.print(f"\n[dim]{len(payload.citations)} citations"
                          + (f", [red]{len(bad)} not in retrieved context: {escape(' '.join(c.raw for c in bad))}[/red]" if bad else "")
                          + f" · retrieve {t.get('retrieve', 0):.2f}s · first token {t.get('first_token', 0):.1f}s"
                          + f" · total {t.get('total', 0):.1f}s[/dim]")
    except LLMError as ex:
        err.print(f"\n[red]{escape(str(ex))}[/red]")
    return text


@app.command()
def ask(question: str = typer.Argument(None, help="Leave empty for an interactive session."),
        repo: str = RepoOpt, k: int = typer.Option(8, "-k"),
        llm: str = typer.Option(None, help="ollama:<model> | openai:<model> | anthropic:<model>"),
        rerank: bool = typer.Option(True), sources: bool = typer.Option(True)):
    """Ask a question; the answer streams with file:line citations."""
    e = _engine(repo, _settings(llm=llm), rerank=rerank)
    if question:
        _answer_once(e, question, k, [], sources)
        return
    con.print(f"[bold]repoqa[/bold] · {e.root} · {e.settings.llm} · empty line or Ctrl-D to quit")
    history: list[dict] = []
    while True:
        try:
            q = con.input("\n[bold cyan]?[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        a = _answer_once(e, q, k, history, sources)
        history = (history + [{"role": "user", "content": q}, {"role": "assistant", "content": a}])[-6:]


@app.command()
def serve(repo: str = RepoOpt, host: str = "127.0.0.1", port: int = 8765,
          llm: str = typer.Option(None, help="ollama:<model> | openai:<model> | anthropic:<model>")):
    """Run the HTTP API and web UI."""
    import uvicorn
    from .engine import NotIndexed
    from .server import create_app
    try:
        application = create_app(repo, _settings(llm=llm))
    except NotIndexed as e:
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None
    con.print(f"[green]serving[/green] http://{host}:{port}")
    uvicorn.run(application, host=host, port=port, log_level="warning")


@app.command("eval")
def eval_cmd(golden: str = typer.Argument(..., help="JSONL golden set"), repo: str = RepoOpt,
             k: int = typer.Option(8, "-k"),
             configs: str = typer.Option(None, help="Comma-separated subset, e.g. bm25,hybrid"),
             answers: bool = typer.Option(False, "--answers", help="Also run the LLM and score citations."),
             out: str = typer.Option(None, help="Write results JSON here.")):
    """Score retrieval on a golden set, comparing configurations side by side."""
    from .evaluate import evaluate, evaluate_answers, load_golden, to_json
    e = _engine(repo, _settings())
    items = load_golden(golden)
    results = evaluate(e.retriever, items, k=k, configs=configs.split(",") if configs else None)
    t = Table(title=f"retrieval @ k={k} · {len(items)} questions", header_style="bold")
    for col in ("config", "hit@k", "recall@k", "MRR", "p50 ms", "p95 ms"):
        t.add_column(col, justify="right" if col != "config" else "left")
    best = max(r.mrr for r in results) if results else 0
    for r in results:
        style = "bold green" if r.mrr == best else None
        t.add_row(r.name, f"{r.hit_at_k:.2f}", f"{r.recall_at_k:.2f}", f"{r.mrr:.3f}", f"{r.p50_ms:.0f}",
                  f"{r.p95_ms:.0f}", style=style)
    con.print(t)
    if results and results[-1].misses:
        con.print(f"[dim]missed by {results[-1].name}:[/dim]")
        for m in results[-1].misses:
            con.print(f"  [dim]- {m}[/dim]")
    payload = {"k": k, "n": len(items), "retrieval": to_json(results)}
    if answers:
        con.print(f"[dim]running {len(items)} questions through {e.settings.llm}…[/dim]")
        ans = evaluate_answers(e.retriever, e.llm, items, k=k, context_chars=e.settings.context_chars)
        payload["answers"] = ans
        con.print(f"answered with citations {ans['answered_with_citations']:.0%} · citation validity "
                  f"{ans['citation_validity']:.0%} · cites an expected file {ans['cites_expected_file']:.0%} · "
                  f"{ans['mean_seconds']:.1f}s/question")
    if out:
        Path(out).write_text(json.dumps(payload, indent=2))
        con.print(f"[dim]wrote {out}[/dim]")


HOOK = """#!/bin/sh
# added by repoqa: refresh the code index in the background after each commit / merge / checkout
command -v repoqa >/dev/null 2>&1 && (repoqa index "$(git rev-parse --show-toplevel)" --quiet >/dev/null 2>&1 &)
"""


@hook_app.command("install")
def hook_install(repo: str = RepoOpt):
    """Install post-commit, post-merge and post-checkout hooks that re-index incrementally."""
    hooks = Path(repo).resolve() / ".git" / "hooks"
    if not hooks.is_dir():
        err.print(f"[red]{hooks} not found; is this a git repo?[/red]")
        raise typer.Exit(1) from None
    for name in ("post-commit", "post-merge", "post-checkout"):
        p = hooks / name
        if p.exists() and "repoqa" not in p.read_text(errors="replace"):
            p.write_text(p.read_text(errors="replace").rstrip() + "\n" + HOOK.split("\n", 1)[1])
        else:
            p.write_text(HOOK)
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    con.print(f"[green]installed[/green] hooks in {hooks}")
    gi = Path(repo).resolve() / ".gitignore"
    if not gi.exists() or ".repoqa" not in gi.read_text(errors="replace"):
        con.print("[dim]tip: add .repoqa/ to your .gitignore[/dim]")


def main():
    app()


if __name__ == "__main__":
    main()
