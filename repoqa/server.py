"""HTTP API + a minimal web UI.

GET  /             chat UI
GET  /health       index stats
GET  /search?q=    retrieval only (JSON)
POST /ask          {"question": str, "k": int, "history": [{"role","content"}]} -> Server-Sent Events:
                   event: sources | token | done | error
POST /reindex      incremental re-index of the repo
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .answer import ask
from .config import Settings
from .engine import load_engine
from .indexer import index_repo
from .llm import LLMError


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    k: int = Field(8, ge=1, le=30)
    history: list[dict[str, str]] = []


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def create_app(root: str | Path, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    state = {"engine": load_engine(root, settings)}
    reindex_lock = threading.Lock()
    app = FastAPI(title="repoqa", version="0.1.0")

    @app.get("/health")
    def health():
        e = state["engine"]
        return {"root": str(e.root), "llm": settings.llm, "reranker": settings.reranker, **e.store.stats()}

    @app.get("/search")
    def search(q: str = Query(min_length=1), k: int = Query(8, ge=1, le=50), mode: str = "hybrid",
               rerank: bool = True, graph: bool = True):
        e = state["engine"]
        try:
            hits = e.retriever.search(q, k=k, mode=mode, rerank=rerank, graph=graph)
        except ValueError as err:
            raise HTTPException(400, str(err)) from None
        return {"hits": [h.to_dict() for h in hits], "timings": e.retriever.last_timings}

    @app.post("/ask")
    def ask_endpoint(body: AskBody):
        e = state["engine"]

        def gen():
            try:
                for kind, payload in ask(body.question, e.retriever, e.llm, k=body.k,
                                         context_chars=settings.context_chars, history=body.history):
                    if kind == "sources":
                        yield _sse("sources", [h.to_dict(with_text=False) for h in payload])
                    elif kind == "token":
                        yield _sse("token", payload)
                    else:
                        yield _sse("done", {"timings": payload.timings,
                                            "citations": [c.__dict__ for c in payload.citations]})
            except (LLMError, ValueError) as err:
                yield _sse("error", str(err))

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/reindex")
    def reindex():
        if not reindex_lock.acquire(blocking=False):
            raise HTTPException(409, "re-index already running")
        try:
            stats = index_repo(state["engine"].root, settings, embedder=state["engine"].retriever.embedder)
            state["engine"].store.bump_generation()
            return stats.__dict__
        finally:
            reindex_lock.release()

    @app.get("/", response_class=HTMLResponse)
    def ui():
        return UI_HTML

    return app


UI_HTML = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>repoqa</title><style>
:root{--bg:#0f1115;--fg:#e6e6e6;--mut:#8b93a1;--card:#171a21;--line:#262b36;--acc:#7aa2f7;--bad:#f7768e}
@media (prefers-color-scheme: light){:root{--bg:#fafafa;--fg:#1d1f24;--mut:#626a77;--card:#fff;--line:#e3e5ea;--acc:#2f5bd3;--bad:#c4314b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px 120px}h1{font-size:18px;margin:0 0 4px}.sub{color:var(--mut);font-size:13px;margin-bottom:20px}
.turn{margin:18px 0}.q{font-weight:600}.a{white-space:pre-wrap;margin-top:8px}
.a code,.src code{font-family:ui-monospace,monospace;font-size:13px}.cite{color:var(--acc)}.cite.bad{color:var(--bad);text-decoration:line-through}
details{margin-top:8px;font-size:13px;color:var(--mut)}.src{padding:2px 0}
form{position:fixed;bottom:0;left:0;right:0;background:var(--bg);border-top:1px solid var(--line);padding:12px 16px}
form div{max-width:860px;margin:0 auto;display:flex;gap:8px}input{flex:1;padding:10px 12px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);font:inherit}
button{padding:10px 16px;border-radius:8px;border:0;background:var(--acc);color:#fff;font:inherit;cursor:pointer}button:disabled{opacity:.5}
</style></head><body><main><h1>repoqa</h1><div class="sub" id="meta">loading…</div><div id="log"></div></main>
<form id="f"><div><input id="q" placeholder="Ask about this codebase…" autocomplete="off" autofocus><button id="b">Ask</button></div></form>
<script>
const log=document.getElementById('log'),f=document.getElementById('f'),q=document.getElementById('q'),b=document.getElementById('b');
let history=[];
fetch('/health').then(r=>r.json()).then(h=>{document.getElementById('meta').textContent=`${h.root} · ${h.files} files · ${h.chunks} chunks · ${h.llm}`});
const esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const render=(t,bad)=>esc(t).replace(/\\[([\\w@+\\-./]+?):(\\d+)(?:\\s*[-–]\\s*(\\d+))?\\]/g,m=>`<span class="cite${bad.has(m)?' bad':''}">${m}</span>`);
f.onsubmit=async e=>{e.preventDefault();const question=q.value.trim();if(!question)return;q.value='';b.disabled=true;
 const turn=document.createElement('div');turn.className='turn';turn.innerHTML=`<div class="q">${esc(question)}</div><div class="a">…</div><details><summary>sources</summary></details>`;
 log.appendChild(turn);const a=turn.querySelector('.a'),d=turn.querySelector('details');let text='';
 const res=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question,history})});
 const rd=res.body.getReader(),dec=new TextDecoder();let buf='';
 while(true){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});let i;
  while((i=buf.indexOf('\\n\\n'))>=0){const raw=buf.slice(0,i);buf=buf.slice(i+2);
   const ev=(raw.match(/^event: (.*)$/m)||[])[1],data=JSON.parse((raw.match(/^data: (.*)$/m)||[])[1]||'null');
   if(ev==='sources')d.innerHTML+=data.map(s=>`<div class="src"><code>${esc(s.ref)}</code> ${esc(s.symbol||s.kind)} <span>· ${s.via.join(', ')}</span></div>`).join('');
   else if(ev==='token'){text+=data;a.innerHTML=render(text,new Set());}
   else if(ev==='done'){const bad=new Set(data.citations.filter(c=>!c.valid).map(c=>c.raw));a.innerHTML=render(text,bad);
     d.querySelector('summary').textContent=`sources · ${data.timings.total.toFixed(1)}s`;}
   else if(ev==='error'){a.innerHTML=`<span class="cite bad">${esc(data)}</span>`;}}}
 history.push({role:'user',content:question},{role:'assistant',content:text});history=history.slice(-6);b.disabled=false;q.focus();};
</script></body></html>"""
