"""Reference web approve/deny host for the spend authorizer (handoff budget feature).

A minimal, self-contained HTTP UI + API that lets a user approve/deny each paid or
rented charge — and set a budget when prompted — from a browser or phone. The router's
:class:`~agentconnect.common.approval.WebApprovalAuthorizer` blocks on a shared
:class:`~agentconnect.common.approval.ApprovalQueue`; this app resolves the items.

Money endpoint: binds loopback by default and supports an optional bearer token on the
API. Expose remotely only behind TLS + a token.

Requires the router's ``web`` extra:  ``pip install "agentconnect-router[web]"``.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from pydantic import BaseModel

from ..common.approval import ApprovalQueue

_log = logging.getLogger(__name__)


class BudgetBody(BaseModel):
    amount_usd: float
    period: str = "monthly"

_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgentConnect · Spend approvals</title>
<style>
 body{font:15px/1.5 system-ui,sans-serif;max-width:640px;margin:2rem auto;padding:0 1rem;color:#111}
 h1{font-size:1.2rem} .item{border:1px solid #ddd;border-radius:10px;padding:1rem;margin:.75rem 0}
 .charge{border-left:4px solid #b45309} .budget{border-left:4px solid #2563eb}
 button{font:inherit;padding:.4rem .8rem;border-radius:8px;border:1px solid #ccc;cursor:pointer;margin-right:.4rem}
 .ok{background:#16a34a;color:#fff;border-color:#16a34a} .no{background:#dc2626;color:#fff;border-color:#dc2626}
 input,select{font:inherit;padding:.35rem;border:1px solid #ccc;border-radius:8px}
 .muted{color:#666} .tok{margin-bottom:1rem}
</style></head><body>
<h1>Spend approvals</h1>
<div class="tok muted">Token (if required): <input id="tok" size="24" oninput="localStorage.tok=this.value"></div>
<div id="list"><p class="muted">Waiting for approval requests…</p></div>
<script>
const tokEl=document.getElementById('tok'); tokEl.value=localStorage.tok||'';
const hdr=()=>tokEl.value?{'Authorization':'Bearer '+tokEl.value}:{};
async function post(u,b){await fetch(u,{method:'POST',headers:{'Content-Type':'application/json',...hdr()},body:b?JSON.stringify(b):null});load();}
function esc(s){return (s+'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
async function load(){
 let r; try{r=await fetch('/api/pending',{headers:hdr()});}catch(e){return;}
 if(!r.ok){document.getElementById('list').innerHTML='<p class="no">Unauthorized — enter a token above.</p>';return;}
 const items=await r.json(); const el=document.getElementById('list');
 if(!items.length){el.innerHTML='<p class="muted">Nothing pending. 🎉</p>';return;}
 el.innerHTML=items.map(it=>{
  if(it.kind==='charge') return `<div class="item charge"><div>${esc(it.text)}</div>
    <div style="margin-top:.6rem"><button class="ok" onclick="post('/api/charges/${it.id}/approve')">Approve</button>
    <button class="no" onclick="post('/api/charges/${it.id}/deny')">Deny</button></div></div>`;
  return `<div class="item budget"><div>${esc(it.text)}</div>
    <div style="margin-top:.6rem">$<input id="a_${it.id}" size="6" placeholder="25">
    <select id="p_${it.id}"><option>monthly</option><option>weekly</option><option>daily</option></select>
    <button class="ok" onclick="post('/api/budget/${it.id}',{amount_usd:parseFloat(document.getElementById('a_'+'${it.id}').value),period:document.getElementById('p_'+'${it.id}').value})">Save</button>
    <button class="no" onclick="post('/api/budget/${it.id}/decline')">Decline</button></div></div>`;
 }).join('');
}
load(); setInterval(load,2000);
</script></body></html>"""


def create_approval_app(queue: ApprovalQueue, token: Optional[str] = None):
    """Build the FastAPI app bound to an ApprovalQueue. `token` (optional) is a bearer
    required on all /api/* routes."""
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="AgentConnect Spend Approvals", version="0.1.0")

    def auth(authorization: Optional[str] = Header(default=None)) -> None:
        if token is None:
            return
        if authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _PAGE

    @app.get("/api/pending")
    def pending(_: None = Depends(auth)):
        return queue.pending()

    @app.post("/api/charges/{approval_id}/approve")
    def approve(approval_id: str, _: None = Depends(auth)):
        return _resolve(approval_id, True)

    @app.post("/api/charges/{approval_id}/deny")
    def deny(approval_id: str, _: None = Depends(auth)):
        return _resolve(approval_id, False)

    @app.post("/api/budget/{approval_id}")
    def set_budget(approval_id: str, body: BudgetBody, _: None = Depends(auth)):
        return _resolve(approval_id, {"amount_usd": body.amount_usd, "period": body.period})

    @app.post("/api/budget/{approval_id}/decline")
    def decline_budget(approval_id: str, _: None = Depends(auth)):
        return _resolve(approval_id, None)

    def _resolve(approval_id: str, result):
        if not queue.resolve(approval_id, result):
            return JSONResponse({"ok": False, "reason": "unknown_or_already_resolved"}, status_code=404)
        return {"ok": True}

    return app


def start_web_approval(
    queue: ApprovalQueue, host: str = "127.0.0.1", port: int = 8770, token: Optional[str] = None
) -> threading.Thread:
    """Launch the approval server in a daemon thread. Returns the thread."""
    import uvicorn

    app = create_approval_app(queue, token=token)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="agentconnect-approvals")
    thread.start()
    _log.warning("Spend-approval UI serving on http://%s:%d/", host, port)
    return thread
