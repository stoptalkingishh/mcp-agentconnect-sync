"""Read-only broker-side operator view over the federated work-queue.

A minimal, self-contained HTTP UI + API for a human to SEE queue state — open
tickets, tiers, capability requirements, and the ``in_review`` backlog awaiting
a local_only reviewer's spot-check — and to act on that backlog via approve/
reject. This host never invents its own authorization: every approve/reject
call is delegated straight to :meth:`WorkQueue.approve` / :meth:`WorkQueue.reject`,
which independently re-check that ``reviewer_tier`` is ``local_only`` (fail-closed
even if this host's own token/identity wiring were somehow bypassed).

PAYLOAD-FREE by construction: every route here returns only what
``WorkQueue.list_tickets`` / ``.pending_review`` / ``.stats`` project — ids,
status, tiers, capability requirements, provenance, and refs. Task/result
*content* is read back through the operator's own local_only-authorized
``read_artifact_chunk`` path, never through this surface.

Money/queue endpoint parity with :mod:`agentconnect.router.approval_web`: binds
loopback by default, optional bearer token on ``/api/*``, never mounted by
default (a separate opt-in module). Requires the router's ``web`` extra:
``pip install "agentconnect-router[web]"``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from pydantic import BaseModel

from ..common.workqueue import WorkQueue

_log = logging.getLogger(__name__)


class RejectBody(BaseModel):
    reason: str = ""


_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgentConnect · Queue operator</title>
<style>
 body{font:15px/1.5 system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#111}
 h1{font-size:1.2rem} h2{font-size:1rem;margin-top:2rem}
 table{border-collapse:collapse;width:100%;font-size:.9rem} td,th{border:1px solid #ddd;padding:.35rem .5rem;text-align:left}
 th{background:#f3f4f6} .muted{color:#666} .tok{margin-bottom:1rem}
 button{font:inherit;padding:.3rem .7rem;border-radius:8px;border:1px solid #ccc;cursor:pointer;margin-right:.3rem}
 .ok{background:#16a34a;color:#fff;border-color:#16a34a} .no{background:#dc2626;color:#fff;border-color:#dc2626}
 input{font:inherit;padding:.3rem;border:1px solid #ccc;border-radius:8px}
 code{background:#f3f4f6;padding:.05rem .3rem;border-radius:4px}
</style></head><body>
<h1>Queue operator</h1>
<div class="tok muted">Token (if required): <input id="tok" size="24" oninput="localStorage.tok=this.value"></div>
<div id="stats" class="muted">Loading stats…</div>
<h2>In-review backlog</h2>
<div id="pending"><p class="muted">Loading…</p></div>
<h2>All tickets</h2>
<div id="list"><p class="muted">Loading…</p></div>
<script>
const tokEl=document.getElementById('tok'); tokEl.value=localStorage.tok||'';
const hdr=()=>tokEl.value?{'Authorization':'Bearer '+tokEl.value}:{};
function esc(s){return (s+'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
async function post(u,b){await fetch(u,{method:'POST',headers:{'Content-Type':'application/json',...hdr()},body:b?JSON.stringify(b):null});load();}
function row(t){return `<tr><td>${esc(t.ticket_id)}</td><td>${esc(t.status)}</td>`+
  `<td>${esc(t.privacy_class)}</td><td>${esc((t.allowed_tiers||[]).join(', '))}</td>`+
  `<td>${esc((t.required_capabilities||[]).join(', '))}</td><td>${esc(t.result_status||'')}</td></tr>`;}
async function load(){
 let r; try{r=await fetch('/api/stats',{headers:hdr()});}catch(e){return;}
 if(!r.ok){document.getElementById('stats').innerHTML='<p class="no">Unauthorized — enter a token above.</p>';
   document.getElementById('pending').innerHTML='';document.getElementById('list').innerHTML='';return;}
 const stats=await r.json();
 document.getElementById('stats').innerHTML='<code>'+esc(JSON.stringify(stats))+'</code>';
 const pend=await (await fetch('/api/pending',{headers:hdr()})).json();
 document.getElementById('pending').innerHTML = pend.length ?
  '<table><tr><th>ticket</th><th>status</th><th>privacy</th><th>tiers</th><th>caps</th><th>result</th><th></th></tr>'+
   pend.map(t=>row(t).replace('</tr>',`<td><button class="ok" onclick="post('/api/tickets/${t.ticket_id}/approve')">Approve</button>`+
    `<button class="no" onclick="post('/api/tickets/${t.ticket_id}/reject')">Reject</button></td></tr>`)).join('')+'</table>'
  : '<p class="muted">Nothing pending review.</p>';
 const all=await (await fetch('/api/list',{headers:hdr()})).json();
 document.getElementById('list').innerHTML = all.length ?
  '<table><tr><th>ticket</th><th>status</th><th>privacy</th><th>tiers</th><th>caps</th><th>result</th></tr>'+
   all.map(row).join('')+'</table>' : '<p class="muted">No tickets.</p>';
}
load(); setInterval(load,3000);
</script></body></html>"""


def create_queue_operator_app(
    wq: WorkQueue, reviewer_id: str, reviewer_tier: str, token: Optional[str] = None
):
    """Build the FastAPI operator app bound to a ``WorkQueue``.

    ``reviewer_id``/``reviewer_tier`` are the trusted LOCAL operator identity
    used for approve/reject — ``wq.approve``/``wq.reject`` independently
    re-check ``reviewer_tier == local_only`` and refuse otherwise, so this host
    grants no authority of its own. ``token`` (optional) is a bearer required on
    all ``/api/*`` routes; the dashboard page itself is public (loopback-only
    deployment), matching :mod:`approval_web`.
    """
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="AgentConnect Queue Operator", version="0.1.0")

    def auth(authorization: Optional[str] = Header(default=None)) -> None:
        if token is None:
            return
        if authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _PAGE

    @app.get("/api/list")
    def api_list(
        status: Optional[str] = None,
        privacy_class: Optional[str] = None,
        limit: int = 100,
        _: None = Depends(auth),
    ) -> list[dict[str, Any]]:
        return wq.list_tickets(status=status, privacy_class=privacy_class, limit=limit)

    @app.get("/api/pending")
    def api_pending(limit: int = 100, _: None = Depends(auth)) -> list[dict[str, Any]]:
        return wq.pending_review(limit=limit)

    @app.get("/api/stats")
    def api_stats(_: None = Depends(auth)) -> dict[str, Any]:
        return wq.stats()

    @app.post("/api/tickets/{ticket_id}/approve")
    def api_approve(ticket_id: str, _: None = Depends(auth)) -> dict[str, Any]:
        return wq.approve(reviewer_id, reviewer_tier, ticket_id)

    @app.post("/api/tickets/{ticket_id}/reject")
    def api_reject(ticket_id: str, body: RejectBody = RejectBody(), _: None = Depends(auth)) -> dict[str, Any]:
        return wq.reject(reviewer_id, reviewer_tier, ticket_id, reason=body.reason)

    return app


def start_queue_operator(
    wq: WorkQueue,
    reviewer_id: str,
    reviewer_tier: str,
    host: str = "127.0.0.1",
    port: int = 8771,
    token: Optional[str] = None,
) -> threading.Thread:
    """Launch the queue operator server in a daemon thread. Returns the thread.

    Never started automatically by any router bootstrap — an operator opts in
    explicitly (mirrors ``start_web_approval``)."""
    import uvicorn

    app = create_queue_operator_app(wq, reviewer_id, reviewer_tier, token=token)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="agentconnect-queue-operator")
    thread.start()
    _log.warning("Queue operator UI serving on http://%s:%d/", host, port)
    return thread
