"""Federated work-queue MCP surface (S2): each queue_* tool round-trips JSON
over the SAME atomic/authorization core as test_workqueue.py, but through the
FastMCP tool boundary — worker/reviewer tiers are resolved ONLY from the
server-side identity->tier map, never from anything the caller passes in.

Offline: an in-memory RouterService (no provider network) + an injected
worker_tiers dict standing in for config/workers.yaml (exactly the injected-seam
style used for tier_resolver in the runtime-transport tests).
"""

import json

from agentconnect.common.config import load_all
from agentconnect.common.memory import SharedMemory
from agentconnect.common.schemas import TaskSubmission
from agentconnect.router.mcp_server import build_mcp_server
from agentconnect.router.service import RouterService

LOCAL = "local_only"
EXTERNAL = "external"

WORKER_TIERS = {
    "trusted-worker": LOCAL,
    "friend-box": EXTERNAL,
    "manager": LOCAL,
}


def _server(worker_tiers=None):
    load_all.cache_clear()
    svc = RouterService.create(memory=SharedMemory())
    mcp = build_mcp_server(service=svc, worker_tiers=worker_tiers if worker_tiers is not None else dict(WORKER_TIERS))
    return mcp, svc


def _call(mcp, name, **kwargs):
    tool = mcp._tool_manager.get_tool(name)
    return json.loads(tool.fn(**kwargs))


# --------------------------------------------------------------------- queue_add
def test_queue_add_round_trips_and_is_idempotent():
    mcp, _ = _server()
    a = _call(mcp, "queue_add", task="do the thing", agent_type="tester", dedup_key="job-1")
    assert a["status"] == "open"
    assert "ticket_id" in a
    b = _call(mcp, "queue_add", task="different text", agent_type="tester", dedup_key="job-1")
    assert b["ticket_id"] == a["ticket_id"]


def test_queue_add_secret_sensitive_is_parked_and_unclaimable():
    mcp, _ = _server()
    t = _call(
        mcp, "queue_add",
        task="here is my api key: sk-live-abcdef1234567890",
        privacy_class="secret_sensitive",
    )
    assert t["status"] == "parked"
    assert t["park_reason"] == "secret_sensitive_not_pullable"
    for worker in ("trusted-worker", "friend-box"):
        assert _call(mcp, "queue_claim", worker_id=worker, ticket_id=t["ticket_id"]) == {
            "error": "not_authorized"
        }


# ------------------------------------------------------------------- authz
def test_queue_next_denies_external_on_repo_sensitive_but_admits_public():
    mcp, _ = _server()
    sensitive = _call(mcp, "queue_add", task="internal repo secrets", privacy_class="repo_sensitive")
    public = _call(mcp, "queue_add", task="hello world", privacy_class="public")

    # External worker sees/claims nothing for the repo_sensitive ticket; it DOES
    # pick up the public one via queue_next (it's authorized for both classes).
    got = _call(mcp, "queue_next", worker_id="friend-box", max=5)
    ids = {t["ticket_id"] for t in got["tickets"]}
    assert sensitive["ticket_id"] not in ids
    assert public["ticket_id"] in ids
    assert _call(mcp, "queue_claim", worker_id="friend-box", ticket_id=sensitive["ticket_id"]) == {
        "error": "not_authorized"
    }
    # The public ticket is already claimed by the queue_next call above; a
    # second targeted claim on it correctly loses the race (not_claimable).
    assert _call(mcp, "queue_claim", worker_id="friend-box", ticket_id=public["ticket_id"]) == {
        "error": "not_claimable"
    }


def test_unknown_worker_identity_is_refused_fail_closed():
    mcp, _ = _server()
    t = _call(mcp, "queue_add", task="hi", privacy_class="public")
    assert _call(mcp, "queue_next", worker_id="nobody") == {"error": "unknown_worker_identity"}
    assert _call(mcp, "queue_claim", worker_id="nobody", ticket_id=t["ticket_id"]) == {
        "error": "unknown_worker_identity"
    }
    assert _call(mcp, "queue_approve", reviewer_id="nobody", ticket_id=t["ticket_id"]) == {
        "error": "unknown_worker_identity"
    }


# -------------------------------------------------------- claim/report/verify
def test_full_cycle_trusted_worker_auto_accepts():
    mcp, _ = _server()
    t = _call(mcp, "queue_add", task="hi", privacy_class="public")
    claim = _call(mcp, "queue_claim", worker_id="trusted-worker", ticket_id=t["ticket_id"])
    hb = _call(
        mcp, "queue_update", worker_id="trusted-worker", ticket_id=t["ticket_id"],
        lease_token=claim["lease_token"], extend_seconds=60,
    )
    assert hb["status"] == "claimed"
    out = _call(
        mcp, "queue_report", worker_id="trusted-worker", ticket_id=t["ticket_id"],
        lease_token=claim["lease_token"], summary="done", confidence=0.9,
    )
    assert out["ticket_status"] == "done"
    assert out["result_status"] == "approved"
    assert out["result_ref"]
    status = _call(mcp, "queue_status", ticket_id=t["ticket_id"])
    assert status[0]["status"] == "done"
    assert "payload" not in status[0]  # never leaks payload content, only the ref
    assert "payload_ref" in status[0]


def test_untrusted_report_requires_local_only_review():
    mcp, _ = _server()
    t = _call(mcp, "queue_add", task="hi", privacy_class="public")
    claim = _call(mcp, "queue_claim", worker_id="friend-box", ticket_id=t["ticket_id"])
    out = _call(
        mcp, "queue_report", worker_id="friend-box", ticket_id=t["ticket_id"],
        lease_token=claim["lease_token"], summary="maybe",
    )
    assert out["ticket_status"] == "in_review"
    assert out["result_status"] == "pending"

    # An external reviewer cannot promote it.
    assert _call(mcp, "queue_approve", reviewer_id="friend-box", ticket_id=t["ticket_id"]) == {
        "error": "reviewer_not_authorized"
    }
    approved = _call(mcp, "queue_approve", reviewer_id="manager", ticket_id=t["ticket_id"])
    assert approved == {"ticket_status": "done", "result_status": "approved"}


def test_reject_requeues_then_fails_via_mcp():
    mcp, _ = _server()
    t = _call(mcp, "queue_add", task="hi", privacy_class="public", required_capabilities=[])
    # Default max_attempts=3: first reject requeues (attempt 1 of 3 burnt), second
    # exercises the same path again; drive a third attempt to exhaustion.
    for _ in range(2):
        c = _call(mcp, "queue_claim", worker_id="friend-box", ticket_id=t["ticket_id"])
        _call(mcp, "queue_report", worker_id="friend-box", ticket_id=t["ticket_id"], lease_token=c["lease_token"])
        r = _call(mcp, "queue_reject", reviewer_id="manager", ticket_id=t["ticket_id"], reason="bad")
        assert r["ticket_status"] == "open"

    c3 = _call(mcp, "queue_claim", worker_id="friend-box", ticket_id=t["ticket_id"])
    _call(mcp, "queue_report", worker_id="friend-box", ticket_id=t["ticket_id"], lease_token=c3["lease_token"])
    r3 = _call(mcp, "queue_reject", reviewer_id="manager", ticket_id=t["ticket_id"], reason="still bad")
    assert r3["ticket_status"] == "failed"
    status = _call(mcp, "queue_status", ticket_id=t["ticket_id"])
    assert status[0]["status"] == "failed"


# ------------------------------------------------------------------------ link
def test_queue_link_rejects_privacy_downgrade_via_mcp():
    mcp, _ = _server()
    parent = _call(mcp, "queue_add", task="p", privacy_class="repo_sensitive")
    child = _call(mcp, "queue_add", task="c", privacy_class="public")
    assert _call(mcp, "queue_link", ticket_id=child["ticket_id"], depends_on=parent["ticket_id"]) == {
        "error": "privacy_downgrade"
    }
    child2 = _call(mcp, "queue_add", task="c2", privacy_class="repo_sensitive")
    assert _call(mcp, "queue_link", ticket_id=child2["ticket_id"], depends_on=parent["ticket_id"]) == {
        "ok": True
    }


# --------------------------------------------------------- router-as-assigner
def test_enqueue_task_sets_advisory_assignee_without_gating_claims():
    mcp, svc = _server()
    ticket = _call(mcp, "enqueue_task", task="please summarize this public doc", agent_type="tester")
    assert ticket["status"] == "open"
    row = svc.workqueue.get(ticket["ticket_id"])
    # assignee is advisory: present in the row (possibly None), but never
    # required for a claim -- claimability is privacy_class x attested tier only.
    assert "assignee" in row
    claim = _call(mcp, "queue_claim", worker_id="friend-box", ticket_id=ticket["ticket_id"])
    assert claim["ticket_id"] == ticket["ticket_id"]


def test_enqueue_task_still_queues_when_advisory_routing_raises():
    # The routing pass in enqueue_task is advisory-only: if it raises, the ticket
    # must still be enqueued (assignee cleared) and the failure logged as a warning.
    _, svc = _server()

    def _boom(ctx, status):
        raise RuntimeError("router blew up")

    svc.engine.route = _boom  # type: ignore[assignment]
    submission = TaskSubmission(task="please summarize this public doc", agent_type="tester")
    ticket = svc.enqueue_task(submission)

    assert ticket["status"] == "open"
    row = svc.workqueue.get(ticket["ticket_id"])
    assert row["assignee"] is None  # advisory hint cleared on failure
    warns = svc.memory.get_log_slice(row["task_id"], level="warn")
    assert any("advisory routing failed" in w["message"] for w in warns)


def test_queue_status_never_leaks_payload_content():
    mcp, _ = _server()
    secret_text = "this is the raw task body that must never leak in status"
    t = _call(mcp, "queue_add", task=secret_text, privacy_class="public")
    rows = _call(mcp, "queue_status", status="open")
    blob = json.dumps(rows)
    assert secret_text not in blob
    assert any(r["ticket_id"] == t["ticket_id"] for r in rows)
