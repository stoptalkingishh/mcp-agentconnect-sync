"""Read-only broker-side queue operator host (S3). Offline via the Starlette
TestClient (no real network), mirroring test_approval_web.py's style."""

import pytest

from agentconnect.common.config import load_routing
from agentconnect.common.memory import SharedMemory
from agentconnect.common.schemas import PrivacyClass
from agentconnect.common.workqueue import WorkQueue
from agentconnect.router.queue_web import create_queue_operator_app

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

LOCAL = "local_only"
EXTERNAL = "external"


def _wq():
    return WorkQueue(SharedMemory(), load_routing())


def _client(wq, token=None, reviewer_id="operator", reviewer_tier=LOCAL):
    from fastapi.testclient import TestClient

    return TestClient(create_queue_operator_app(wq, reviewer_id, reviewer_tier, token=token))


def test_list_pending_stats_are_payload_free():
    wq = _wq()
    secret_text = "the raw task body that must never leak to the operator web host"
    t = wq.add(privacy_class=PrivacyClass.public, payload=secret_text, origin="o")
    claim = wq.claim(EXTERNAL, EXTERNAL, t["ticket_id"])
    result_text = "the raw worker result body that must never leak either"
    wq.report(EXTERNAL, EXTERNAL, t["ticket_id"], claim["lease_token"],
              {"status": "completed", "summary": result_text})

    c = _client(wq)
    listed = c.get("/api/list").json()
    pending = c.get("/api/pending").json()
    stats = c.get("/api/stats").json()

    blob = str(listed) + str(pending) + str(stats)
    assert secret_text not in blob
    assert result_text not in blob
    assert all("task_id" not in r for r in listed)
    assert all("task_id" not in r for r in pending)

    assert any(r["ticket_id"] == t["ticket_id"] for r in listed)
    assert len(pending) == 1 and pending[0]["ticket_id"] == t["ticket_id"]
    assert pending[0]["status"] == "in_review"
    assert stats["by_status"]["in_review"] == 1
    assert "capability_requirements" in stats


def test_approve_transitions_ticket_and_clears_pending():
    wq = _wq()
    t = wq.add(privacy_class=PrivacyClass.public, payload="hi", origin="o")
    claim = wq.claim(EXTERNAL, EXTERNAL, t["ticket_id"])
    wq.report(EXTERNAL, EXTERNAL, t["ticket_id"], claim["lease_token"], {"status": "completed"})

    c = _client(wq)
    assert len(c.get("/api/pending").json()) == 1
    resp = c.post(f"/api/tickets/{t['ticket_id']}/approve")
    assert resp.json() == {"ticket_status": "done", "result_status": "approved"}
    assert wq.get(t["ticket_id"])["status"] == "done"
    assert c.get("/api/pending").json() == []


def test_reject_requeues_ticket():
    wq = _wq()
    t = wq.add(privacy_class=PrivacyClass.public, payload="hi", origin="o", max_attempts=3)
    claim = wq.claim(EXTERNAL, EXTERNAL, t["ticket_id"])
    wq.report(EXTERNAL, EXTERNAL, t["ticket_id"], claim["lease_token"], {"status": "completed"})

    c = _client(wq)
    resp = c.post(f"/api/tickets/{t['ticket_id']}/reject", json={"reason": "looks wrong"})
    assert resp.json() == {"ticket_status": "open", "result_status": "rejected"}
    assert wq.get(t["ticket_id"])["status"] == "open"


def test_reviewer_tier_is_reenforced_by_the_workqueue_not_the_web_host():
    # A misconfigured operator identity (non-local_only) is refused by the
    # WorkQueue itself, even though this host never re-derives that check.
    wq = _wq()
    t = wq.add(privacy_class=PrivacyClass.public, payload="hi", origin="o")
    claim = wq.claim(EXTERNAL, EXTERNAL, t["ticket_id"])
    wq.report(EXTERNAL, EXTERNAL, t["ticket_id"], claim["lease_token"], {"status": "completed"})

    c = _client(wq, reviewer_id="not-a-reviewer", reviewer_tier=EXTERNAL)
    resp = c.post(f"/api/tickets/{t['ticket_id']}/approve")
    assert resp.json() == {"error": "reviewer_not_authorized"}
    assert wq.get(t["ticket_id"])["status"] == "in_review"


def test_unauthenticated_access_is_refused():
    wq = _wq()
    wq.add(privacy_class=PrivacyClass.public, payload="hi", origin="o")
    c = _client(wq, token="s3cret")
    assert c.get("/api/list").status_code == 401
    assert c.get("/api/pending").status_code == 401
    assert c.get("/api/stats").status_code == 401
    assert c.post("/api/tickets/nope/approve").status_code == 401
    assert c.post("/api/tickets/nope/reject").status_code == 401

    hdr = {"Authorization": "Bearer s3cret"}
    assert c.get("/api/list", headers=hdr).status_code == 200
    assert c.get("/api/pending", headers=hdr).status_code == 200
    assert c.get("/api/stats", headers=hdr).status_code == 200
    # The dashboard page itself is public (loopback), matching approval_web.
    assert c.get("/").status_code == 200
