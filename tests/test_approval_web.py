"""Reference web approve/deny host for the spend authorizer. Offline via threads +
Starlette TestClient (no real network)."""

import threading
import time

import pytest

from agentconnect.common.approval import ApprovalQueue, WebApprovalAuthorizer
from agentconnect.common.authorization import ChargeRequest
from agentconnect.common.memory import SharedMemory
from agentconnect.common.schemas import TaskConstraints, TaskState, TaskSubmission
from agentconnect.model_manager.residency import ResidencyManager
from agentconnect.router.approval_web import create_approval_app
from agentconnect.router.local_client import InProcessLocalClient
from agentconnect.router.provisioning import StubProvisioner
from agentconnect.router.service import RouterService

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _charge():
    return ChargeRequest(provider="openai_paid", kind="paid_cloud", estimated_cost_usd=0.02,
                         task_summary="do a thing", period="monthly",
                         budget_amount_usd=20.0, remaining_usd=19.5)


# ---------------------------------------------------------- queue + authorizer
def test_confirm_charge_blocks_until_resolved():
    q = ApprovalQueue()
    auth = WebApprovalAuthorizer(q, timeout_seconds=5)
    result = {}

    t = threading.Thread(target=lambda: result.setdefault("v", auth.confirm_charge(_charge())))
    t.start()
    # The item shows up as pending, then we approve it.
    for _ in range(100):
        pend = q.pending()
        if pend:
            break
        time.sleep(0.01)
    assert len(pend) == 1 and pend[0]["kind"] == "charge"
    assert q.resolve(pend[0]["id"], True)
    t.join(timeout=5)
    assert result["v"] is True


def test_confirm_charge_denied():
    q = ApprovalQueue()
    auth = WebApprovalAuthorizer(q, timeout_seconds=5)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("v", auth.confirm_charge(_charge())))
    t.start()
    while not q.pending():
        time.sleep(0.01)
    q.resolve(q.pending()[0]["id"], False)
    t.join(timeout=5)
    assert out["v"] is False


def test_timeout_fails_closed():
    q = ApprovalQueue()
    auth = WebApprovalAuthorizer(q, timeout_seconds=0.05)
    assert auth.confirm_charge(_charge()) is False
    assert auth.request_budget("monthly") is None


def test_request_budget_returns_submission():
    q = ApprovalQueue()
    auth = WebApprovalAuthorizer(q, timeout_seconds=5)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("v", auth.request_budget("monthly")))
    t.start()
    while not q.pending():
        time.sleep(0.01)
    q.resolve(q.pending()[0]["id"], {"amount_usd": 30.0, "period": "weekly"})
    t.join(timeout=5)
    assert out["v"] == {"amount_usd": 30.0, "period": "weekly"}


# ------------------------------------------------------------------- HTTP API
def _client(queue, token=None):
    from fastapi.testclient import TestClient

    return TestClient(create_approval_app(queue, token=token))


def test_api_lists_and_resolves_charge():
    q = ApprovalQueue()
    auth = WebApprovalAuthorizer(q, timeout_seconds=5)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("v", auth.confirm_charge(_charge())))
    t.start()
    while not q.pending():
        time.sleep(0.01)
    c = _client(q)
    listed = c.get("/api/pending").json()
    assert listed[0]["provider"] == "openai_paid"
    aid = listed[0]["id"]
    assert c.post(f"/api/charges/{aid}/approve").json()["ok"] is True
    t.join(timeout=5)
    assert out["v"] is True
    # Already resolved -> 404 on a second attempt.
    assert c.post(f"/api/charges/{aid}/deny").status_code == 404


def test_api_budget_submit():
    q = ApprovalQueue()
    auth = WebApprovalAuthorizer(q, timeout_seconds=5)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("v", auth.request_budget("monthly")))
    t.start()
    while not q.pending():
        time.sleep(0.01)
    c = _client(q)
    aid = c.get("/api/pending").json()[0]["id"]
    assert c.post(f"/api/budget/{aid}", json={"amount_usd": 15, "period": "daily"}).json()["ok"]
    t.join(timeout=5)
    assert out["v"] == {"amount_usd": 15.0, "period": "daily"}


def test_api_token_enforced():
    q = ApprovalQueue()
    c = _client(q, token="s3cret")
    assert c.get("/api/pending").status_code == 401
    assert c.get("/api/pending", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    # The dashboard page itself is public (loopback); only /api/* is gated.
    assert c.get("/").status_code == 200


# ----------------------------------------------------------------- end-to-end
def test_router_end_to_end_with_web_authorizer():
    q = ApprovalQueue()
    svc = RouterService.create(
        memory=SharedMemory(), local_client=None,
        provisioner=StubProvisioner(),
        rented_client_factory=lambda cfg, h: InProcessLocalClient(ResidencyManager()),
        authorizer=WebApprovalAuthorizer(q, timeout_seconds=5),
    )
    svc.set_budget(50.0, "monthly")
    result = {}
    t = threading.Thread(
        target=lambda: result.setdefault("s", svc.submit_task(
            TaskSubmission(task="huge private reasoning job", agent_type="repo_scout",
                           constraints=TaskConstraints(privacy_class="repo_sensitive",
                                                       allow_external=False, allow_rented=True))
        ))
    )
    t.start()
    while not q.pending():
        time.sleep(0.01)
    # Approve the rented charge via the queue (as the web endpoint would).
    q.resolve(q.pending()[0]["id"], True)
    t.join(timeout=5)
    assert result["s"].status == TaskState.COMPLETE
