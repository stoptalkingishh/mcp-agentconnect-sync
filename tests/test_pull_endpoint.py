"""Federated work-queue pull surface over HTTP (S3): the additive mount on
``create_worker_app`` that lets a peer identity discover, claim, and report
work through the same trust x privacy authorization core as the MCP surface
(test_workqueue_mcp.py) and the atomic claim core (test_workqueue.py).

Offline via starlette/FastAPI TestClient: ``X-Client-Cert-DN`` stands in for
the mTLS peer certificate (same seam ``asgi_identity._peer_identity`` and
``test_runtime_transport.py::test_worker_allowlist_middleware`` already use),
and a scripted ``tier_resolver`` dict stands in for config/workers.yaml.
"""

import pytest
from fastapi.testclient import TestClient

from agentconnect.common.config import load_routing
from agentconnect.common.memory import SharedMemory
from agentconnect.common.schemas import PrivacyClass
from agentconnect.common.workqueue import WorkQueue
from agentconnect.runtime import add_pull_routes, create_worker_app
from agentconnect.runtime.agent import LangGraphAgentRuntime, RuntimeConfig

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

LOCAL = "local_only"
EXTERNAL = "external"

TIERS = {
    "trusted-worker": LOCAL,
    "friend-box": EXTERNAL,
}


class _NoopModelSource:
    def generate(self, req):
        raise AssertionError("the pull surface never calls the runtime's model source")


def _app(tmp_path, queue=None, tier_resolver=None, trust_proxy_headers=True):
    runtime = LangGraphAgentRuntime(_NoopModelSource(), RuntimeConfig(workspace_root=str(tmp_path)))
    app = create_worker_app(runtime)
    resolver = tier_resolver if tier_resolver is not None else TIERS.get
    # The TestClient supplies the peer identity via an X-Client-Cert-DN header,
    # which stands in for a header-stripping mTLS proxy — so tests opt in.
    return add_pull_routes(app, queue, resolver, trust_proxy_headers=trust_proxy_headers)


def _wq():
    mem = SharedMemory()
    return WorkQueue(mem, load_routing()), mem


def _dn(client: TestClient, identity: str) -> dict:
    return {"X-Client-Cert-DN": identity}


# --------------------------------------------------------------------- trim
def test_routes_absent_when_queue_is_none(tmp_path):
    app = _app(tmp_path, queue=None)
    client = TestClient(app)
    # Core worker routes untouched.
    assert client.get("/can_accept").status_code == 200
    # Pull routes never mounted.
    assert client.get("/queue/next", headers=_dn(client, "trusted-worker")).status_code == 404


# ------------------------------------------------------------------ identity
def test_missing_identity_is_403(tmp_path):
    wq, _ = _wq()
    client = TestClient(_app(tmp_path, queue=wq))
    assert client.get("/queue/next").status_code == 403


def test_unknown_identity_is_403(tmp_path):
    wq, _ = _wq()
    client = TestClient(_app(tmp_path, queue=wq))
    resp = client.get("/queue/next", headers=_dn(client, "nobody"))
    assert resp.status_code == 403


def test_forwarded_header_identity_not_trusted_without_optin(tmp_path):
    # SECURITY: with the default (no trusted header-stripping proxy asserted), a
    # client-settable X-Client-Cert-DN header MUST NOT be accepted as an identity
    # anchor — otherwise a direct peer spoofs a trusted tier and claims/auto-
    # approves work. The header is ignored -> no identity -> 403, even for a name
    # the resolver would otherwise map to local_only.
    wq, _ = _wq()
    wq.add(privacy_class=PrivacyClass.repo_sensitive, payload="internal secrets", origin="o")
    client = TestClient(_app(tmp_path, queue=wq, trust_proxy_headers=False))
    resp = client.get("/queue/next", headers=_dn(client, "trusted-worker"), params={"max": 5})
    assert resp.status_code == 403


# ---------------------------------------------------------------- authz gate
def test_low_trust_identity_denied_repo_sensitive_work(tmp_path):
    wq, _ = _wq()
    ticket = wq.add(privacy_class=PrivacyClass.repo_sensitive, payload="internal secrets", origin="o")
    public = wq.add(privacy_class=PrivacyClass.public, payload="hello", origin="o")

    client = TestClient(_app(tmp_path, queue=wq))
    resp = client.get("/queue/next", headers=_dn(client, "friend-box"), params={"max": 5})
    assert resp.status_code == 200
    ids = {t["ticket_id"] for t in resp.json()["tickets"]}
    assert ticket["ticket_id"] not in ids
    assert public["ticket_id"] in ids

    # A trusted local_only identity CAN see/claim the repo_sensitive ticket.
    wq2, _ = _wq()
    sensitive2 = wq2.add(privacy_class=PrivacyClass.repo_sensitive, payload="s", origin="o")
    client2 = TestClient(_app(tmp_path, queue=wq2))
    resp2 = client2.get("/queue/next", headers=_dn(client2, "trusted-worker"), params={"max": 5})
    assert sensitive2["ticket_id"] in {t["ticket_id"] for t in resp2.json()["tickets"]}


# ------------------------------------------------------------ claim/report
def test_claim_heartbeat_report_round_trip_trusted(tmp_path):
    wq, _ = _wq()
    wq.add(privacy_class=PrivacyClass.public, payload="hi", origin="o")
    client = TestClient(_app(tmp_path, queue=wq))

    got = client.get("/queue/next", headers=_dn(client, "trusted-worker")).json()["tickets"]
    assert len(got) == 1
    ticket_id = got[0]["ticket_id"]
    lease_token = got[0]["lease_token"]

    hb = client.post(
        f"/queue/{ticket_id}/heartbeat",
        headers=_dn(client, "trusted-worker"),
        json={"lease_token": lease_token, "extend_seconds": 60},
    )
    assert hb.status_code == 200
    assert hb.json()["status"] == "claimed"

    report = client.post(
        f"/queue/{ticket_id}/report",
        headers=_dn(client, "trusted-worker"),
        json={"lease_token": lease_token, "summary": "done", "confidence": 0.9},
    )
    assert report.status_code == 200
    body = report.json()
    assert body["ticket_status"] == "done"
    assert body["result_status"] == "approved"


def test_untrusted_report_lands_in_review_over_http(tmp_path):
    wq, _ = _wq()
    wq.add(privacy_class=PrivacyClass.public, payload="hi", origin="o")
    client = TestClient(_app(tmp_path, queue=wq))

    got = client.get("/queue/next", headers=_dn(client, "friend-box")).json()["tickets"]
    ticket_id = got[0]["ticket_id"]
    lease_token = got[0]["lease_token"]

    report = client.post(
        f"/queue/{ticket_id}/report",
        headers=_dn(client, "friend-box"),
        json={"lease_token": lease_token, "summary": "maybe"},
    )
    body = report.json()
    assert body["ticket_status"] == "in_review"
    assert body["result_status"] == "pending"


def test_report_optional_worker_result_fields_survive(tmp_path):
    # QueueReportBody's evidence_refs / recommended_next_action feed WorkerResult
    # and must reach the stored result artifact intact.
    import json

    wq, mem = _wq()
    wq.add(privacy_class=PrivacyClass.public, payload="hi", origin="o")
    client = TestClient(_app(tmp_path, queue=wq))

    got = client.get("/queue/next", headers=_dn(client, "trusted-worker")).json()["tickets"]
    ticket_id, lease_token = got[0]["ticket_id"], got[0]["lease_token"]

    report = client.post(
        f"/queue/{ticket_id}/report",
        headers=_dn(client, "trusted-worker"),
        json={
            "lease_token": lease_token,
            "summary": "done",
            "evidence_refs": ["artifact_ev1", "artifact_ev2"],
            "recommended_next_action": "merge the patch",
        },
    )
    assert report.status_code == 200
    result_ref = report.json()["result_ref"]
    stored = json.loads(mem.read_artifact_chunk(result_ref, 0, 10000).content)
    assert stored["evidence_refs"] == ["artifact_ev1", "artifact_ev2"]
    assert stored["recommended_next_action"] == "merge the patch"


def test_report_with_stale_lease_token_is_refused(tmp_path):
    wq, _ = _wq()
    wq.add(privacy_class=PrivacyClass.public, payload="hi", origin="o")
    client = TestClient(_app(tmp_path, queue=wq))

    got = client.get("/queue/next", headers=_dn(client, "trusted-worker")).json()["tickets"]
    ticket_id = got[0]["ticket_id"]

    report = client.post(
        f"/queue/{ticket_id}/report",
        headers=_dn(client, "trusted-worker"),
        json={"lease_token": "not-the-real-token", "summary": "done"},
    )
    assert report.status_code == 200
    assert report.json() == {"error": "lease_lost"}
