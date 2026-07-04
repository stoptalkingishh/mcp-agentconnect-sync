"""Worker-side pull loop (PullWorker): the compute-contributor program that
claims work off the broker's federated pull endpoint, executes it with its OWN
local runtime, and reports back — turning "the queue exists" into "a box is
consuming it". Offline via FastAPI TestClient; X-Client-Cert-DN stands in for
the mTLS peer cert (same seam as test_pull_endpoint.py), trust_proxy_headers
opted in.
"""

import pytest
from fastapi.testclient import TestClient

from agentconnect.common.config import load_routing
from agentconnect.common.memory import SharedMemory
from agentconnect.common.schemas import GenerateResponse
from agentconnect.common.workqueue import WorkQueue
from agentconnect.runtime import PullWorker, add_pull_routes, create_worker_app
from agentconnect.runtime.agent import LangGraphAgentRuntime, RuntimeConfig

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

TIERS = {"trusted-worker": "local_only", "friend-box": "external"}


class _FinishSource:
    """A model source that immediately finishes, echoing the task it was given
    so tests can assert the redacted payload actually reached the worker."""

    def generate(self, req):
        last_user = ""
        for m in reversed(req.messages):
            if m.get("role") == "user":
                last_user = str(m.get("content", ""))
                break
        return GenerateResponse(
            request_id=req.request_id,
            model_id=req.model_id,
            output_text=f"done: {last_user}",
        )


def _broker(tmp_path):
    """A broker app in front of a fresh work-queue. Returns (client, wq, mem)."""
    mem = SharedMemory()
    wq = WorkQueue(mem, load_routing())
    # The broker's own runtime is never exercised by the pull path — the compute
    # happens in the PullWorker's runtime, not here.
    app = create_worker_app(LangGraphAgentRuntime(_FinishSource(), RuntimeConfig(workspace_root=str(tmp_path))))
    add_pull_routes(app, wq, TIERS.get, trust_proxy_headers=True)
    return TestClient(app), wq, mem


def _worker(client, identity, tmp_path, **kw):
    runtime = LangGraphAgentRuntime(_FinishSource(), RuntimeConfig(workspace_root=str(tmp_path)))
    return PullWorker(
        runtime, client=client, identity_headers={"X-Client-Cert-DN": identity}, **kw
    )


def _status(wq, ticket_id):
    return wq._raw(ticket_id)["status"]


# --------------------------------------------------------------- happy path
def test_trusted_worker_claims_runs_and_reports_to_done(tmp_path):
    client, wq, _ = _broker(tmp_path)
    t = wq.add(task="t", origin="test", privacy_class="public", payload="summarize the notes")
    worker = _worker(client, "trusted-worker", tmp_path)

    outcome = worker.run_once()

    assert outcome is not None
    assert outcome["ticket_id"] == t["ticket_id"]
    # The redacted payload actually reached the worker and drove the run.
    assert "summarize the notes" in outcome["result"].summary
    # A local_only (trusted) worker's completed result is auto-accepted.
    assert _status(wq, t["ticket_id"]) == "done"


def test_payload_is_delivered_inline_on_claim(tmp_path):
    client, wq, _ = _broker(tmp_path)
    wq.add(task="t", origin="test", privacy_class="public", payload="THE ACTUAL TASK BODY")
    worker = _worker(client, "trusted-worker", tmp_path)

    tickets = worker.claim(max_tickets=1)

    assert tickets and tickets[0]["payload"] == "THE ACTUAL TASK BODY"
    # The internal task_id is still never handed to the worker.
    assert "task_id" not in tickets[0]


# ------------------------------------------------------------- authorization
def test_untrusted_worker_cannot_see_repo_sensitive(tmp_path):
    client, wq, _ = _broker(tmp_path)
    wq.add(task="t", origin="test", privacy_class="repo_sensitive", payload="private code")
    worker = _worker(client, "friend-box", tmp_path)  # external tier

    assert worker.claim(max_tickets=5) == []
    assert worker.run_once() is None


def test_untrusted_worker_result_lands_in_review_not_done(tmp_path):
    client, wq, _ = _broker(tmp_path)
    t = wq.add(task="t", origin="test", privacy_class="public", payload="public task")
    worker = _worker(client, "friend-box", tmp_path)  # external tier

    outcome = worker.run_once()

    assert outcome is not None  # external MAY claim public work
    # ...but an untrusted result is never silently promoted to truth.
    assert _status(wq, t["ticket_id"]) == "in_review"


def test_unknown_identity_is_refused(tmp_path):
    client, wq, _ = _broker(tmp_path)
    wq.add(task="t", origin="test", privacy_class="public", payload="x")
    worker = _worker(client, "stranger", tmp_path)  # not in TIERS -> 403

    with pytest.raises(Exception):
        worker.claim()


# ----------------------------------------------------------------- draining
def test_empty_queue_returns_none(tmp_path):
    client, _, _ = _broker(tmp_path)
    worker = _worker(client, "trusted-worker", tmp_path)
    assert worker.run_once() is None


def test_run_forever_drains_multiple_then_stops(tmp_path):
    client, wq, _ = _broker(tmp_path)
    ids = [
        wq.add(task="t", origin="test", privacy_class="public", payload=f"job {i}", dedup_key=f"j{i}")[
            "ticket_id"
        ]
        for i in range(3)
    ]
    worker = _worker(client, "trusted-worker", tmp_path, poll_interval=0)

    # Bounded so an always-empty tail can't loop forever; sleep is a no-op.
    processed = worker.run_forever(max_iterations=6, sleep=lambda _s: None)

    assert processed == 3
    assert all(_status(wq, tid) == "done" for tid in ids)


def test_run_forever_survives_transient_http_status_error(tmp_path):
    # Regression: a transient broker failure (e.g. 503 store_busy under write
    # contention) is retryable by contract, but run_once's raise_for_status()
    # calls used to let the httpx.HTTPStatusError propagate straight out of
    # run_forever, killing the whole worker daemon on the first blip instead
    # of backing off and polling again.
    import httpx

    client, wq, _ = _broker(tmp_path)
    t = wq.add(task="t", origin="test", privacy_class="public", payload="job")["ticket_id"]
    worker = _worker(client, "trusted-worker", tmp_path, poll_interval=0)

    real_claim = worker.claim
    calls = {"n": 0}

    def flaky_claim(max_tickets=1):
        calls["n"] += 1
        if calls["n"] == 1:
            req = httpx.Request("GET", "http://test/queue/next")
            resp = httpx.Response(503, request=req)
            raise httpx.HTTPStatusError("store_busy", request=req, response=resp)
        return real_claim(max_tickets=max_tickets)

    worker.claim = flaky_claim

    processed = worker.run_forever(max_iterations=5, sleep=lambda _s: None)

    assert calls["n"] >= 2  # the loop retried past the first failure
    assert processed == 1
    assert _status(wq, t) == "done"


# ----------------------------------------------------------------- heartbeat
def _make_slow(worker, seconds):
    """Make the worker's execute() take `seconds`, so a heartbeat can fire mid-run."""
    import time as _t

    orig = worker.execute
    worker.execute = lambda ticket: (_t.sleep(seconds), orig(ticket))[1]


def test_heartbeat_renews_lease_while_a_slow_task_runs(tmp_path):
    client, wq, _ = _broker(tmp_path)
    t = wq.add(task="t", origin="test", privacy_class="public", payload="slow job")["ticket_id"]
    worker = _worker(client, "trusted-worker", tmp_path, heartbeat_interval=0.02)

    beats = []
    real_hb = worker.heartbeat
    worker.heartbeat = lambda tid, tok: (beats.append(tid), real_hb(tid, tok))[1]
    _make_slow(worker, 0.06)  # spans ~3 heartbeat intervals

    outcome = worker.run_once()

    assert outcome is not None
    assert len(beats) >= 1  # the lease was actively renewed mid-run
    assert _status(wq, t) == "done"


def test_heartbeat_disabled_by_default(tmp_path):
    client, wq, _ = _broker(tmp_path)
    wq.add(task="t", origin="test", privacy_class="public", payload="job")
    worker = _worker(client, "trusted-worker", tmp_path)  # heartbeat_interval defaults to 0

    beats = []
    worker.heartbeat = lambda tid, tok: beats.append(tid)
    _make_slow(worker, 0.03)

    assert worker.run_once() is not None
    assert beats == []


# ------------------------------------------------------- payload delivery
def test_run_once_refuses_payload_error_without_executing(tmp_path):
    # A delivery error (e.g. a lease race between claim and payload_for) must
    # short-circuit to a reported failure — never fall through to execute() on
    # an empty/missing payload and report a bogus success.
    client, wq, _ = _broker(tmp_path)
    worker = _worker(client, "trusted-worker", tmp_path)

    fake_ticket = {"ticket_id": "wq_fake", "lease_token": "tok",
                   "payload": None, "payload_error": "lease_lost"}
    worker.claim = lambda max_tickets=1: [fake_ticket]
    executed = []
    worker.execute = lambda ticket: executed.append(ticket) or pytest.fail("must not execute")
    reported = []
    real_report = worker.report
    worker.report = lambda tid, tok, result: (reported.append(result), {"ticket_status": "failed"})[1]

    outcome = worker.run_once()

    assert executed == []
    assert len(reported) == 1
    assert reported[0].status == "failed"
    assert "lease_lost" in reported[0].summary
    assert outcome["report"] == {"ticket_status": "failed"}


def test_run_once_refuses_null_payload_without_payload_error_key(tmp_path):
    # payload=None with no payload_error key at all (e.g. a future delivery
    # path that forgets to set the reason) must still be refused, not executed.
    client, wq, _ = _broker(tmp_path)
    worker = _worker(client, "trusted-worker", tmp_path)

    fake_ticket = {"ticket_id": "wq_fake2", "lease_token": "tok", "payload": None}
    worker.claim = lambda max_tickets=1: [fake_ticket]
    worker.execute = lambda ticket: pytest.fail("must not execute")
    reported = []
    worker.report = lambda tid, tok, result: (reported.append(result), {"ticket_status": "failed"})[1]

    worker.run_once()

    assert len(reported) == 1
    assert reported[0].status == "failed"
    assert "payload_missing" in reported[0].summary


def test_heartbeat_failure_is_swallowed_report_is_authoritative(tmp_path):
    client, wq, _ = _broker(tmp_path)
    t = wq.add(task="t", origin="test", privacy_class="public", payload="job")["ticket_id"]
    worker = _worker(client, "trusted-worker", tmp_path, heartbeat_interval=0.02)

    def boom(tid, tok):
        raise RuntimeError("transport hiccup")

    worker.heartbeat = boom
    _make_slow(worker, 0.05)

    # A throwing heartbeat must not crash the run; report still lands the result.
    outcome = worker.run_once()
    assert outcome is not None
    assert _status(wq, t) == "done"
