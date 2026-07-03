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
