"""Smoke test for examples/federation_demo.py — it is executable documentation of
the whole federation loop, so it must stay runnable and keep demonstrating the
crown-jewel invariants. Runs offline, in-process."""

import io
import importlib.util
from contextlib import redirect_stdout
from pathlib import Path

_DEMO = Path(__file__).resolve().parents[1] / "examples" / "federation_demo.py"


def _load_demo():
    spec = importlib.util.spec_from_file_location("federation_demo", _DEMO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_demo_runs_and_shows_the_invariants():
    mod = _load_demo()
    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.main()
    out = buf.getvalue()

    # The authorization boundary fired: external refused repo_sensitive, and the
    # ticket stayed open (payload never left the broker).
    assert "external claim of repo_sensitive -> {'error': 'not_authorized'}" in out
    assert "still 'open'" in out

    # Untrusted result did NOT auto-accept — it landed in_review for spot-check.
    assert "ticket_status=in_review" in out
    assert "awaiting review" in out

    # secret_sensitive parked at enqueue, never leasable.
    assert "secret_sensitive_not_pullable" in out

    # Trusted local result accepted immediately; review then clears the backlog.
    assert "ticket_status=done" in out
    assert "reviewer approved" in out


def test_demo_final_state_is_two_done_one_parked():
    """Reach past stdout into the queue: drive the same flow and assert the end
    state numerically, so a silent behavioural regression can't hide behind the
    printed banners."""
    import sys

    root = _DEMO.resolve().parents[1]
    sys.path.insert(0, str(root / "packages" / "agentconnect-core" / "src"))
    from agentconnect.common.config import load_routing
    from agentconnect.common.memory import SharedMemory
    from agentconnect.common.schemas import PrivacyClass, WorkerResult
    from agentconnect.common.workqueue import WorkQueue

    wq = WorkQueue(SharedMemory(), load_routing())
    pub = wq.add(privacy_class=PrivacyClass.public, payload="p", origin="o",
                 required_capabilities=["python"])
    repo = wq.add(privacy_class=PrivacyClass.repo_sensitive, payload="r", origin="o",
                  required_capabilities=["python"])
    sec = wq.add(privacy_class=PrivacyClass.secret_sensitive, payload="s", origin="o")
    assert sec["park_reason"] == "secret_sensitive_not_pullable"

    # external drains public -> in_review
    c = wq.claim_next("friend@box", "external", capabilities={"python"}, max=1)[0]
    wq.report("friend@box", "external", c["ticket_id"], c["lease_token"],
              WorkerResult(status="completed", summary="x"))
    assert wq.get(pub["ticket_id"])["status"] == "in_review"
    # external refused repo_sensitive
    assert wq.claim("friend@box", "external", repo["ticket_id"]) == {"error": "not_authorized"}
    assert wq.get(repo["ticket_id"])["status"] == "open"

    # local drains repo_sensitive -> done immediately
    c2 = wq.claim_next("me@laptop", "local_only", capabilities={"python"}, max=1)
    c2 = [t for t in c2 if t["privacy_class"] == "repo_sensitive"][0]
    wq.report("me@laptop", "local_only", c2["ticket_id"], c2["lease_token"],
              WorkerResult(status="completed", summary="y"))
    assert wq.get(repo["ticket_id"])["status"] == "done"

    # reviewer approves the external result
    wq.approve("me@laptop", "local_only", pub["ticket_id"])

    stats = wq.stats()
    assert stats["by_status"] == {"done": 2, "parked": 1}
