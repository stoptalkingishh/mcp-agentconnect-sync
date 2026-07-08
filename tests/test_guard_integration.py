"""fascia-guard hook in the RouterService (task-in + output scan).

The hook is dormant by default (no FASCIA_GUARD env) — covered implicitly by the
rest of the suite still passing. Here we exercise the ACTIVE and ENFORCING paths.

Determinism note: we use a Stripe-style key (`sk_live_...`), which the privacy
classifier does NOT hard-block (its `sk-` pattern needs a hyphen) but fascia-guard
DOES flag as a secret — so this isolates the guard's added coverage from
privacy_mod's existing secret_sensitive path.
"""
import pytest

from agentconnect.common.memory import SharedMemory
from agentconnect.common.schemas import TaskConstraints, TaskState, TaskSubmission
from agentconnect.model_manager.residency import ResidencyManager
from agentconnect.router import guard_hook
from agentconnect.router.local_client import InProcessLocalClient
from agentconnect.router.service import RouterService

pytestmark = pytest.mark.skipif(
    not guard_hook.available(), reason="fascia-guard not installed"
)

_STRIPE = "update the billing integration with sk_live_0123456789abcdefghij here"


def _service():
    mem = SharedMemory()
    local = InProcessLocalClient(ResidencyManager())
    return RouterService.create(memory=mem, local_client=local)


def test_dormant_by_default(monkeypatch):
    monkeypatch.delenv("FASCIA_GUARD", raising=False)
    monkeypatch.delenv("FASCIA_GUARD_ENFORCE", raising=False)
    assert not guard_hook.active()
    assert guard_hook.scan_task(_STRIPE, "t") is None


def test_advisory_logs_but_does_not_block(monkeypatch):
    # FASCIA_GUARD=1 -> scan + log, but never change the outcome.
    monkeypatch.setenv("FASCIA_GUARD", "1")
    monkeypatch.delenv("FASCIA_GUARD_ENFORCE", raising=False)
    assert guard_hook.active() and not guard_hook.enforcing()

    svc = _service()
    sub = TaskSubmission(
        task="Summarize this public changelog for the release notes.",
        agent_type="patch_worker",
        constraints=TaskConstraints(privacy_class="public"),
    )
    summary = svc.submit_task(sub)
    # Still completes — advisory mode is non-disruptive.
    assert summary.status == TaskState.COMPLETE
    # And the guard actually ran (a log line was emitted).
    logs = svc.memory.get_log_slice(summary.task_id, query="fascia-guard")
    assert any("fascia-guard[task]" in row["message"] for row in logs)


def test_enforce_blocks_secret_privacy_misses(monkeypatch):
    monkeypatch.setenv("FASCIA_GUARD_ENFORCE", "1")
    assert guard_hook.enforcing()

    svc = _service()
    sub = TaskSubmission(task=_STRIPE, agent_type="patch_worker")
    summary = svc.submit_task(sub)

    assert summary.status == TaskState.REJECTED
    assert "fascia-guard" in (summary.summary or "").lower()
    # Blocked before any routing decision was made.
    assert svc.memory.get_routing_decisions(summary.task_id) == []


def test_enforce_lets_clean_task_through(monkeypatch):
    monkeypatch.setenv("FASCIA_GUARD_ENFORCE", "1")
    svc = _service()
    sub = TaskSubmission(
        task="Refactor the pagination helper in this module.",
        agent_type="patch_worker",
        constraints=TaskConstraints(privacy_class="repo_sensitive"),
    )
    summary = svc.submit_task(sub)
    assert summary.status == TaskState.COMPLETE
