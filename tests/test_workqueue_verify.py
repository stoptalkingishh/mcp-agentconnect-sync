"""Work-queue verification gate (S1): a report from untrusted federated compute
can NEVER silently become truth. Only local_only auto-accepts; every other tier
lands in_review until a local_only reviewer approves. Reject requeues while
attempts remain, then fails. Evaluation rows mirror the outcome."""

from agentconnect.common.config import load_routing
from agentconnect.common.memory import SharedMemory
from agentconnect.common.schemas import PrivacyClass, TaskState
from agentconnect.common.workqueue import WorkQueue

LOCAL = "local_only"
EXTERNAL = "external"


def _wq():
    mem = SharedMemory()
    return WorkQueue(mem, load_routing()), mem


def test_trusted_report_auto_accepts_and_completes_task():
    wq, mem = _wq()
    task_id = mem.create_task({"task": "demo"})
    t = wq.add(privacy_class=PrivacyClass.repo_sensitive, payload="x", origin="o", task_id=task_id)
    c = wq.claim("trusted", LOCAL, t["ticket_id"])
    out = wq.report("trusted", LOCAL, t["ticket_id"], c["lease_token"],
                    {"summary": "ok", "confidence": 0.9})
    assert out["ticket_status"] == "done"
    assert out["result_status"] == "approved"
    assert wq.get(t["ticket_id"])["result_ref"] is not None
    assert mem.get_task(task_id)["state"] == TaskState.COMPLETE.value


def test_untrusted_report_lands_in_review_not_complete():
    wq, mem = _wq()
    task_id = mem.create_task({"task": "demo"})
    t = wq.add(privacy_class=PrivacyClass.public, payload="x", origin="o", task_id=task_id)
    c = wq.claim("friend", EXTERNAL, t["ticket_id"])
    out = wq.report("friend", EXTERNAL, t["ticket_id"], c["lease_token"], {"summary": "maybe"})
    assert out["ticket_status"] == "in_review"
    assert out["result_status"] == "pending"
    # Truth is withheld: the task is only REVIEW_READY, not COMPLETE.
    assert mem.get_task(task_id)["state"] == TaskState.REVIEW_READY.value


def test_approve_requires_local_only_reviewer_then_promotes():
    wq, mem = _wq()
    task_id = mem.create_task({"task": "demo"})
    t = wq.add(privacy_class=PrivacyClass.public, payload="x", origin="o", task_id=task_id)
    c = wq.claim("friend", EXTERNAL, t["ticket_id"])
    wq.report("friend", EXTERNAL, t["ticket_id"], c["lease_token"], {"summary": "r"})
    # A non-local reviewer cannot promote.
    assert wq.approve("mgr", EXTERNAL, t["ticket_id"]) == {"error": "reviewer_not_authorized"}
    assert wq.get(t["ticket_id"])["status"] == "in_review"
    # A local_only manager promotes it to truth.
    out = wq.approve("mgr", LOCAL, t["ticket_id"])
    assert out == {"ticket_status": "done", "result_status": "approved"}
    assert mem.get_task(task_id)["state"] == TaskState.COMPLETE.value


def test_reject_requeues_while_attempts_remain_then_fails():
    wq, mem = _wq()
    task_id = mem.create_task({"task": "demo"})
    t = wq.add(privacy_class=PrivacyClass.public, payload="x", origin="o",
               task_id=task_id, max_attempts=2)
    # Attempt 1: report -> in_review -> reject -> requeued open.
    c1 = wq.claim("friend", EXTERNAL, t["ticket_id"])
    wq.report("friend", EXTERNAL, t["ticket_id"], c1["lease_token"], {})
    r1 = wq.reject("mgr", LOCAL, t["ticket_id"], reason="bad")
    assert r1["ticket_status"] == "open"
    assert wq.get(t["ticket_id"])["status"] == "open"
    # Attempt 2 exhausts max_attempts: reject -> failed.
    c2 = wq.claim("friend", EXTERNAL, t["ticket_id"])
    wq.report("friend", EXTERNAL, t["ticket_id"], c2["lease_token"], {})
    r2 = wq.reject("mgr", LOCAL, t["ticket_id"], reason="still bad")
    assert r2["ticket_status"] == "failed"
    assert mem.get_task(task_id)["state"] == TaskState.FAILED.value


def test_trusted_failure_report_is_not_promoted_to_done():
    # A trusted worker that reports a FAILURE must not have it recorded as truth.
    # With attempts remaining the ticket requeues; the task never goes COMPLETE.
    wq, mem = _wq()
    task_id = mem.create_task({"task": "demo"})
    t = wq.add(privacy_class=PrivacyClass.repo_sensitive, payload="x", origin="o",
               task_id=task_id, max_attempts=3)
    c = wq.claim("trusted", LOCAL, t["ticket_id"])
    out = wq.report("trusted", LOCAL, t["ticket_id"], c["lease_token"],
                    {"status": "error", "confidence": 0.0})
    assert out["ticket_status"] == "open"          # requeued, not done
    assert out["result_status"] == "failed"
    row = wq.get(t["ticket_id"])
    assert row["status"] == "open"
    assert row["result_status"] == "failed"
    assert row["lease_token"] is None              # lease dropped on requeue
    assert mem.get_task(task_id)["state"] != TaskState.COMPLETE.value
    # The failure is recorded as a failed eval, not a success.
    agg = {r["provider"]: r for r in mem.provider_eval_aggregate()}
    assert agg["trusted"]["successes"] == 0


def test_trusted_failure_report_fails_terminally_when_attempts_exhausted():
    wq, mem = _wq()
    task_id = mem.create_task({"task": "demo"})
    t = wq.add(privacy_class=PrivacyClass.repo_sensitive, payload="x", origin="o",
               task_id=task_id, max_attempts=1)
    c = wq.claim("trusted", LOCAL, t["ticket_id"])  # attempts -> 1 == max
    out = wq.report("trusted", LOCAL, t["ticket_id"], c["lease_token"], {"status": "failed"})
    assert out["ticket_status"] == "failed"
    assert wq.get(t["ticket_id"])["status"] == "failed"
    assert mem.get_task(task_id)["state"] == TaskState.FAILED.value


def test_report_cannot_clobber_a_concurrent_reclaim_after_reaper():
    # The report fence is a guarded write, not check-then-act: a slow worker whose
    # lease expired and was re-claimed by another worker cannot overwrite the live
    # claim with its stale result.
    wq, _ = _wq()
    t = wq.add(privacy_class=PrivacyClass.public, payload="x", origin="o", max_attempts=3)
    c1 = wq.claim("slow", LOCAL, t["ticket_id"], lease_seconds=10, now=1000.0)
    wq.reap_expired(now=2000.0)                       # requeue, clear token
    c2 = wq.claim("fast", LOCAL, t["ticket_id"], now=2001.0)  # fresh claim + token
    # Slow worker reports under its stale token: refused, and fast's claim intact.
    assert wq.report("slow", LOCAL, t["ticket_id"], c1["lease_token"], {"summary": "stale"},
                     now=2002.0) == {"error": "lease_lost"}
    row = wq.get(t["ticket_id"])
    assert row["status"] == "claimed"
    assert row["lease_holder"] == "fast"
    assert row["lease_token"] == c2["lease_token"]


def test_evaluation_rows_reflect_outcomes():
    wq, mem = _wq()
    # Trusted success -> one 'completed' eval for the worker.
    t = wq.add(privacy_class=PrivacyClass.public, payload="x", origin="o")
    c = wq.claim("worker_ok", LOCAL, t["ticket_id"])
    wq.report("worker_ok", LOCAL, t["ticket_id"], c["lease_token"], {"confidence": 0.8})
    agg = {r["provider"]: r for r in mem.provider_eval_aggregate()}
    assert agg["worker_ok"]["successes"] == 1

    # Untrusted then rejected -> a 'failed' eval recorded at review time.
    t2 = wq.add(privacy_class=PrivacyClass.public, payload="y", origin="o", max_attempts=1)
    c2 = wq.claim("worker_bad", EXTERNAL, t2["ticket_id"])
    wq.report("worker_bad", EXTERNAL, t2["ticket_id"], c2["lease_token"], {})
    wq.reject("mgr", LOCAL, t2["ticket_id"], reason="no")
    agg2 = {r["provider"]: r for r in mem.provider_eval_aggregate()}
    assert agg2["worker_bad"]["samples"] >= 1
    assert agg2["worker_bad"]["successes"] == 0
