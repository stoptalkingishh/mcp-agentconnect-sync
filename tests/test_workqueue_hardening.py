"""Hardening regressions for the work queue: the SHARED-connection concurrency
path the broker actually uses, add()'s dependency privacy check, the opt-in
reaper thread, and delivery-error surfacing.

test_workqueue.py's race test deliberately gives each thread its OWN connection
(the multi-process path, serialized by SQLite's file lock). The broker instead
serves remote workers from a thread pool over ONE shared connection — so these
tests share a single WorkQueue across threads, which is where an unsynchronized
commit/rollback would corrupt a peer's in-flight claim.
"""

import threading
import time

from agentconnect.common.config import load_routing
from agentconnect.common.memory import SharedMemory
from agentconnect.common.schemas import PrivacyClass, WorkerResult
from agentconnect.common.workqueue import WorkQueue

LOCAL = "local_only"
EXTERNAL = "external"


def _wq():
    mem = SharedMemory()
    return WorkQueue(mem, load_routing()), mem


# ------------------------------------------------------ shared-connection races
def test_shared_connection_no_double_claim_under_contention():
    """Many threads, ONE WorkQueue/connection, fewer tickets than threads. Every
    ticket must be won by exactly one thread and incremented exactly once — no
    double-claim, no claim silently rolled back by a losing peer."""
    wq, _ = _wq()
    n_tickets = 12
    ids = [wq.add(privacy_class=PrivacyClass.public, payload=f"t{i}", origin="o")["ticket_id"]
           for i in range(n_tickets)]

    n_workers = 8
    barrier = threading.Barrier(n_workers)
    won: dict[str, list[str]] = {}
    lock = threading.Lock()

    def worker(name: str):
        barrier.wait()
        while True:
            got = wq.claim_next(name, LOCAL, max=1)
            if not got:
                break
            with lock:
                won.setdefault(got[0]["ticket_id"], []).append(name)

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one winner per ticket, and every ticket claimed.
    assert set(won) == set(ids), "some tickets never claimed or unknown id appeared"
    doubles = {tid: ws for tid, ws in won.items() if len(ws) != 1}
    assert not doubles, f"tickets claimed by more than one worker: {doubles}"
    for tid in ids:
        row = wq.get(tid)
        assert row["status"] == "claimed"
        assert row["attempts"] == 1, f"{tid} attempts={row['attempts']} (double-count)"


def test_shared_connection_claim_and_report_interleave():
    """Interleave claims (one thread) with reports (another) on the same
    connection; the store's writes (put_artifact commits) must not corrupt an
    in-flight claim. Assert every processed ticket ends terminal-and-consistent."""
    wq, _ = _wq()
    ids = [wq.add(privacy_class=PrivacyClass.public, payload=f"t{i}", origin="o")["ticket_id"]
           for i in range(20)]
    barrier = threading.Barrier(2)
    claimed: list[dict] = []
    clock = threading.Lock()

    def claimer():
        barrier.wait()
        while len(claimed) < len(ids):
            got = wq.claim_next("c", LOCAL, max=1)
            if got:
                with clock:
                    claimed.append(got[0])
            else:
                time.sleep(0.001)

    def reporter():
        barrier.wait()
        done = 0
        while done < len(ids):
            with clock:
                pending = claimed[done:]
            for t in pending:
                wq.report("c", LOCAL, t["ticket_id"], t["lease_token"],
                          WorkerResult(status="completed", summary="ok"))
                done += 1
            time.sleep(0.001)

    threads = [threading.Thread(target=claimer), threading.Thread(target=reporter)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for tid in ids:
        row = wq.get(tid)
        assert row["status"] == "done", f"{tid} ended {row['status']}"
        assert row["result_status"] == "approved"


# --------------------------------------------------- add() dependency privacy
def test_add_rejects_privacy_downgrade_dependency():
    """A public (widely-claimable) child depending on a repo_sensitive parent is a
    laundering path: add() must refuse it, exactly as link() does."""
    wq, _ = _wq()
    parent = wq.add(privacy_class=PrivacyClass.repo_sensitive, payload="secret", origin="o")
    child = wq.add(privacy_class=PrivacyClass.public, payload="pub", origin="o",
                   depends_on=[parent["ticket_id"]])
    assert child == {"error": "privacy_downgrade", "depends_on": parent["ticket_id"]}


def test_add_allows_monotonic_dependency():
    """A repo_sensitive child (narrower) depending on a public parent (wider) is
    fine — the child is at least as restrictive."""
    wq, _ = _wq()
    parent = wq.add(privacy_class=PrivacyClass.public, payload="pub", origin="o")
    child = wq.add(privacy_class=PrivacyClass.repo_sensitive, payload="secret", origin="o",
                   depends_on=[parent["ticket_id"]])
    assert "ticket_id" in child


# ------------------------------------------------------------- reaper thread
def test_start_reaper_requeues_expired_lease():
    """The opt-in daemon reaper requeues a ticket whose lease expired — the
    self-healing a manual-only reaper never provided."""
    wq, _ = _wq()
    t = wq.add(privacy_class=PrivacyClass.public, payload="x", origin="o")["ticket_id"]
    # Claim with a zero-length lease so it is already expired.
    got = wq.claim("w", LOCAL, t, lease_seconds=0)
    assert got.get("ticket_id") == t
    assert wq.get(t)["status"] == "claimed"

    thread, stop = wq.start_reaper(interval=0.02)
    try:
        deadline = time.time() + 2.0
        while time.time() < deadline and wq.get(t)["status"] != "open":
            time.sleep(0.02)
    finally:
        stop.set()
        thread.join(timeout=1.0)

    assert wq.get(t)["status"] == "open", "reaper thread did not requeue the expired lease"
