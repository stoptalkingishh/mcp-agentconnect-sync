"""Federated work-queue core (S1): the authorization boundary, the race-free
atomic claim, lease/reaper/fencing, idempotency, and the dependency gate.

All offline: in-memory (or temp-file) SharedMemory, live routing config, no
network/DB server. The authorization DENIAL cases are the headline — a worker of
a given attested tier may claim a ticket only if its tier is in the live
routing.yaml privacy.classes[privacy_class] set, fail-closed.
"""

import sqlite3
import threading

from agentconnect.common.config import load_routing
from agentconnect.common.memory import SharedMemory
from agentconnect.common.schemas import PrivacyClass
from agentconnect.common.workqueue import MAX_CLAIM_BATCH, WorkQueue

LOCAL = "local_only"
RENTED = "private_rented"
EXTERNAL = "external"
PAID = "external_paid"
ALL_TIERS = [LOCAL, RENTED, EXTERNAL, PAID]


def _wq(mem=None):
    mem = mem or SharedMemory()
    return WorkQueue(mem, load_routing()), mem


# --------------------------------------------------------------- authorization
def test_public_ticket_claimable_by_every_tier():
    for tier in ALL_TIERS:
        wq, _ = _wq()
        wq.add(privacy_class=PrivacyClass.public, payload="hi", origin="o")
        got = wq.claim_next("w", tier)
        assert len(got) == 1, f"{tier} should claim public"
        assert got[0]["attempts"] == 1


def test_repo_sensitive_denied_to_all_but_local():
    for tier in (RENTED, EXTERNAL, PAID):
        wq, _ = _wq()
        t = wq.add(privacy_class=PrivacyClass.repo_sensitive, payload="secret repo", origin="o")
        assert wq.claim_next("w", tier) == []
        err = wq.claim("w", tier, t["ticket_id"])
        assert err == {"error": "not_authorized"}
    # local_only may.
    wq, _ = _wq()
    t = wq.add(privacy_class=PrivacyClass.repo_sensitive, payload="secret repo", origin="o")
    assert wq.claim("w", LOCAL, t["ticket_id"]).get("ticket_id") == t["ticket_id"]


def test_restricted_denied_to_external():
    wq, _ = _wq()
    t = wq.add(privacy_class=PrivacyClass.restricted, payload="x", origin="o")
    assert wq.claim("w", EXTERNAL, t["ticket_id"]) == {"error": "not_authorized"}
    assert wq.claim("w", LOCAL, t["ticket_id"]).get("ticket_id") == t["ticket_id"]


def test_secret_sensitive_claimable_by_no_tier_and_parked():
    wq, _ = _wq()
    t = wq.add(privacy_class=PrivacyClass.secret_sensitive, payload="APIKEY", origin="o")
    assert t["park_reason"] == "secret_sensitive_not_pullable"
    row = wq.get(t["ticket_id"])
    assert row["status"] == "parked"
    for tier in ALL_TIERS:  # including local_only
        assert wq.claim_next("w", tier) == []
        assert wq.claim("w", tier, t["ticket_id"]) == {"error": "not_authorized"}


def test_unknown_identity_or_tier_is_denied_fail_closed():
    wq, _ = _wq()
    t = wq.add(privacy_class=PrivacyClass.public, payload="hi", origin="o")
    assert wq.claim_next("w", None) == []
    assert wq.claim_next("w", "bogus_tier") == []
    assert wq.claim("w", None, t["ticket_id"]) == {"error": "not_authorized"}
    assert wq.claim("w", "bogus_tier", t["ticket_id"]) == {"error": "not_authorized"}


def test_poisoned_allowed_tiers_column_cannot_widen_access():
    wq, mem = _wq()
    t = wq.add(privacy_class=PrivacyClass.repo_sensitive, payload="x", origin="o")
    # Attacker rewrites the denormalized cache to admit external.
    mem._conn.execute(
        "UPDATE work_queue SET allowed_tiers=? WHERE ticket_id=?",
        ('["external"]', t["ticket_id"]),
    )
    mem._conn.commit()
    # Claim recomputes from live config, so external is still denied.
    assert wq.claim("w", EXTERNAL, t["ticket_id"]) == {"error": "not_authorized"}


def test_attested_tier_is_the_only_authority_no_self_elevation():
    # The core only accepts the attested tier; there is no body-tier parameter to
    # elevate through. An external worker cannot reach repo_sensitive.
    wq, _ = _wq()
    t = wq.add(privacy_class=PrivacyClass.repo_sensitive, payload="x", origin="o")
    assert wq.claim("w", EXTERNAL, t["ticket_id"]) == {"error": "not_authorized"}


# ------------------------------------------------------------------- claim race
def test_concurrent_claim_exactly_one_winner(tmp_path):
    db = str(tmp_path / "race.db")
    seed, _ = _wq(SharedMemory(db))
    ticket = seed.add(privacy_class=PrivacyClass.public, payload="hi", origin="o")["ticket_id"]

    results: list[dict] = []
    barrier = threading.Barrier(2)

    def worker(name):
        wq = WorkQueue(SharedMemory(db), load_routing())  # separate connection
        barrier.wait()
        results.append(wq.claim(name, LOCAL, ticket))

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    wins = [r for r in results if "ticket_id" in r]
    losses = [r for r in results if "error" in r]
    assert len(wins) == 1
    assert len(losses) == 1 and losses[0] == {"error": "not_claimable"}
    final = WorkQueue(SharedMemory(db), load_routing()).get(ticket)
    assert final["status"] == "claimed"
    assert final["attempts"] == 1  # incremented exactly once


def test_claim_next_clamps_max_to_batch_ceiling():
    # SECURITY: an unbounded `max` would let one caller drain the whole open
    # backlog in a single lock-held call, starving same-tier peers -- the same
    # anti-starvation reasoning the HTTP /queue/next edge enforces via
    # MAX_CLAIM_BATCH. claim_next must clamp itself too, since it is also
    # reachable from mcp_server.queue_next (no HTTP-edge clamp there).
    wq, _ = _wq()
    for i in range(MAX_CLAIM_BATCH + 10):
        wq.add(privacy_class=PrivacyClass.public, payload=f"x{i}", origin="o")
    got = wq.claim_next("w", LOCAL, max=MAX_CLAIM_BATCH + 10_000)
    assert len(got) == MAX_CLAIM_BATCH


def test_claim_next_returns_partial_batch_on_mid_batch_operational_error(monkeypatch):
    # Regression: claim_next loops calling _try_claim per ticket, each with
    # its own commit. If a LATER iteration's commit hit a transient
    # sqlite3.OperationalError (e.g. SQLITE_BUSY under cross-process write
    # contention), the exception used to propagate out of claim_next entirely
    # -- discarding the tickets EARLIER iterations had already committed as
    # claimed (attempts++, lease minted). Those tickets would be stranded
    # until the reaper requeues them, because the HTTP caller (transport's
    # _guard) turns the exception into a bare 503 and never learns their
    # ticket_id/lease_token.
    wq, _ = _wq()
    for i in range(3):
        wq.add(privacy_class=PrivacyClass.public, payload=f"x{i}", origin="o")

    real_try_claim = wq._try_claim
    calls = {"n": 0}

    def flaky_try_claim(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise sqlite3.OperationalError("simulated contention")
        return real_try_claim(*args, **kwargs)

    monkeypatch.setattr(wq, "_try_claim", flaky_try_claim)
    got = wq.claim_next("w", LOCAL, max=3)
    # The first, already-committed claim is returned rather than lost.
    assert len(got) == 1
    stats = wq.stats()
    assert stats["by_status"]["claimed"] == 1
    assert stats["by_status"]["open"] == 2


# ---------------------------------------------------------------------- reaper
def test_reaper_requeues_when_attempts_remain():
    wq, _ = _wq()
    t = wq.add(privacy_class=PrivacyClass.public, payload="x", origin="o", max_attempts=3)
    wq.claim("w", LOCAL, t["ticket_id"], lease_seconds=10, now=1000.0)
    out = wq.reap_expired(now=2000.0)
    assert out == {"requeued": [t["ticket_id"]], "parked": []}
    assert wq.get(t["ticket_id"])["status"] == "open"
    assert wq.get(t["ticket_id"])["lease_token"] is None


def test_reaper_parks_when_attempts_exhausted():
    wq, _ = _wq()
    t = wq.add(privacy_class=PrivacyClass.public, payload="x", origin="o", max_attempts=1)
    wq.claim("w", LOCAL, t["ticket_id"], lease_seconds=10, now=1000.0)  # attempts -> 1
    out = wq.reap_expired(now=2000.0)
    assert out == {"requeued": [], "parked": [t["ticket_id"]]}
    assert wq.get(t["ticket_id"])["status"] == "parked"


def test_reaper_park_fails_linked_task_and_cascades_to_dependents():
    # Lease-expiry-driven attempts-exhaustion must NOT diverge from report/reject-
    # driven exhaustion: a parked (exhausted) parent can never reach 'done', so it
    # must drive its linked task to FAILED and cascade failure to every dependent,
    # rather than stranding the task non-terminal and its children blocked forever.
    from agentconnect.common.schemas import TaskState

    wq, mem = _wq()
    task_id = mem.create_task({"task": "parent work"})
    parent = wq.add(privacy_class=PrivacyClass.public, payload="p", origin="o",
                    task_id=task_id, max_attempts=1)
    child = wq.add(privacy_class=PrivacyClass.public, payload="c", origin="o",
                   depends_on=[parent["ticket_id"]])
    grandchild = wq.add(privacy_class=PrivacyClass.public, payload="g", origin="o",
                        depends_on=[child["ticket_id"]])
    wq.claim("w", LOCAL, parent["ticket_id"], lease_seconds=10, now=1000.0)  # attempts -> 1
    out = wq.reap_expired(now=2000.0)
    assert out == {"requeued": [], "parked": [parent["ticket_id"]]}
    assert wq.get(parent["ticket_id"])["status"] == "parked"
    # The linked task is driven terminal (mirrors report()/reject()).
    assert mem.get_task(task_id)["state"] == TaskState.FAILED.value
    # Failure cascades transitively to every dependent, so nothing is stranded.
    child_row = wq.get(child["ticket_id"])
    assert child_row["status"] == "failed"
    assert child_row["result_status"] == "dependency_failed"
    assert wq.get(grandchild["ticket_id"])["status"] == "failed"


# ----------------------------------------------------------- idempotency / lease
def test_double_report_is_refused():
    wq, _ = _wq()
    t = wq.add(privacy_class=PrivacyClass.public, payload="x", origin="o")
    c = wq.claim("w", EXTERNAL, t["ticket_id"])  # untrusted -> lands in_review
    first = wq.report("w", EXTERNAL, t["ticket_id"], c["lease_token"], {"summary": "done"})
    assert first["ticket_status"] == "in_review"
    second = wq.report("w", EXTERNAL, t["ticket_id"], c["lease_token"], {"summary": "again"})
    assert second == {"error": "already_reported"}


def test_stale_token_after_requeue_and_reclaim_is_lease_lost():
    wq, _ = _wq()
    t = wq.add(privacy_class=PrivacyClass.public, payload="x", origin="o", max_attempts=3)
    c1 = wq.claim("slow", LOCAL, t["ticket_id"], lease_seconds=10, now=1000.0)
    wq.reap_expired(now=2000.0)  # requeues, clears token
    c2 = wq.claim("fast", LOCAL, t["ticket_id"], now=2001.0)
    assert c2["lease_token"] != c1["lease_token"]
    # The resurrected slow worker reports with its stale token, same ticket.
    assert wq.report("slow", LOCAL, t["ticket_id"], c1["lease_token"], {}, now=2002.0) == {
        "error": "lease_lost"
    }


def test_report_trust_is_gated_on_claim_time_tier_not_report_time():
    # SECURITY: the verification gate ('only local_only auto-accepts') must key
    # off the tier that actually PERFORMED the work (stored at claim time as
    # lease_tier), never the tier resolved at report time. Otherwise an
    # identity promoted mid-lease (or a tier_resolver reload) launders an
    # untrusted-computed result straight to done/approved with no review.
    wq, _ = _wq()
    t = wq.add(privacy_class=PrivacyClass.public, payload="x", origin="o")
    c = wq.claim("w", EXTERNAL, t["ticket_id"])  # claimed under EXTERNAL
    assert wq.get(t["ticket_id"])["lease_tier"] == EXTERNAL
    # Same identity now (re-)resolves to LOCAL at report time — must NOT grant
    # trust for work actually done under EXTERNAL.
    out = wq.report("w", LOCAL, t["ticket_id"], c["lease_token"], {"status": "completed"})
    assert out["ticket_status"] == "in_review"
    assert out["result_status"] == "pending"
    assert wq.get(t["ticket_id"])["status"] == "in_review"
    # The genuine trusted path (claimed AND reported under LOCAL) still auto-accepts.
    t2 = wq.add(privacy_class=PrivacyClass.public, payload="y", origin="o")
    c2 = wq.claim("w", LOCAL, t2["ticket_id"])
    out2 = wq.report("w", LOCAL, t2["ticket_id"], c2["lease_token"], {"status": "completed"})
    assert out2["ticket_status"] == "done"
    assert out2["result_status"] == "approved"


def test_dedup_key_returns_existing_and_never_reopens_done():
    wq, _ = _wq()
    a = wq.add(privacy_class=PrivacyClass.public, payload="x", origin="o", dedup_key="job-1")
    b = wq.add(privacy_class=PrivacyClass.public, payload="x2", origin="o", dedup_key="job-1")
    assert a["ticket_id"] == b["ticket_id"]
    # Drive to done (trusted auto-accept), then re-add: still the same done ticket.
    c = wq.claim("w", LOCAL, a["ticket_id"])
    wq.report("w", LOCAL, a["ticket_id"], c["lease_token"], {"summary": "ok"})
    assert wq.get(a["ticket_id"])["status"] == "done"
    again = wq.add(privacy_class=PrivacyClass.public, payload="x3", origin="o", dedup_key="job-1")
    assert again["ticket_id"] == a["ticket_id"]
    assert again["status"] == "done"


def test_renew_refuses_non_holder():
    wq, _ = _wq()
    t = wq.add(privacy_class=PrivacyClass.public, payload="x", origin="o")
    c = wq.claim("w", LOCAL, t["ticket_id"], now=1000.0)
    assert wq.renew("w", t["ticket_id"], "wrong-token", now=1001.0) == {"error": "lease_lost"}
    ok = wq.renew("w", t["ticket_id"], c["lease_token"], extend_seconds=60, now=1001.0)
    assert ok["lease_expires_at"] == 1061.0


def test_renew_clamps_unbounded_extend_seconds():
    # SECURITY: an unbounded extend_seconds would let an authorized worker pin
    # a lease centuries into the future, permanently defeating the reaper for
    # that ticket (it would never look expired). renew() must clamp it itself
    # (not just at the HTTP edge), since it is also reachable from
    # mcp_server.queue_update.
    from agentconnect.common.workqueue import MAX_LEASE_EXTEND_SECONDS

    wq, _ = _wq()
    t = wq.add(privacy_class=PrivacyClass.public, payload="x", origin="o")
    c = wq.claim("w", LOCAL, t["ticket_id"], now=1000.0)
    ok = wq.renew("w", t["ticket_id"], c["lease_token"], extend_seconds=999_999_999_999, now=1001.0)
    assert ok["lease_expires_at"] == 1001.0 + MAX_LEASE_EXTEND_SECONDS


def test_renew_refuses_non_positive_extend_seconds():
    # A floor of 0 would set lease_expires_at==now, report "success", and hand
    # the reaper an immediately-reapable lease on its very next tick — a
    # caller-visible success that silently loses the lease. Must be refused.
    wq, _ = _wq()
    t = wq.add(privacy_class=PrivacyClass.public, payload="x", origin="o")
    c = wq.claim("w", LOCAL, t["ticket_id"], lease_seconds=100, now=1000.0)
    assert wq.renew("w", t["ticket_id"], c["lease_token"], extend_seconds=0, now=1001.0) == {
        "error": "invalid_extend_seconds"
    }
    assert wq.renew("w", t["ticket_id"], c["lease_token"], extend_seconds=-5, now=1001.0) == {
        "error": "invalid_extend_seconds"
    }
    # The original lease is untouched by the refused calls.
    assert wq.get(t["ticket_id"])["lease_expires_at"] == 1100.0


# ------------------------------------------------------- dependency + monotonicity
def test_dependency_gate_blocks_until_parent_done():
    wq, _ = _wq()
    parent = wq.add(privacy_class=PrivacyClass.public, payload="p", origin="o")
    child = wq.add(privacy_class=PrivacyClass.public, payload="c", origin="o",
                   depends_on=[parent["ticket_id"]])
    # Child not claimable while the parent is open.
    assert wq.claim("w", LOCAL, child["ticket_id"]) == {"error": "not_claimable"}
    assert wq.status(child["ticket_id"])[0]["status"] == "blocked"
    # Finish the parent.
    pc = wq.claim("w", LOCAL, parent["ticket_id"])
    wq.report("w", LOCAL, parent["ticket_id"], pc["lease_token"], {"summary": "ok"})
    # Now the child is claimable.
    got = wq.claim("w", LOCAL, child["ticket_id"])
    assert got.get("ticket_id") == child["ticket_id"]


def test_dependency_gate_requires_ALL_parents_done_not_just_one():
    # Crown-jewel: a child with MULTIPLE depends_on parents stays blocked until
    # EVERY parent is 'done' (AND-across-parents), never claimable as soon as one
    # finishes. A regression narrowing the claim gate's NOT EXISTS to the first
    # parent (e.g. a stray LIMIT 1) would let multi-input work start early.
    wq, _ = _wq()
    a = wq.add(privacy_class=PrivacyClass.public, payload="a", origin="o")
    b = wq.add(privacy_class=PrivacyClass.public, payload="b", origin="o")
    child = wq.add(privacy_class=PrivacyClass.public, payload="c", origin="o",
                   depends_on=[a["ticket_id"], b["ticket_id"]])
    # Complete ONLY parent A.
    ca = wq.claim("w", LOCAL, a["ticket_id"])
    wq.report("w", LOCAL, a["ticket_id"], ca["lease_token"], {"status": "completed"})
    assert wq.get(a["ticket_id"])["status"] == "done"
    # Child must still be blocked: B is not done yet.
    assert wq.claim("w", LOCAL, child["ticket_id"]) == {"error": "not_claimable"}
    assert wq.status(child["ticket_id"])[0]["status"] == "blocked"
    # Complete B; only now is the AND-gate satisfied.
    cb = wq.claim("w", LOCAL, b["ticket_id"])
    wq.report("w", LOCAL, b["ticket_id"], cb["lease_token"], {"status": "completed"})
    got = wq.claim("w", LOCAL, child["ticket_id"])
    assert got.get("ticket_id") == child["ticket_id"]


def test_dependency_gate_blocks_until_parent_reviewed_not_just_reported():
    # SECURITY crown-jewel: an untrusted (in_review) result must never itself be
    # the 'done' truth that unblocks a dependent. A parent claimed/reported by
    # an untrusted tier lands 'in_review' (not 'done'), so the child must stay
    # blocked until a local_only reviewer approves it.
    wq, _ = _wq()
    parent = wq.add(privacy_class=PrivacyClass.public, payload="p", origin="o")
    child = wq.add(privacy_class=PrivacyClass.public, payload="c", origin="o",
                   depends_on=[parent["ticket_id"]])
    pc = wq.claim("w", EXTERNAL, parent["ticket_id"])
    wq.report("w", EXTERNAL, parent["ticket_id"], pc["lease_token"], {"status": "completed"})
    assert wq.get(parent["ticket_id"])["status"] == "in_review"
    # The child must still be blocked, not claimable, while the parent merely
    # awaits review — 'in_review' must never satisfy the depends_on='done' gate.
    assert wq.claim("w", LOCAL, child["ticket_id"]) == {"error": "not_claimable"}
    assert wq.status(child["ticket_id"])[0]["status"] == "blocked"
    # Once a local_only reviewer approves, the parent becomes 'done' and the
    # child becomes claimable.
    out = wq.approve("rev", LOCAL, parent["ticket_id"])
    assert out["ticket_status"] == "done"
    got = wq.claim("w", LOCAL, child["ticket_id"])
    assert got.get("ticket_id") == child["ticket_id"]


def test_claim_response_never_exposes_task_id():
    # The internal task_id is a live handle into the raw, un-redacted submission;
    # a claimer must only ever see the redacted payload_ref (WORK_QUEUE.md).
    wq, mem = _wq()
    task_id = mem.create_task({"task": "raw secret body"})
    t = wq.add(privacy_class=PrivacyClass.public, payload="redacted", origin="o", task_id=task_id)
    got = wq.claim_next("w", LOCAL)
    assert len(got) == 1
    assert "task_id" not in got[0]
    assert "payload_ref" in got[0]
    # Targeted claim path too.
    t2 = wq.add(privacy_class=PrivacyClass.public, payload="redacted", origin="o", task_id=task_id)
    c = wq.claim("w", LOCAL, t2["ticket_id"])
    assert "task_id" not in c


def test_add_rejects_dangling_dependency():
    wq, _ = _wq()
    res = wq.add(privacy_class=PrivacyClass.public, payload="c", origin="o",
                 depends_on=["wq_does_not_exist"])
    assert res == {"error": "unknown_dependency", "depends_on": "wq_does_not_exist"}
    # And nothing was enqueued.
    assert wq.status(status="open") == []


def test_dangling_dependency_edge_blocks_claim_fail_closed():
    # Even if a dangling edge is forced into the store, the claim gate treats an
    # unsatisfiable (missing) parent as blocking, never as satisfied.
    wq, mem = _wq()
    child = wq.add(privacy_class=PrivacyClass.public, payload="c", origin="o")
    mem._conn.execute(
        "INSERT INTO work_queue_deps(ticket_id, depends_on) VALUES(?,?)",
        (child["ticket_id"], "wq_ghost_parent"),
    )
    mem._conn.commit()
    assert wq.claim("w", LOCAL, child["ticket_id"]) == {"error": "not_claimable"}
    assert wq.claim_next("w", LOCAL) == []
    assert wq.status(child["ticket_id"])[0]["status"] == "blocked"


def test_link_rejects_dependency_cycle():
    wq, _ = _wq()
    a = wq.add(privacy_class=PrivacyClass.public, payload="a", origin="o")
    b = wq.add(privacy_class=PrivacyClass.public, payload="b", origin="o")
    assert wq.link(a["ticket_id"], b["ticket_id"]) == {"ok": True}
    # The reverse edge would deadlock both tickets forever.
    assert wq.link(b["ticket_id"], a["ticket_id"]) == {"error": "dependency_cycle"}
    # A longer cycle is caught too: a->b->c, then c->a.
    c = wq.add(privacy_class=PrivacyClass.public, payload="c", origin="o")
    assert wq.link(b["ticket_id"], c["ticket_id"]) == {"ok": True}
    assert wq.link(c["ticket_id"], a["ticket_id"]) == {"error": "dependency_cycle"}


def test_low_sensitive_not_cloud_safe_is_parked():
    # low_sensitive requires redaction; an un-redacted (cloud_safe=False) ticket is
    # parked, never leasable — mirrors the router's cloud_safe gate.
    wq, _ = _wq()
    parked = wq.add(privacy_class=PrivacyClass.low_sensitive, payload="x",
                    origin="o", cloud_safe=False)
    assert parked["status"] == "parked"
    assert parked["park_reason"] == "redaction_required_not_cloud_safe"
    assert wq.claim("w", LOCAL, parked["ticket_id"]) == {"error": "not_claimable"}
    # The redacted (cloud_safe=True, the default) path stays open.
    ok = wq.add(privacy_class=PrivacyClass.low_sensitive, payload="x", origin="o", cloud_safe=True)
    assert ok["status"] == "open"


def test_concurrent_add_same_dedup_key_is_race_safe(tmp_path):
    # Two independent agents submitting the same idempotent unit of work must both
    # succeed with the SAME ticket — never a raw sqlite3.IntegrityError.
    db = str(tmp_path / "dedup.db")
    SharedMemory(db).close()  # ensure schema exists before the racers open it

    results: list[dict] = []
    barrier = threading.Barrier(2)

    def worker():
        wq = WorkQueue(SharedMemory(db), load_routing())  # separate connection
        barrier.wait()
        results.append(wq.add(task="x", origin="o", payload="x", dedup_key="same-key"))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert len(results) == 2
    assert all("ticket_id" in r for r in results), results
    assert results[0]["ticket_id"] == results[1]["ticket_id"]


def test_link_rejects_privacy_downgrade():
    wq, _ = _wq()
    parent = wq.add(privacy_class=PrivacyClass.repo_sensitive, payload="p", origin="o")
    child = wq.add(privacy_class=PrivacyClass.public, payload="c", origin="o")
    # A public child of a repo_sensitive parent would launder sensitive output down.
    assert wq.link(child["ticket_id"], parent["ticket_id"]) == {"error": "privacy_downgrade"}
    # The monotonic direction (more-restrictive child) is allowed.
    child2 = wq.add(privacy_class=PrivacyClass.repo_sensitive, payload="c2", origin="o")
    pub_parent = wq.add(privacy_class=PrivacyClass.public, payload="pp", origin="o")
    assert wq.link(child2["ticket_id"], pub_parent["ticket_id"]) == {"ok": True}


# ------------------------------------------------------- operator view (S3)
def test_list_tickets_is_payload_free_and_filterable():
    wq, _ = _wq()
    secret_text = "the raw task body that must never leak to an operator"
    t1 = wq.add(privacy_class=PrivacyClass.public, payload=secret_text, origin="o")
    t2 = wq.add(privacy_class=PrivacyClass.repo_sensitive, payload="other", origin="o")

    all_rows = wq.list_tickets()
    assert {r["ticket_id"] for r in all_rows} == {t1["ticket_id"], t2["ticket_id"]}
    blob = str(all_rows)
    assert secret_text not in blob
    assert all("task_id" not in r for r in all_rows)

    only_repo = wq.list_tickets(privacy_class="repo_sensitive")
    assert {r["ticket_id"] for r in only_repo} == {t2["ticket_id"]}

    only_open = wq.list_tickets(status="open")
    assert {r["ticket_id"] for r in only_open} == {t1["ticket_id"], t2["ticket_id"]}


def test_pending_review_lists_only_in_review_tickets():
    wq, _ = _wq()
    t = wq.add(privacy_class=PrivacyClass.public, payload="hi", origin="o")
    assert wq.pending_review() == []
    claim = wq.claim(EXTERNAL, EXTERNAL, t["ticket_id"])
    wq.report(EXTERNAL, EXTERNAL, t["ticket_id"], claim["lease_token"], {"status": "completed"})
    pending = wq.pending_review()
    assert len(pending) == 1 and pending[0]["ticket_id"] == t["ticket_id"]
    assert pending[0]["status"] == "in_review"


def test_terminally_failed_parent_cascades_failure_to_children():
    # A parent driven to terminal 'failed' can never satisfy the depends_on='done'
    # claim gate, so its children would be blocked forever. The failure must
    # cascade down the whole dependency subtree instead of stranding them.
    wq, _ = _wq()
    parent = wq.add(privacy_class=PrivacyClass.public, payload="p", origin="o", max_attempts=1)
    child = wq.add(privacy_class=PrivacyClass.public, payload="c", origin="o",
                   depends_on=[parent["ticket_id"]])
    grandchild = wq.add(privacy_class=PrivacyClass.public, payload="g", origin="o",
                        depends_on=[child["ticket_id"]])
    # Report a non-'completed' status with attempts exhausted -> parent terminal.
    c = wq.claim("w", LOCAL, parent["ticket_id"])
    out = wq.report("w", LOCAL, parent["ticket_id"], c["lease_token"], {"status": "failed"})
    assert out["ticket_status"] == "failed"
    assert wq.get(parent["ticket_id"])["status"] == "failed"
    # The entire subtree is terminally failed, not stuck open/blocked.
    assert wq.get(child["ticket_id"])["status"] == "failed"
    assert wq.get(child["ticket_id"])["result_status"] == "dependency_failed"
    assert wq.get(grandchild["ticket_id"])["status"] == "failed"
    # No longer claimable (was previously blocked forever).
    assert wq.claim("w", LOCAL, child["ticket_id"]) == {"error": "not_claimable"}


def test_cascade_failure_drives_linked_child_task_to_failed():
    wq, mem = _wq()
    parent = wq.add(privacy_class=PrivacyClass.public, payload="p", origin="o", max_attempts=1)
    child_task = mem.create_task({"task": "child work"})
    child = wq.add(privacy_class=PrivacyClass.public, payload="c", origin="o",
                   task_id=child_task, depends_on=[parent["ticket_id"]])
    c = wq.claim("w", LOCAL, parent["ticket_id"])
    wq.report("w", LOCAL, parent["ticket_id"], c["lease_token"], {"status": "failed"})
    assert wq.get(child["ticket_id"])["status"] == "failed"
    assert mem.get_task(child_task)["state"] == "FAILED"


def test_rejected_terminal_parent_cascades_failure_to_children():
    # The other terminal-failure seam: a reviewer rejecting an attempts-exhausted
    # parent must also cascade to dependents.
    wq, _ = _wq()
    parent = wq.add(privacy_class=PrivacyClass.public, payload="p", origin="o", max_attempts=1)
    child = wq.add(privacy_class=PrivacyClass.public, payload="c", origin="o",
                   depends_on=[parent["ticket_id"]])
    c = wq.claim("w", EXTERNAL, parent["ticket_id"])  # untrusted -> in_review
    wq.report("w", EXTERNAL, parent["ticket_id"], c["lease_token"], {"status": "completed"})
    out = wq.reject("rev", LOCAL, parent["ticket_id"], reason="bad")
    assert out["ticket_status"] == "failed"
    assert wq.get(child["ticket_id"])["status"] == "failed"


def test_reject_terminal_failure_clears_lease_tier():
    # reject()'s terminal ('failed') branch must clear lease_tier just like its
    # own requeue branch, report()'s clear_lease, and reap_expired()'s park --
    # otherwise a stale lease_tier lingers on a ticket that holds no live lease
    # and pollutes list_tickets/operator views.
    wq, _ = _wq()
    parent = wq.add(privacy_class=PrivacyClass.public, payload="p", origin="o", max_attempts=1)
    c = wq.claim("w", EXTERNAL, parent["ticket_id"])  # untrusted -> in_review
    wq.report("w", EXTERNAL, parent["ticket_id"], c["lease_token"], {"status": "completed"})
    assert wq.get(parent["ticket_id"])["lease_tier"] == EXTERNAL
    out = wq.reject("rev", LOCAL, parent["ticket_id"], reason="bad")
    assert out["ticket_status"] == "failed"  # attempts exhausted -> terminal, not requeued
    row = wq.get(parent["ticket_id"])
    assert row["lease_holder"] is None
    assert row["lease_token"] is None
    assert row["lease_tier"] is None


def test_payload_for_null_payload_ref_is_typed_error_not_empty_success():
    # A ticket added without a payload has a NULL payload_ref. payload_for must
    # return a typed error (never payload="") so the worker's null-payload safety
    # net fires instead of running an empty task and reporting a bogus success.
    wq, _ = _wq()
    t = wq.add(task="x", privacy_class=PrivacyClass.public, origin="o")  # no payload
    assert wq.get(t["ticket_id"])["payload_ref"] is None
    c = wq.claim("w", LOCAL, t["ticket_id"])
    res = wq.payload_for("w", t["ticket_id"], c["lease_token"], LOCAL)
    assert res == {"error": "no_payload"}
    # An empty-STRING payload is a real artifact and still delivers "".
    t2 = wq.add(task="y", privacy_class=PrivacyClass.public, payload="", origin="o")
    c2 = wq.claim("w", LOCAL, t2["ticket_id"])
    res2 = wq.payload_for("w", t2["ticket_id"], c2["lease_token"], LOCAL)
    assert res2.get("payload") == "" and "error" not in res2


def test_payload_for_refuses_downgraded_tier_and_lost_lease():
    # Crown-jewel delivery seam: the redacted body reaches ONLY the current lease
    # holder whose attested tier still admits the class.
    wq, _ = _wq()
    t = wq.add(privacy_class=PrivacyClass.repo_sensitive, payload="secret body", origin="o")
    c = wq.claim("w", LOCAL, t["ticket_id"], now=1000.0)
    # Happy path: holder + valid tier gets the body.
    ok = wq.payload_for("w", t["ticket_id"], c["lease_token"], LOCAL, now=1001.0)
    assert ok["payload"] == "secret body"
    # Tier downgraded after claim -> re-check refuses (belt-and-suspenders).
    assert wq.payload_for("w", t["ticket_id"], c["lease_token"], EXTERNAL, now=1001.0) == {
        "error": "not_authorized"
    }
    # Wrong identity, stale token, expired lease all -> lease_lost.
    assert wq.payload_for("intruder", t["ticket_id"], c["lease_token"], LOCAL, now=1001.0) == {
        "error": "lease_lost"
    }
    assert wq.payload_for("w", t["ticket_id"], "stale-token", LOCAL, now=1001.0) == {
        "error": "lease_lost"
    }
    assert wq.payload_for("w", t["ticket_id"], c["lease_token"], LOCAL, now=1e12) == {
        "error": "lease_lost"
    }


def test_approve_is_compare_and_set_second_action_refused():
    wq, _ = _wq()
    t = wq.add(privacy_class=PrivacyClass.public, payload="x", origin="o")
    c = wq.claim("w", EXTERNAL, t["ticket_id"])
    wq.report("w", EXTERNAL, t["ticket_id"], c["lease_token"], {"status": "completed"})
    assert wq.approve("rev", LOCAL, t["ticket_id"])["ticket_status"] == "done"
    # A second operator action cannot flip a done ticket back (guarded WHERE).
    assert wq.approve("rev", LOCAL, t["ticket_id"]) == {"error": "not_in_review"}
    assert wq.reject("rev", LOCAL, t["ticket_id"]) == {"error": "not_in_review"}
    assert wq.get(t["ticket_id"])["status"] == "done"


def test_link_rejects_claimed_or_in_review_child():
    # SECURITY: retroactively attaching a dependency to an actively-worked
    # child is unsafe — if the new parent later fails terminally,
    # _cascade_failure would force the child to 'failed' and clear its live
    # lease out from under a worker, silently discarding a legitimate result.
    wq, _ = _wq()
    child = wq.add(privacy_class=PrivacyClass.public, payload="c", origin="o")
    parent = wq.add(privacy_class=PrivacyClass.public, payload="p", origin="o", max_attempts=1)
    c = wq.claim("w", LOCAL, child["ticket_id"])
    assert wq.link(child["ticket_id"], parent["ticket_id"]) == {"error": "child_not_linkable"}
    # Still claimed, lease intact — the worker's in-flight lease was not touched.
    row = wq.get(child["ticket_id"])
    assert row["status"] == "claimed"
    assert row["lease_token"] == c["lease_token"]
    # Report under an untrusted tier -> in_review; still not linkable.
    child2 = wq.add(privacy_class=PrivacyClass.public, payload="c2", origin="o")
    c2 = wq.claim("w", EXTERNAL, child2["ticket_id"])
    wq.report("w", EXTERNAL, child2["ticket_id"], c2["lease_token"], {"status": "completed"})
    assert wq.get(child2["ticket_id"])["status"] == "in_review"
    assert wq.link(child2["ticket_id"], parent["ticket_id"]) == {"error": "child_not_linkable"}
    # An open child is linkable as before.
    child3 = wq.add(privacy_class=PrivacyClass.public, payload="c3", origin="o")
    assert wq.link(child3["ticket_id"], parent["ticket_id"]) == {"ok": True}


def test_link_atomic_guard_blocks_race_past_stale_precheck(monkeypatch):
    # Regression: the child-status guard must be enforced in the SAME atomic
    # statement as the edge INSERT, not only in the Python-level pre-check --
    # otherwise a concurrent claim() landing between the pre-check read and
    # the INSERT could attach a dependency to an already-claimed child (the
    # exact cascade-failure data-loss scenario the guard exists to prevent).
    # Simulate that race by feeding link() a stale "open" read for the child
    # while the ticket is ACTUALLY claimed in the store.
    wq, _ = _wq()
    child = wq.add(privacy_class=PrivacyClass.public, payload="c", origin="o")
    parent = wq.add(privacy_class=PrivacyClass.public, payload="p", origin="o")
    wq.claim("w", LOCAL, child["ticket_id"])  # really claimed in the DB

    real_raw = wq._raw
    calls = {"n": 0}

    def stale_raw(ticket_id):
        calls["n"] += 1
        if ticket_id == child["ticket_id"] and calls["n"] == 1:
            row = dict(real_raw(ticket_id))
            row["status"] = "open"  # stale/racy pre-check read
            return row
        return real_raw(ticket_id)

    monkeypatch.setattr(wq, "_raw", stale_raw)
    # The Python pre-check sees the stale "open" and would proceed, but the
    # atomic INSERT guard must still refuse against the true DB state.
    assert wq.link(child["ticket_id"], parent["ticket_id"]) == {"error": "child_not_linkable"}
    # No edge was actually attached.
    assert wq._conn.execute(
        "SELECT 1 FROM work_queue_deps WHERE ticket_id=? AND depends_on=?",
        (child["ticket_id"], parent["ticket_id"]),
    ).fetchone() is None


def test_link_duplicate_edge_is_idempotent_ok():
    wq, _ = _wq()
    a = wq.add(privacy_class=PrivacyClass.public, payload="a", origin="o")
    b = wq.add(privacy_class=PrivacyClass.public, payload="b", origin="o")
    assert wq.link(a["ticket_id"], b["ticket_id"]) == {"ok": True}
    # Re-linking the same edge is a no-op success, not a spurious cycle error.
    assert wq.link(a["ticket_id"], b["ticket_id"]) == {"ok": True}


def test_stats_counts_by_status_and_privacy_class():
    wq, _ = _wq()
    wq.add(privacy_class=PrivacyClass.public, payload="a", origin="o")
    wq.add(privacy_class=PrivacyClass.repo_sensitive, payload="b", origin="o")
    stats = wq.stats()
    assert stats["by_status"]["open"] == 2
    assert stats["by_privacy_class"]["public"] == 1
    assert stats["by_privacy_class"]["repo_sensitive"] == 1
    assert "capability_requirements" in stats
