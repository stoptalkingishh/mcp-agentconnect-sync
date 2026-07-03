"""Federated work-queue core (S1): the authorization boundary, the race-free
atomic claim, lease/reaper/fencing, idempotency, and the dependency gate.

All offline: in-memory (or temp-file) SharedMemory, live routing config, no
network/DB server. The authorization DENIAL cases are the headline — a worker of
a given attested tier may claim a ticket only if its tier is in the live
routing.yaml privacy.classes[privacy_class] set, fail-closed.
"""

import threading

from agentconnect.common.config import load_routing
from agentconnect.common.memory import SharedMemory
from agentconnect.common.schemas import PrivacyClass
from agentconnect.common.workqueue import WorkQueue

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
