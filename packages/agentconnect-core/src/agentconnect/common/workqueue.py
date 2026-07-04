"""Federated pull-based work-queue with a trust x privacy authorization boundary.

This is the security+concurrency core of the open, pull-based, federated work
surface: separate agents and a friend's untrusted compute can discover, claim,
and complete work off a shared surface while AgentConnect's privacy guarantees
hold. It is queue + lease semantics layered onto the EXISTING SQLite store
(``SharedMemory._conn``) — no new server, no Postgres/Redis. Queue state, task
state, artifacts and audit stay in ONE transactional store.

The differentiator — the whole point — is which participant may SEE and CLAIM
which work, and whether we TRUST the result they return. Invariants the code
below enforces, stated here because they are the security contract:

Authorization (fail-closed). A worker's tier is its ATTESTED
:class:`ProviderPrivacyTier`, resolved from an authenticated identity by the
caller (MCP/HTTP) — NEVER a tier in the request body. A worker of tier ``T`` may
see/claim a ticket of class ``P`` iff ``T`` is in routing.yaml
``privacy.classes[P]`` — the identical mapping the router uses to decide which
provider tiers may receive a class. This set is recomputed LIVE from
:class:`RoutingConfig` on every claim, so the denormalized ``allowed_tiers``
column can never widen access even if poisoned. Consequences from the real
config: ``repo_sensitive``/``restricted`` -> only ``local_only``;
``secret_sensitive`` -> NO tier at all (such tickets are stored ``parked`` and
are never leasable); an un-redacted ``low_sensitive`` ticket is likewise parked.
An unmapped/empty tier yields an empty admissible-class set -> claims nothing.

Concurrency. The claim is a single guarded ``UPDATE ... WHERE status='open'``,
so exactly one poller wins one ticket (SQLite's answer to ``SELECT ... FOR
UPDATE SKIP LOCKED``). ``attempts`` is incremented at claim, so lease expiry
never double-counts. Two isolation regimes both hold: ACROSS PROCESSES separate
connections are serialized by SQLite's file lock (with ``busy_timeout``/WAL set
in :class:`SharedMemory`); WITHIN a process every writer holds the store's
reentrant lock (``_synchronized``) across its whole execute→commit span, because
a shared connection has ONE transaction and an unsynchronized peer's
commit/rollback would otherwise act on this claim's in-flight ``UPDATE``. The
broker serves remote workers from a thread pool over one connection, so this
in-process lock — not SQLite's file lock — is what makes that path race-free.

Leasing / fencing. Each claim mints a fresh ``lease_token``; report/heartbeat
require ``status='claimed' AND lease_holder=? AND lease_token=? AND
lease_expires_at>now``. After a reaper requeue + re-claim the token is
regenerated, so a resurrected slow worker's report is refused even under the
same identity. The reaper requeues expired leases with attempts remaining and
parks the rest; it runs on an explicit periodic call by default (keeps the
offline gate deterministic), with ``start_reaper`` as an opt-in daemon thread
for a long-lived broker that needs self-healing without an external scheduler.

Verification gate. A report from an untrusted tier can NEVER silently become
truth: only ``local_only`` auto-accepts (``done``); every other tier lands
``in_review``/``pending`` until a ``local_only`` reviewer approves.

Idempotency. ``dedup_key`` is partial-unique, so re-adding done work returns the
existing ticket and never reopens it; reporting refuses a non-``claimed`` ticket;
terminal is terminal.

Dependencies. A ticket is claimable only when every ``depends_on`` parent is
``done`` (checked in the claim's ``NOT EXISTS`` guard). ``link`` additionally
enforces privacy monotonicity: a child's admissible-tier set must be a subset of
every parent's, so sensitive output cannot be laundered down a dependency edge.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Any, Iterable, Optional, Union

from .config import RoutingConfig
from .memory import SharedMemory, _new_id, _now, _synchronized
from .schemas import PrivacyClass, ProviderPrivacyTier, TaskState, WorkerResult

_TRUSTED_TIER = ProviderPrivacyTier.local_only.value

# Terminal / non-reopenable ticket statuses.
_TERMINAL = {"done", "failed"}


def _tier_value(tier: Union[str, ProviderPrivacyTier, None]) -> Optional[str]:
    if tier is None:
        return None
    return tier.value if isinstance(tier, ProviderPrivacyTier) else str(tier)


def _class_value(pc: Union[str, PrivacyClass]) -> str:
    return pc.value if isinstance(pc, PrivacyClass) else str(pc)


def _priority_rank_sql(col: str = "priority") -> str:
    return f"CASE {col} WHEN 'urgent' THEN 0 WHEN 'normal' THEN 1 WHEN 'low' THEN 2 ELSE 1 END"


class WorkQueue:
    """Queue + lease + authorization over ``memory._conn`` (one shared store)."""

    def __init__(self, memory: SharedMemory, routing: RoutingConfig, default_lease_seconds: int = 120):
        self.memory = memory
        self.routing = routing
        self.default_lease_seconds = default_lease_seconds
        self._conn = memory._conn
        # Bind the STORE's lock (not a fresh one): claim/report/approve must
        # serialize against the store's own writes (create_task, put_artifact)
        # on the shared connection, not merely against each other.
        self._lock = memory._lock

    # ---------------------------------------------------------- authorization
    def _classes_map(self) -> dict[str, list[str]]:
        return self.routing.privacy.get("classes", {}) or {}

    def allowed_tiers(self, privacy_class: Union[str, PrivacyClass]) -> list[str]:
        """Tiers that may claim this class — recomputed LIVE from routing config.

        Deliberately NOT widened by any task opt-in (e.g. allow_rented). A pull
        worker's capability is its attested tier alone; fail-closed."""
        return list(self._classes_map().get(_class_value(privacy_class), []))

    def _admissible_classes(self, attested_tier: Union[str, ProviderPrivacyTier, None]) -> list[str]:
        """Every privacy_class this attested tier is authorized to claim."""
        tier = _tier_value(attested_tier)
        if tier is None:
            return []
        return [pc for pc, tiers in self._classes_map().items() if tier in (tiers or [])]

    def may_claim(
        self,
        attested_tier: Union[str, ProviderPrivacyTier, None],
        privacy_class: Union[str, PrivacyClass],
    ) -> bool:
        tier = _tier_value(attested_tier)
        if tier is None:
            return False
        return tier in self.allowed_tiers(privacy_class)

    # ------------------------------------------------------------------- add
    @_synchronized
    def add(
        self,
        *,
        task: str = "",
        origin: str = "unknown",
        privacy_class: Union[str, PrivacyClass] = PrivacyClass.public,
        payload: Optional[str] = None,
        payload_ref: Optional[str] = None,
        task_id: Optional[str] = None,
        required_capabilities: Optional[Iterable[str]] = None,
        priority: str = "normal",
        dedup_key: Optional[str] = None,
        depends_on: Optional[Iterable[str]] = None,
        max_attempts: int = 3,
        assignee: Optional[str] = None,
        cloud_safe: bool = True,
        now: Optional[float] = None,
    ) -> dict[str, Any]:
        """Enqueue a ticket. Idempotent on ``dedup_key``: an existing key returns
        the existing ticket unchanged (a done ticket is never reopened).

        secret_sensitive (and any class admitting no tier) is stored ``parked``
        and is never leasable. An un-redacted low_sensitive ticket is parked too
        (mirrors the router's cloud_safe gate).

        A ``depends_on`` that names a non-existent parent is rejected with a typed
        ``unknown_dependency`` error (mirrors ``link``): a dangling edge must never
        silently pass the claim gate. Idempotent re-add on ``dedup_key`` is
        race-safe — a concurrent writer that loses the unique-index race receives
        the winner's existing ticket, never a raw IntegrityError."""
        now = _now() if now is None else now
        pc = _class_value(privacy_class)

        if dedup_key is not None:
            existing = self._conn.execute(
                "SELECT * FROM work_queue WHERE dedup_key=?", (dedup_key,)
            ).fetchone()
            if existing is not None:
                return self._public_row(existing)

        # Fail-closed on dangling dependencies: a parent that does not exist would
        # make the NOT EXISTS claim gate vacuously true (child instantly
        # claimable). Reject up front, before any artifact/row is written. Apply
        # the SAME privacy-monotonicity check as link(): an edge created here at
        # enqueue must not be a laundering path that link() would refuse — the
        # child's admissible-tier set must be a subset of every parent's, or the
        # dependency could pull sensitive output down to a lower class.
        dep_list = list(depends_on or [])
        child_tiers = set(self.allowed_tiers(pc))
        for dep in dep_list:
            parent = self._raw(dep)
            if parent is None:
                return {"error": "unknown_dependency", "depends_on": dep}
            if not child_tiers.issubset(set(self.allowed_tiers(parent["privacy_class"]))):
                return {"error": "privacy_downgrade", "depends_on": dep}

        if payload_ref is None and payload is not None:
            payload_ref = self.memory.put_artifact(task_id, "work_payload", payload)

        tiers = self.allowed_tiers(pc)
        require_redaction = set(self.routing.privacy.get("require_redaction", []) or [])
        status = "open"
        park_reason: Optional[str] = None
        if not tiers:
            status, park_reason = "parked", "secret_sensitive_not_pullable"
        elif pc in require_redaction and not cloud_safe:
            status, park_reason = "parked", "redaction_required_not_cloud_safe"

        ticket_id = _new_id("wq")
        req_caps = list(required_capabilities or [])
        provenance = [{"event": "enqueue", "origin": origin, "ts": now}]
        if park_reason:
            provenance.append({"event": "park", "reason": park_reason, "ts": now})

        try:
            self._conn.execute(
                "INSERT INTO work_queue(ticket_id, dedup_key, origin, task_id, payload_ref,"
                " privacy_class, allowed_tiers, required_capabilities, priority, status,"
                " assignee, attempts, max_attempts, result_status, provenance, park_reason,"
                " created_at, updated_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ticket_id, dedup_key, origin, task_id, payload_ref, pc,
                    json.dumps(tiers), json.dumps(req_caps), priority, status,
                    assignee, 0, max_attempts, None, json.dumps(provenance), park_reason,
                    now, now,
                ),
            )
            for dep in dep_list:
                self._conn.execute(
                    "INSERT OR IGNORE INTO work_queue_deps(ticket_id, depends_on) VALUES(?,?)",
                    (ticket_id, dep),
                )
            self._conn.commit()
        except sqlite3.IntegrityError:
            # Idempotency race: a concurrent add() with the same dedup_key committed
            # first, tripping the partial-unique index. Return the winner's ticket
            # rather than leaking a raw exception (repo convention: typed results).
            self._conn.rollback()
            if dedup_key is not None:
                existing = self._conn.execute(
                    "SELECT * FROM work_queue WHERE dedup_key=?", (dedup_key,)
                ).fetchone()
                if existing is not None:
                    return self._public_row(existing)
            return {"error": "duplicate_ticket"}
        return self._public_row(self._raw(ticket_id))

    # ---------------------------------------------------------------- claim
    @_synchronized
    def claim_next(
        self,
        identity: str,
        attested_tier: Union[str, ProviderPrivacyTier, None],
        capabilities: Optional[Iterable[str]] = None,
        max: int = 1,
        lease_seconds: Optional[int] = None,
        now: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Atomically claim up to ``max`` authorized, unblocked tickets for a
        worker. Returns worker-visible rows (payload_ref + lease_token); the
        internal task_id/submission is never handed out. Empty when nothing is
        authorized/available."""
        now = _now() if now is None else now
        admissible = self._admissible_classes(attested_tier)
        if not admissible:
            return []
        worker_caps = set(capabilities or [])
        claimed: list[dict[str, Any]] = []
        ph = ",".join("?" for _ in admissible)
        while len(claimed) < max:
            rows = self._conn.execute(
                f"SELECT ticket_id, required_capabilities, provenance FROM work_queue"
                f" WHERE status='open' AND privacy_class IN ({ph})"
                f" ORDER BY {_priority_rank_sql()}, created_at",
                admissible,
            ).fetchall()
            won = None
            for row in rows:
                req = set(json.loads(row["required_capabilities"] or "[]"))
                if not req.issubset(worker_caps):
                    continue
                got = self._try_claim(
                    row["ticket_id"], identity, attested_tier, admissible,
                    row["provenance"], lease_seconds, now,
                )
                if got is not None:
                    won = got
                    break
            if won is None:
                break
            claimed.append(won)
        return claimed

    @_synchronized
    def claim(
        self,
        identity: str,
        attested_tier: Union[str, ProviderPrivacyTier, None],
        ticket_id: str,
        lease_seconds: Optional[int] = None,
        now: Optional[float] = None,
    ) -> dict[str, Any]:
        """Targeted claim of one ticket. Fail-closed: unknown ticket, wrong
        privacy class for the tier, unsatisfied deps, or a lost race all return a
        typed error and never a payload."""
        now = _now() if now is None else now
        admissible = self._admissible_classes(attested_tier)
        raw = self._raw(ticket_id)
        if raw is None:
            return {"error": "unknown_ticket"}
        if raw["privacy_class"] not in admissible:
            return {"error": "not_authorized"}
        got = self._try_claim(
            ticket_id, identity, attested_tier, admissible, raw["provenance"], lease_seconds, now
        )
        if got is None:
            return {"error": "not_claimable"}
        return got

    def _try_claim(
        self,
        ticket_id: str,
        identity: str,
        attested_tier: Union[str, ProviderPrivacyTier, None],
        admissible: list[str],
        provenance_json: Optional[str],
        lease_seconds: Optional[int],
        now: float,
    ) -> Optional[dict[str, Any]]:
        """The race-free heart: a single guarded UPDATE. Recomputes the privacy
        gate in the WHERE clause and gates on unsatisfied deps, so a poisoned
        column or a concurrent winner cannot slip a ticket through. rowcount==1
        means we won."""
        lease_seconds = self.default_lease_seconds if lease_seconds is None else lease_seconds
        tier = _tier_value(attested_tier)
        token = uuid.uuid4().hex
        expires = now + lease_seconds
        prov = json.loads(provenance_json or "[]")
        prov.append({"event": "claim", "identity": identity, "tier": tier, "ts": now})
        ph = ",".join("?" for _ in admissible)
        cur = self._conn.execute(
            f"UPDATE work_queue"
            f"   SET status='claimed', lease_holder=?, lease_tier=?, lease_token=?,"
            f"       lease_expires_at=?, attempts=attempts+1, claimed_at=?, updated_at=?,"
            f"       provenance=?"
            f" WHERE ticket_id=? AND status='open' AND privacy_class IN ({ph})"
            f"   AND NOT EXISTS (SELECT 1 FROM work_queue_deps d"
            f"                    WHERE d.ticket_id=?"
            f"                      AND NOT EXISTS (SELECT 1 FROM work_queue p"
            f"                                       WHERE p.ticket_id=d.depends_on"
            f"                                         AND p.status='done'))",
            (
                identity, tier, token, expires, now, now, json.dumps(prov),
                ticket_id, *admissible, ticket_id,
            ),
        )
        if cur.rowcount != 1:
            self._conn.rollback()
            return None
        self._conn.commit()
        raw = self._raw(ticket_id)
        # Worker-visible row ONLY: the internal task_id is a live handle into the
        # raw, un-redacted submission and must never cross to a claimer — the
        # worker gets the redacted payload_ref alone (WORK_QUEUE.md invariant).
        return {
            "ticket_id": ticket_id,
            "payload_ref": raw["payload_ref"],
            "privacy_class": raw["privacy_class"],
            "lease_token": token,
            "lease_expires_at": expires,
            "attempts": raw["attempts"],
            "status": raw["status"],
        }

    # -------------------------------------------------------------- heartbeat
    @_synchronized
    def renew(
        self,
        identity: str,
        ticket_id: str,
        lease_token: str,
        extend_seconds: Optional[int] = None,
        now: Optional[float] = None,
    ) -> dict[str, Any]:
        now = _now() if now is None else now
        extend = self.default_lease_seconds if extend_seconds is None else extend_seconds
        expires = now + extend
        cur = self._conn.execute(
            "UPDATE work_queue SET lease_expires_at=?, updated_at=?"
            " WHERE ticket_id=? AND status='claimed' AND lease_holder=?"
            "   AND lease_token=? AND lease_expires_at>?",
            (expires, now, ticket_id, identity, lease_token, now),
        )
        if cur.rowcount != 1:
            self._conn.rollback()
            return {"error": "lease_lost"}
        self._conn.commit()
        return {"status": "claimed", "lease_expires_at": expires}

    # ---------------------------------------------------------------- payload
    def _read_artifact_full(self, artifact_id: str) -> str:
        """Reassemble an artifact's full content from bounded chunks."""
        parts: list[str] = []
        offset = 0
        while True:
            chunk = self.memory.read_artifact_chunk(artifact_id, offset, 65536)
            if chunk is None:
                break
            parts.append(chunk.content)
            if chunk.next_offset is None:
                break
            offset = chunk.next_offset
        return "".join(parts)

    def payload_for(
        self,
        identity: str,
        ticket_id: str,
        lease_token: str,
        attested_tier: Union[str, ProviderPrivacyTier, None] = None,
        now: Optional[float] = None,
    ) -> dict[str, Any]:
        """Return the (redacted) task body to the CURRENT lease holder.

        This is the deliberate, authorized delivery seam: the internal task_id
        and un-redacted submission never cross to a claimer, but the redacted
        ``work_payload`` — the thing the worker must actually run — does, and
        only to the identity holding the live lease whose attested tier still
        admits the ticket's class. Lease-gated (mirrors ``report``/``renew``:
        the token must match a non-expired claim held by ``identity``) and
        authorization-gated (belt-and-suspenders re-check of ``may_claim`` so a
        tier that was downgraded after the claim is refused). Anyone else gets a
        typed error, never the body."""
        now = _now() if now is None else now
        raw = self._raw(ticket_id)
        if raw is None:
            return {"error": "unknown_ticket"}
        if (
            raw["status"] != "claimed"
            or raw["lease_holder"] != identity
            or raw["lease_token"] != lease_token
            or (raw["lease_expires_at"] or 0) <= now
        ):
            return {"error": "lease_lost"}
        if attested_tier is not None and not self.may_claim(attested_tier, raw["privacy_class"]):
            return {"error": "not_authorized"}
        ref = raw["payload_ref"]
        payload = self._read_artifact_full(ref) if ref else ""
        return {"ticket_id": ticket_id, "payload": payload, "payload_ref": ref}

    # ----------------------------------------------------------------- report
    @_synchronized
    def report(
        self,
        identity: str,
        attested_tier: Union[str, ProviderPrivacyTier, None],
        ticket_id: str,
        lease_token: str,
        result: Union[WorkerResult, dict[str, Any]],
        now: Optional[float] = None,
    ) -> dict[str, Any]:
        """Report a result under the fencing token. Idempotent: a second report,
        or a stale token after reaper-requeue-and-reclaim, is refused. Untrusted
        tiers land ``in_review`` (never silently truth)."""
        now = _now() if now is None else now
        raw = self._raw(ticket_id)
        if raw is None:
            return {"error": "unknown_ticket"}
        if raw["status"] != "claimed":
            if raw["status"] in ("in_review", "done", "failed"):
                return {"error": "already_reported"}
            return {"error": "not_claimable"}
        if (
            raw["lease_holder"] != identity
            or raw["lease_token"] != lease_token
            or (raw["lease_expires_at"] or 0) <= now
        ):
            return {"error": "lease_lost"}

        if isinstance(result, WorkerResult):
            wr = result
        else:
            wr = WorkerResult(**(result or {}))
        result_ref = self.memory.put_artifact(raw["task_id"], "work_result", wr.model_dump_json())

        tier = _tier_value(attested_tier)
        trusted = tier == _TRUSTED_TIER
        # A worker signals success with status=='completed' (the codebase's
        # canonical marker; see RouterService._run_agentic). ANY other status is a
        # reported failure and must NEVER become truth — not even on the trusted
        # fast-path, which previously promoted a failure straight to done/approved.
        succeeded = wr.status == "completed"
        prov = json.loads(raw["provenance"] or "[]")
        prov.append({"event": "report", "identity": identity, "tier": tier,
                     "status": wr.status, "ts": now})

        clear_lease = False
        if not succeeded:
            # Requeue for another attempt while attempts remain (mirrors reject /
            # the reaper); otherwise fail terminally. Either way, drop the lease.
            clear_lease = True
            if raw["attempts"] < raw["max_attempts"]:
                new_status, result_status, completed_at = "open", "failed", None
            else:
                new_status, result_status, completed_at = "failed", "failed", now
        elif trusted:
            new_status, result_status, completed_at = "done", "approved", now
        else:
            new_status, result_status, completed_at = "in_review", "pending", None

        # Atomic fence: put the lease guard IN the WHERE clause (as claim/renew do)
        # so validation and write are one operation. A commit happened between the
        # prior read and here (put_artifact), so an unguarded write could clobber a
        # concurrent re-claim after a reaper requeue.
        set_cols = ["status=?", "result_ref=?", "result_status=?", "completed_at=?",
                    "provenance=?", "updated_at=?"]
        params: list[Any] = [new_status, result_ref, result_status, completed_at,
                              json.dumps(prov), now]
        if clear_lease:
            set_cols += ["lease_holder=NULL", "lease_tier=NULL", "lease_token=NULL",
                         "lease_expires_at=NULL"]
        cur = self._conn.execute(
            "UPDATE work_queue SET " + ", ".join(set_cols) +
            " WHERE ticket_id=? AND status='claimed' AND lease_holder=?"
            "   AND lease_token=? AND lease_expires_at>?",
            (*params, ticket_id, identity, lease_token, now),
        )
        if cur.rowcount != 1:
            self._conn.rollback()
            return {"error": "lease_lost"}
        self._conn.commit()

        eval_status = "failed" if not succeeded else ("completed" if trusted else "pending_review")
        self.memory.record_evaluation({
            "provider": identity,
            "task_id": raw["task_id"],
            "status": eval_status,
            "confidence": wr.confidence,
        })
        if raw["task_id"]:
            if not succeeded:
                # Only a terminal failure drives the task to FAILED; a requeue
                # leaves the task state to be resolved by the next report.
                if new_status == "failed":
                    self._set_task_state(raw["task_id"], TaskState.FAILED)
            else:
                self._set_task_state(raw["task_id"],
                                     TaskState.COMPLETE if trusted else TaskState.REVIEW_READY)
        return {"ticket_status": new_status, "result_status": result_status, "result_ref": result_ref}

    # ---------------------------------------------------------- review gate
    @_synchronized
    def approve(
        self,
        reviewer_id: str,
        reviewer_tier: Union[str, ProviderPrivacyTier, None],
        ticket_id: str,
        now: Optional[float] = None,
    ) -> dict[str, Any]:
        now = _now() if now is None else now
        if _tier_value(reviewer_tier) != _TRUSTED_TIER:
            return {"error": "reviewer_not_authorized"}
        raw = self._raw(ticket_id)
        if raw is None:
            return {"error": "unknown_ticket"}
        if raw["status"] != "in_review":
            return {"error": "not_in_review"}
        prov = json.loads(raw["provenance"] or "[]")
        prov.append({"event": "review", "reviewer": reviewer_id, "verdict": "approved", "ts": now})
        self._conn.execute(
            "UPDATE work_queue SET status='done', result_status='approved', completed_at=?,"
            " provenance=?, updated_at=? WHERE ticket_id=?",
            (now, json.dumps(prov), now, ticket_id),
        )
        self._conn.commit()
        self.memory.record_evaluation({
            "provider": raw["lease_holder"], "task_id": raw["task_id"], "status": "completed",
        })
        if raw["task_id"]:
            self._set_task_state(raw["task_id"], TaskState.COMPLETE)
        return {"ticket_status": "done", "result_status": "approved"}

    @_synchronized
    def reject(
        self,
        reviewer_id: str,
        reviewer_tier: Union[str, ProviderPrivacyTier, None],
        ticket_id: str,
        reason: str = "",
        now: Optional[float] = None,
    ) -> dict[str, Any]:
        """Reject an in-review result. Requeues for another attempt while
        attempts remain; otherwise the ticket (and any linked task) fails."""
        now = _now() if now is None else now
        if _tier_value(reviewer_tier) != _TRUSTED_TIER:
            return {"error": "reviewer_not_authorized"}
        raw = self._raw(ticket_id)
        if raw is None:
            return {"error": "unknown_ticket"}
        if raw["status"] != "in_review":
            return {"error": "not_in_review"}
        self.memory.record_evaluation({
            "provider": raw["lease_holder"], "task_id": raw["task_id"], "status": "failed",
        })
        prov = json.loads(raw["provenance"] or "[]")
        prov.append({"event": "review", "reviewer": reviewer_id, "verdict": "rejected",
                     "reason": reason, "ts": now})
        if raw["attempts"] < raw["max_attempts"]:
            self._conn.execute(
                "UPDATE work_queue SET status='open', result_status='rejected',"
                " lease_holder=NULL, lease_tier=NULL, lease_token=NULL, lease_expires_at=NULL,"
                " provenance=?, updated_at=? WHERE ticket_id=?",
                (json.dumps(prov), now, ticket_id),
            )
            self._conn.commit()
            return {"ticket_status": "open", "result_status": "rejected"}
        self._conn.execute(
            "UPDATE work_queue SET status='failed', result_status='rejected',"
            " lease_holder=NULL, lease_token=NULL, lease_expires_at=NULL,"
            " provenance=?, updated_at=? WHERE ticket_id=?",
            (json.dumps(prov), now, ticket_id),
        )
        self._conn.commit()
        if raw["task_id"]:
            self._set_task_state(raw["task_id"], TaskState.FAILED)
        return {"ticket_status": "failed", "result_status": "rejected"}

    # ---------------------------------------------------------------- link
    @_synchronized
    def link(self, ticket_id: str, depends_on: str) -> dict[str, Any]:
        """Add a dependency edge with a privacy-monotonicity check: the child's
        admissible-tier set must be a subset of the parent's (child at least as
        restrictive), so sensitive output cannot be laundered down to a lower
        class through the edge."""
        child = self._raw(ticket_id)
        parent = self._raw(depends_on)
        if child is None or parent is None:
            return {"error": "unknown_ticket"}
        if ticket_id == depends_on:
            return {"error": "self_dependency"}
        # Cycle guard: if the parent already (transitively) depends on the child,
        # adding child->parent closes a loop where each ticket waits on the other
        # forever (both permanently 'blocked'). Reject at link time.
        if self._depends_transitively(depends_on, ticket_id):
            return {"error": "dependency_cycle"}
        child_tiers = set(self.allowed_tiers(child["privacy_class"]))
        parent_tiers = set(self.allowed_tiers(parent["privacy_class"]))
        if not child_tiers.issubset(parent_tiers):
            return {"error": "privacy_downgrade"}
        self._conn.execute(
            "INSERT OR IGNORE INTO work_queue_deps(ticket_id, depends_on) VALUES(?,?)",
            (ticket_id, depends_on),
        )
        self._conn.commit()
        return {"ok": True}

    # ---------------------------------------------------------------- reaper
    @_synchronized
    def reap_expired(self, now: Optional[float] = None) -> dict[str, list[str]]:
        """Requeue expired leases that still have attempts, park the exhausted.
        Two statements, one commit. Call periodically from an explicit loop, or
        let ``start_reaper`` run it on an opt-in daemon thread."""
        now = _now() if now is None else now
        requeued = [
            r["ticket_id"]
            for r in self._conn.execute(
                "UPDATE work_queue SET status='open', lease_holder=NULL, lease_tier=NULL,"
                " lease_token=NULL, lease_expires_at=NULL, updated_at=?"
                " WHERE status='claimed' AND lease_expires_at<? AND attempts<max_attempts"
                " RETURNING ticket_id",
                (now, now),
            ).fetchall()
        ]
        parked = [
            r["ticket_id"]
            for r in self._conn.execute(
                "UPDATE work_queue SET status='parked', park_reason='max_attempts_exhausted',"
                " lease_holder=NULL, lease_tier=NULL, lease_token=NULL, lease_expires_at=NULL,"
                " updated_at=?"
                " WHERE status='claimed' AND lease_expires_at<? AND attempts>=max_attempts"
                " RETURNING ticket_id",
                (now, now),
            ).fetchall()
        ]
        self._conn.commit()
        return {"requeued": requeued, "parked": parked}

    def start_reaper(
        self, interval: float = 30.0, stop: Optional[threading.Event] = None
    ) -> tuple[threading.Thread, threading.Event]:
        """Opt-in background reaper: a daemon thread that calls ``reap_expired``
        every ``interval`` seconds until ``stop`` is set. Without this (or an
        explicit periodic call), a worker that dies mid-lease leaves its ticket
        ``claimed`` forever — heartbeating only keeps a LIVE worker's lease fresh.

        Returns ``(thread, stop_event)``; call ``stop.set()`` (then optionally
        ``thread.join()``) to end it. The loop swallows per-tick errors so a
        transient DB hiccup can't kill the reaper. Off by default so the offline
        test gate stays deterministic — callers wanting self-healing turn it on
        explicitly (e.g. the broker process hosting the pull endpoints)."""
        stop = stop or threading.Event()

        def _loop() -> None:
            while not stop.wait(interval):
                try:
                    self.reap_expired()
                except Exception:  # noqa: BLE001 — a bad tick must not kill the reaper
                    pass

        thread = threading.Thread(target=_loop, daemon=True, name="agentconnect-wq-reaper")
        thread.start()
        return thread, stop

    # ---------------------------------------------------------------- status
    def status(
        self,
        ticket_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Redacted ticket rows for audit/UX. NEVER returns payload content —
        only the payload_ref id and queue metadata. ``blocked`` is derived (an
        open ticket with an unsatisfied dependency), never a stored state."""
        if ticket_id is not None:
            rows = self._conn.execute(
                "SELECT * FROM work_queue WHERE ticket_id=?", (ticket_id,)
            ).fetchall()
        elif status is not None:
            rows = self._conn.execute(
                "SELECT * FROM work_queue WHERE status=? ORDER BY updated_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM work_queue ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._status_row(r) for r in rows]

    def open_capability_requirements(self) -> list[dict[str, Any]]:
        """Distinct required-capability sets among OPEN tickets, with counts —
        lets an operator spot "no worker can ever claim this" without exposing
        any payload. Capabilities remain a matching filter only (never an
        authorization boundary); this is observability, not a gate."""
        rows = self._conn.execute(
            "SELECT required_capabilities AS rc FROM work_queue WHERE status='open'"
        ).fetchall()
        # Group by the capability SET, not the raw JSON text: two tickets that
        # require the same capabilities but serialized their list in a different
        # order (e.g. ["a","b"] vs ["b","a"]) are ONE requirement, so we
        # normalize each to a sorted, de-duplicated tuple before counting.
        # Doing this in Python (not SQL GROUP BY) is what keeps the
        # "no capable worker can ever claim this" signal from fragmenting.
        counts: dict[tuple[str, ...], int] = {}
        for r in rows:
            key = tuple(sorted(set(json.loads(r["rc"] or "[]"))))
            counts[key] = counts.get(key, 0) + 1
        return [
            {"required_capabilities": list(key), "open_tickets": n}
            for key, n in sorted(counts.items())
        ]

    def _operator_row(self, row: Any) -> dict[str, Any]:
        """Broader payload-free projection for the operator surface: adds
        capabilities/provenance/refs/timestamps on top of ``_status_row``. Still
        NEVER the payload/result content, and never ``task_id`` (a live handle
        into the un-redacted submission) — only ids, tiers, status, and the
        provenance audit trail."""
        d = dict(row)
        derived = d["status"]
        if derived == "open" and self._is_blocked(d["ticket_id"]):
            derived = "blocked"
        return {
            "ticket_id": d["ticket_id"],
            "status": derived,
            "privacy_class": d["privacy_class"],
            "allowed_tiers": json.loads(d["allowed_tiers"] or "[]"),
            "required_capabilities": json.loads(d["required_capabilities"] or "[]"),
            "priority": d["priority"],
            "attempts": d["attempts"],
            "max_attempts": d["max_attempts"],
            "result_status": d["result_status"],
            "assignee": d["assignee"],
            "park_reason": d["park_reason"],
            "origin": d["origin"],
            "payload_ref": d["payload_ref"],
            "result_ref": d["result_ref"],
            "provenance": json.loads(d["provenance"] or "[]"),
            "created_at": d["created_at"],
            "updated_at": d["updated_at"],
        }

    def list_tickets(
        self,
        status: Optional[str] = None,
        privacy_class: Optional[Union[str, PrivacyClass]] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Operator-facing ticket listing, payload-free (see ``_operator_row``).
        Optional filters on status and/or privacy_class; ordered most-recently-
        updated first."""
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status=?")
            params.append(status)
        if privacy_class is not None:
            clauses.append("privacy_class=?")
            params.append(_class_value(privacy_class))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM work_queue{where} ORDER BY updated_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [self._operator_row(r) for r in rows]

    def pending_review(self, limit: int = 100) -> list[dict[str, Any]]:
        """The ``in_review`` backlog awaiting a local_only reviewer's
        ``approve``/``reject`` — the human-spot-check triage queue."""
        return self.list_tickets(status="in_review", limit=limit)

    def stats(self) -> dict[str, Any]:
        """Counts by status and by privacy_class, plus the distinct open
        capability requirements — a payload-free operator dashboard summary."""
        by_status = {
            r["status"]: r["n"]
            for r in self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM work_queue GROUP BY status"
            ).fetchall()
        }
        by_privacy = {
            r["privacy_class"]: r["n"]
            for r in self._conn.execute(
                "SELECT privacy_class, COUNT(*) AS n FROM work_queue GROUP BY privacy_class"
            ).fetchall()
        }
        return {
            "by_status": by_status,
            "by_privacy_class": by_privacy,
            "capability_requirements": self.open_capability_requirements(),
        }

    def _is_blocked(self, ticket_id: str) -> bool:
        # Blocked iff any dependency edge lacks a parent that is 'done' — a
        # dangling edge (parent missing) counts as unsatisfied (fail-closed),
        # matching the claim gate in _try_claim.
        row = self._conn.execute(
            "SELECT 1 FROM work_queue_deps d WHERE d.ticket_id=?"
            "   AND NOT EXISTS (SELECT 1 FROM work_queue p"
            "                    WHERE p.ticket_id=d.depends_on AND p.status='done')"
            " LIMIT 1",
            (ticket_id,),
        ).fetchone()
        return row is not None

    def _depends_transitively(self, start: str, target: str) -> bool:
        """Is there a dependency path start -> ... -> target following
        ``depends_on`` edges? Used by link() to reject cycles."""
        seen: set[str] = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur == target:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            rows = self._conn.execute(
                "SELECT depends_on FROM work_queue_deps WHERE ticket_id=?", (cur,)
            ).fetchall()
            stack.extend(r["depends_on"] for r in rows)
        return False

    # ------------------------------------------------------------- internals
    def _raw(self, ticket_id: str) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM work_queue WHERE ticket_id=?", (ticket_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def get(self, ticket_id: str) -> Optional[dict[str, Any]]:
        """Full internal row (includes lease_token) — for the trusted control
        plane and tests, NOT for handing to a worker."""
        return self._raw(ticket_id)

    def _status_row(self, row: Any) -> dict[str, Any]:
        d = dict(row)
        derived = d["status"]
        if derived == "open" and self._is_blocked(d["ticket_id"]):
            derived = "blocked"
        return {
            "ticket_id": d["ticket_id"],
            "status": derived,
            "privacy_class": d["privacy_class"],
            "priority": d["priority"],
            "attempts": d["attempts"],
            "max_attempts": d["max_attempts"],
            "result_status": d["result_status"],
            "assignee": d["assignee"],
            "park_reason": d["park_reason"],
            "payload_ref": d["payload_ref"],
        }

    def _public_row(self, row: Any) -> dict[str, Any]:
        d = dict(row)
        return {
            "ticket_id": d["ticket_id"],
            "status": d["status"],
            "privacy_class": d["privacy_class"],
            "allowed_tiers": json.loads(d["allowed_tiers"] or "[]"),
            "park_reason": d["park_reason"],
        }

    def _set_task_state(self, task_id: str, state: TaskState) -> None:
        """Drive a linked task's state. S1 core is framework-free, so it writes
        the store directly (the router pipeline's strict FSM is layered in S2)."""
        if self.memory.get_task(task_id) is not None:
            self.memory.update_task(task_id, state=state.value)
