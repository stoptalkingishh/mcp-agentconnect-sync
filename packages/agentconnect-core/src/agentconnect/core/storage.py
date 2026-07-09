"""SQLite persistence for the task ledger (spec §8).

The *class* is the storage seam: `AgentConnectService` only ever calls the method
surface below, never SQL. A Postgres implementation later needs to provide the
same methods plus `transaction()` with the same semantics — a serialized
read-modify-write span, so invariants like "one active primary_manager claim"
hold under concurrency.

Concurrency note (learned the hard way in `common/memory.py`): a SQLite
transaction belongs to the *connection*, not the thread. A FastAPI sync endpoint
pool shares one connection across threads, so a peer's `commit()` would land in
the middle of another thread's read-modify-write. Every write path therefore runs
under one reentrant lock held across the whole execute→commit span, and file DBs
get WAL + `busy_timeout` for the cross-process case.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from .execution import ExecutionHandle
from .models import (
    ApprovalRecord,
    Artifact,
    ArtifactSummary,
    Attempt,
    Claim,
    Constraint,
    Decision,
    Event,
    ExternalRef,
    InboxItem,
    Review,
    Subtask,
    Task,
    TaskFilters,
    TaskSummary,
    WorkerRun,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, goal TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL, priority TEXT NOT NULL, created_by TEXT NOT NULL,
    created_at REAL NOT NULL, updated_at REAL NOT NULL,
    current_manager TEXT, handoff_summary TEXT,
    linear_issue_id TEXT, linear_issue_url TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS constraints (
    id TEXT PRIMARY KEY, task_id TEXT NOT NULL, text TEXT NOT NULL,
    created_by TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY, task_id TEXT NOT NULL, manager_id TEXT NOT NULL,
    role TEXT NOT NULL, expires_at REAL NOT NULL, created_at REAL NOT NULL,
    released_at REAL
);
CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY, task_id TEXT NOT NULL, made_by TEXT NOT NULL,
    decision TEXT NOT NULL, rationale TEXT NOT NULL DEFAULT '',
    locked INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL,
    superseded_by TEXT
);
CREATE TABLE IF NOT EXISTS attempts (
    id TEXT PRIMARY KEY, task_id TEXT NOT NULL, actor_id TEXT NOT NULL,
    actor_type TEXT NOT NULL, summary TEXT NOT NULL, outcome TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL, artifact_refs_json TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY, task_id TEXT NOT NULL, type TEXT NOT NULL, path TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '', created_by TEXT NOT NULL, created_at REAL NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0, metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS reviews (
    id TEXT PRIMARY KEY, task_id TEXT NOT NULL, requested_by TEXT NOT NULL,
    assigned_to TEXT NOT NULL, status TEXT NOT NULL,
    criteria_json TEXT NOT NULL DEFAULT '[]', artifact_refs_json TEXT NOT NULL DEFAULT '[]',
    result_artifact_id TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS subtasks (
    id TEXT PRIMARY KEY, parent_task_id TEXT NOT NULL, title TEXT NOT NULL,
    instructions TEXT NOT NULL, status TEXT NOT NULL, privacy_tier TEXT NOT NULL,
    preferred_worker TEXT, assigned_worker TEXT,
    created_at REAL NOT NULL, updated_at REAL NOT NULL,
    result_artifact_id TEXT, route_reason_json TEXT NOT NULL DEFAULT '{}',
    sandbox_json TEXT NOT NULL DEFAULT '{}',
    required_capabilities_json TEXT NOT NULL DEFAULT '[]',
    approved_by TEXT, approved_max_cost_usd REAL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS worker_runs (
    id TEXT PRIMARY KEY, subtask_id TEXT NOT NULL, worker_id TEXT NOT NULL,
    harness TEXT NOT NULL, model TEXT, status TEXT NOT NULL,
    route_reason_json TEXT NOT NULL DEFAULT '{}',
    started_at REAL NOT NULL, finished_at REAL,
    input_artifact_id TEXT, output_artifact_id TEXT,
    metrics_json TEXT NOT NULL DEFAULT '{}', error TEXT
);
CREATE TABLE IF NOT EXISTS external_refs (
    id TEXT PRIMARY KEY, entity_type TEXT NOT NULL, entity_id TEXT NOT NULL,
    provider TEXT NOT NULL, external_id TEXT NOT NULL, external_url TEXT,
    sync_enabled INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL, updated_at REAL NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (entity_type, entity_id, provider)
);
CREATE TABLE IF NOT EXISTS inbox_items (
    id TEXT PRIMARY KEY, manager_id TEXT NOT NULL, kind TEXT NOT NULL,
    ref_id TEXT NOT NULL, task_id TEXT, title TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL, dismissed_at REAL,
    UNIQUE (manager_id, kind, ref_id)
);
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY, task_id TEXT, kind TEXT NOT NULL, actor TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY, task_id TEXT NOT NULL, subtask_id TEXT NOT NULL,
    status TEXT NOT NULL, requested_worker TEXT, requested_location TEXT,
    estimated_cost_usd REAL NOT NULL DEFAULT 0, max_cost_usd REAL,
    decided_by TEXT, reason TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL, decided_at REAL
);
CREATE TABLE IF NOT EXISTS executions (
    handle_id TEXT PRIMARY KEY, backend TEXT NOT NULL, entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL, workflow_id TEXT, run_id TEXT, state TEXT NOT NULL,
    created_at REAL NOT NULL, updated_at REAL NOT NULL, detail TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_approvals_subtask ON approvals(subtask_id, status);
CREATE INDEX IF NOT EXISTS idx_executions_entity ON executions(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_executions_workflow ON executions(workflow_id);
CREATE INDEX IF NOT EXISTS idx_constraints_task ON constraints(task_id);
CREATE INDEX IF NOT EXISTS idx_claims_task ON claims(task_id);
CREATE INDEX IF NOT EXISTS idx_decisions_task ON decisions(task_id);
CREATE INDEX IF NOT EXISTS idx_attempts_task ON attempts(task_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_task ON artifacts(task_id);
CREATE INDEX IF NOT EXISTS idx_reviews_task ON reviews(task_id);
CREATE INDEX IF NOT EXISTS idx_reviews_assignee ON reviews(assigned_to, status);
CREATE INDEX IF NOT EXISTS idx_subtasks_task ON subtasks(parent_task_id);
CREATE INDEX IF NOT EXISTS idx_runs_subtask ON worker_runs(subtask_id);
CREATE INDEX IF NOT EXISTS idx_events_task ON events(task_id);
CREATE INDEX IF NOT EXISTS idx_extrefs_lookup ON external_refs(provider, external_id);
"""


def default_db_path() -> str:
    env = os.environ.get("AGENTCONNECT_DB_PATH")
    if env:
        return env
    return str(Path.home() / ".agentconnect" / "agentconnect.db")


def _j(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _u(text: Optional[str], fallback: Any) -> Any:
    if not text:
        return fallback
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return fallback


class SqliteStorage:
    def __init__(self, path: str | os.PathLike[str] = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
            self._conn.execute("PRAGMA busy_timeout = 5000")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Serialized read-modify-write span. Reentrant: nesting is a no-op."""
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # -------------------------------------------------------------- tasks
    def insert_task(self, task: Task) -> Task:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO tasks (id,title,goal,status,priority,created_by,created_at,"
                "updated_at,current_manager,handoff_summary,linear_issue_id,linear_issue_url,"
                "metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (task.id, task.title, task.goal, task.status.value, task.priority.value,
                 task.created_by, task.created_at, task.updated_at, task.current_manager,
                 task.handoff_summary, task.linear_issue_id, task.linear_issue_url,
                 _j(task.metadata)),
            )
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return self._task(row) if row else None

    def update_task(self, task_id: str, **fields: Any) -> None:
        if not fields:
            return
        if "metadata" in fields:
            fields["metadata_json"] = _j(fields.pop("metadata"))
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.transaction() as c:
            c.execute(f"UPDATE tasks SET {cols} WHERE id=?", (*fields.values(), task_id))

    def list_tasks(self, filters: TaskFilters) -> list[TaskSummary]:
        sql = "SELECT * FROM tasks"
        where, params = [], []
        if filters.status:
            where.append("status=?")
            params.append(filters.status.value)
        if filters.current_manager:
            where.append("current_manager=?")
            params.append(filters.current_manager)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params += [max(0, filters.limit), max(0, filters.offset)]
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            TaskSummary(
                id=r["id"], title=r["title"], status=r["status"], priority=r["priority"],
                current_manager=r["current_manager"], updated_at=r["updated_at"],
                linear_issue_url=r["linear_issue_url"],
            )
            for r in rows
        ]

    @staticmethod
    def _task(r: sqlite3.Row) -> Task:
        return Task(
            id=r["id"], title=r["title"], goal=r["goal"], status=r["status"],
            priority=r["priority"], created_by=r["created_by"], created_at=r["created_at"],
            updated_at=r["updated_at"], current_manager=r["current_manager"],
            handoff_summary=r["handoff_summary"], linear_issue_id=r["linear_issue_id"],
            linear_issue_url=r["linear_issue_url"], metadata=_u(r["metadata_json"], {}),
        )

    # --------------------------------------------------------- constraints
    def insert_constraint(self, c_: Constraint) -> Constraint:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO constraints (id,task_id,text,created_by,created_at) VALUES (?,?,?,?,?)",
                (c_.id, c_.task_id, c_.text, c_.created_by, c_.created_at),
            )
        return c_

    def list_constraints(self, task_id: str) -> list[Constraint]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM constraints WHERE task_id=? ORDER BY created_at", (task_id,)
            ).fetchall()
        return [Constraint(**dict(r)) for r in rows]

    # -------------------------------------------------------------- claims
    def insert_claim(self, claim: Claim, conn: Optional[sqlite3.Connection] = None) -> Claim:
        sql = ("INSERT INTO claims (id,task_id,manager_id,role,expires_at,created_at,released_at)"
               " VALUES (?,?,?,?,?,?,?)")
        args = (claim.id, claim.task_id, claim.manager_id, claim.role.value,
                claim.expires_at, claim.created_at, claim.released_at)
        if conn is not None:
            conn.execute(sql, args)
        else:
            with self.transaction() as c:
                c.execute(sql, args)
        return claim

    def active_claims(self, task_id: str, at: float,
                      conn: Optional[sqlite3.Connection] = None) -> list[Claim]:
        c = conn or self._conn
        with self._lock:
            rows = c.execute(
                "SELECT * FROM claims WHERE task_id=? AND released_at IS NULL AND expires_at > ?"
                " ORDER BY created_at",
                (task_id, at),
            ).fetchall()
        return [self._claim(r) for r in rows]

    def list_claims(self, task_id: str) -> list[Claim]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM claims WHERE task_id=? ORDER BY created_at", (task_id,)
            ).fetchall()
        return [self._claim(r) for r in rows]

    def release_claims(self, task_id: str, manager_id: str, at: float) -> int:
        with self.transaction() as c:
            cur = c.execute(
                "UPDATE claims SET released_at=? WHERE task_id=? AND manager_id=?"
                " AND released_at IS NULL",
                (at, task_id, manager_id),
            )
            return cur.rowcount

    @staticmethod
    def _claim(r: sqlite3.Row) -> Claim:
        return Claim(
            id=r["id"], task_id=r["task_id"], manager_id=r["manager_id"], role=r["role"],
            expires_at=r["expires_at"], created_at=r["created_at"], released_at=r["released_at"],
        )

    # ----------------------------------------------------------- decisions
    def insert_decision(self, d: Decision, conn: Optional[sqlite3.Connection] = None) -> Decision:
        sql = ("INSERT INTO decisions (id,task_id,made_by,decision,rationale,locked,created_at,"
               "superseded_by) VALUES (?,?,?,?,?,?,?,?)")
        args = (d.id, d.task_id, d.made_by, d.decision, d.rationale, int(d.locked),
                d.created_at, d.superseded_by)
        if conn is not None:
            conn.execute(sql, args)
        else:
            with self.transaction() as c:
                c.execute(sql, args)
        return d

    def get_decision(self, decision_id: str,
                     conn: Optional[sqlite3.Connection] = None) -> Optional[Decision]:
        c = conn or self._conn
        with self._lock:
            row = c.execute("SELECT * FROM decisions WHERE id=?", (decision_id,)).fetchone()
        return self._decision(row) if row else None

    def list_decisions(self, task_id: str) -> list[Decision]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM decisions WHERE task_id=? ORDER BY created_at", (task_id,)
            ).fetchall()
        return [self._decision(r) for r in rows]

    def mark_superseded(self, decision_id: str, by: str,
                        conn: Optional[sqlite3.Connection] = None) -> None:
        sql = "UPDATE decisions SET superseded_by=? WHERE id=?"
        if conn is not None:
            conn.execute(sql, (by, decision_id))
        else:
            with self.transaction() as c:
                c.execute(sql, (by, decision_id))

    @staticmethod
    def _decision(r: sqlite3.Row) -> Decision:
        return Decision(
            id=r["id"], task_id=r["task_id"], made_by=r["made_by"], decision=r["decision"],
            rationale=r["rationale"], locked=bool(r["locked"]), created_at=r["created_at"],
            superseded_by=r["superseded_by"],
        )

    # ------------------------------------------------------------ attempts
    def insert_attempt(self, a: Attempt) -> Attempt:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO attempts (id,task_id,actor_id,actor_type,summary,outcome,created_at,"
                "artifact_refs_json) VALUES (?,?,?,?,?,?,?,?)",
                (a.id, a.task_id, a.actor_id, a.actor_type.value, a.summary, a.outcome,
                 a.created_at, _j(a.artifact_refs)),
            )
        return a

    def list_attempts(self, task_id: str, limit: Optional[int] = None) -> list[Attempt]:
        sql = "SELECT * FROM attempts WHERE task_id=? ORDER BY created_at"
        with self._lock:
            rows = self._conn.execute(sql, (task_id,)).fetchall()
        out = [
            Attempt(
                id=r["id"], task_id=r["task_id"], actor_id=r["actor_id"],
                actor_type=r["actor_type"], summary=r["summary"], outcome=r["outcome"],
                created_at=r["created_at"], artifact_refs=_u(r["artifact_refs_json"], []),
            )
            for r in rows
        ]
        return out[-limit:] if limit else out

    # ----------------------------------------------------------- artifacts
    def insert_artifact(self, a: Artifact) -> Artifact:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO artifacts (id,task_id,type,path,summary,created_by,created_at,"
                "size_bytes,metadata_json) VALUES (?,?,?,?,?,?,?,?,?)",
                (a.id, a.task_id, a.type.value, a.path, a.summary, a.created_by, a.created_at,
                 a.size_bytes, _j(a.metadata)),
            )
        return a

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM artifacts WHERE id=?", (artifact_id,)
            ).fetchone()
        if not row:
            return None
        return Artifact(
            id=row["id"], task_id=row["task_id"], type=row["type"], path=row["path"],
            summary=row["summary"], created_by=row["created_by"], created_at=row["created_at"],
            size_bytes=row["size_bytes"], metadata=_u(row["metadata_json"], {}),
        )

    def list_artifacts(self, task_id: str) -> list[ArtifactSummary]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM artifacts WHERE task_id=? ORDER BY created_at", (task_id,)
            ).fetchall()
        return [
            ArtifactSummary(
                id=r["id"], task_id=r["task_id"], type=r["type"], summary=r["summary"],
                size_bytes=r["size_bytes"], created_by=r["created_by"], created_at=r["created_at"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------- reviews
    def insert_review(self, rv: Review) -> Review:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO reviews (id,task_id,requested_by,assigned_to,status,criteria_json,"
                "artifact_refs_json,result_artifact_id,created_at,updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (rv.id, rv.task_id, rv.requested_by, rv.assigned_to, rv.status.value,
                 _j(rv.criteria), _j(rv.artifact_refs), rv.result_artifact_id,
                 rv.created_at, rv.updated_at),
            )
        return rv

    def get_review(self, review_id: str,
                   conn: Optional[sqlite3.Connection] = None) -> Optional[Review]:
        c = conn or self._conn
        with self._lock:
            row = c.execute("SELECT * FROM reviews WHERE id=?", (review_id,)).fetchone()
        return self._review(row) if row else None

    def update_review(self, review_id: str, conn: Optional[sqlite3.Connection] = None,
                      **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        sql = f"UPDATE reviews SET {cols} WHERE id=?"
        args = (*fields.values(), review_id)
        if conn is not None:
            conn.execute(sql, args)
        else:
            with self.transaction() as c:
                c.execute(sql, args)

    def list_reviews(self, task_id: str) -> list[Review]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM reviews WHERE task_id=? ORDER BY created_at", (task_id,)
            ).fetchall()
        return [self._review(r) for r in rows]

    def reviews_for_manager(self, manager_id: str, statuses: tuple[str, ...]) -> list[Review]:
        marks = ",".join("?" * len(statuses))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM reviews WHERE assigned_to=? AND status IN ({marks})"
                " ORDER BY created_at",
                (manager_id, *statuses),
            ).fetchall()
        return [self._review(r) for r in rows]

    @staticmethod
    def _review(r: sqlite3.Row) -> Review:
        return Review(
            id=r["id"], task_id=r["task_id"], requested_by=r["requested_by"],
            assigned_to=r["assigned_to"], status=r["status"],
            criteria=_u(r["criteria_json"], []), artifact_refs=_u(r["artifact_refs_json"], []),
            result_artifact_id=r["result_artifact_id"], created_at=r["created_at"],
            updated_at=r["updated_at"],
        )

    # ------------------------------------------------------------ subtasks
    def insert_subtask(self, s: Subtask) -> Subtask:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO subtasks (id,parent_task_id,title,instructions,status,privacy_tier,"
                "preferred_worker,assigned_worker,created_at,updated_at,result_artifact_id,"
                "route_reason_json,sandbox_json,required_capabilities_json,approved_by,"
                "approved_max_cost_usd,metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (s.id, s.parent_task_id, s.title, s.instructions, s.status.value,
                 s.privacy_tier.value, s.preferred_worker, s.assigned_worker, s.created_at,
                 s.updated_at, s.result_artifact_id, _j(s.route_reason),
                 _j(s.sandbox.model_dump(mode="json")), _j(s.required_capabilities),
                 s.approved_by, s.approved_max_cost_usd, _j(s.metadata)),
            )
        return s

    def get_subtask(self, subtask_id: str) -> Optional[Subtask]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM subtasks WHERE id=?", (subtask_id,)
            ).fetchone()
        return self._subtask(row) if row else None

    def update_subtask(self, subtask_id: str, **fields: Any) -> None:
        if not fields:
            return
        if "route_reason" in fields:
            fields["route_reason_json"] = _j(fields.pop("route_reason"))
        if "metadata" in fields:
            fields["metadata_json"] = _j(fields.pop("metadata"))
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.transaction() as c:
            c.execute(f"UPDATE subtasks SET {cols} WHERE id=?", (*fields.values(), subtask_id))

    def list_subtasks(self, task_id: str) -> list[Subtask]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM subtasks WHERE parent_task_id=? ORDER BY created_at", (task_id,)
            ).fetchall()
        return [self._subtask(r) for r in rows]

    @staticmethod
    def _subtask(r: sqlite3.Row) -> Subtask:
        return Subtask(
            id=r["id"], parent_task_id=r["parent_task_id"], title=r["title"],
            instructions=r["instructions"], status=r["status"], privacy_tier=r["privacy_tier"],
            preferred_worker=r["preferred_worker"], assigned_worker=r["assigned_worker"],
            created_at=r["created_at"], updated_at=r["updated_at"],
            result_artifact_id=r["result_artifact_id"],
            route_reason=_u(r["route_reason_json"], {}),
            sandbox=_u(r["sandbox_json"], {}),
            required_capabilities=_u(r["required_capabilities_json"], []),
            approved_by=r["approved_by"], approved_max_cost_usd=r["approved_max_cost_usd"],
            metadata=_u(r["metadata_json"], {}),
        )

    # --------------------------------------------------------- worker runs
    def insert_run(self, run: WorkerRun) -> WorkerRun:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO worker_runs (id,subtask_id,worker_id,harness,model,status,"
                "route_reason_json,started_at,finished_at,input_artifact_id,output_artifact_id,"
                "metrics_json,error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run.id, run.subtask_id, run.worker_id, run.harness, run.model, run.status.value,
                 _j(run.route_reason), run.started_at, run.finished_at, run.input_artifact_id,
                 run.output_artifact_id, _j(run.metrics), run.error),
            )
        return run

    def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        if "metrics" in fields:
            fields["metrics_json"] = _j(fields.pop("metrics"))
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.transaction() as c:
            c.execute(f"UPDATE worker_runs SET {cols} WHERE id=?", (*fields.values(), run_id))

    def list_runs(self, subtask_id: str) -> list[WorkerRun]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM worker_runs WHERE subtask_id=? ORDER BY started_at", (subtask_id,)
            ).fetchall()
        return [
            WorkerRun(
                id=r["id"], subtask_id=r["subtask_id"], worker_id=r["worker_id"],
                harness=r["harness"], model=r["model"], status=r["status"],
                route_reason=_u(r["route_reason_json"], {}), started_at=r["started_at"],
                finished_at=r["finished_at"], input_artifact_id=r["input_artifact_id"],
                output_artifact_id=r["output_artifact_id"], metrics=_u(r["metrics_json"], {}),
                error=r["error"],
            )
            for r in rows
        ]

    # ------------------------------------------------------- external refs
    def upsert_external_ref(self, ref: ExternalRef) -> ExternalRef:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO external_refs (id,entity_type,entity_id,provider,external_id,"
                "external_url,sync_enabled,created_at,updated_at,metadata_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(entity_type,entity_id,provider) DO UPDATE SET"
                " external_id=excluded.external_id, external_url=excluded.external_url,"
                " sync_enabled=excluded.sync_enabled, updated_at=excluded.updated_at,"
                " metadata_json=excluded.metadata_json",
                (ref.id, ref.entity_type, ref.entity_id, ref.provider, ref.external_id,
                 ref.external_url, int(ref.sync_enabled), ref.created_at, ref.updated_at,
                 _j(ref.metadata)),
            )
        return self.get_external_ref(ref.entity_type, ref.entity_id, ref.provider) or ref

    def get_external_ref(self, entity_type: str, entity_id: str,
                         provider: str) -> Optional[ExternalRef]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM external_refs WHERE entity_type=? AND entity_id=? AND provider=?",
                (entity_type, entity_id, provider),
            ).fetchone()
        return self._extref(row) if row else None

    def find_by_external_id(self, provider: str, external_id: str) -> Optional[ExternalRef]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM external_refs WHERE provider=? AND external_id=?",
                (provider, external_id),
            ).fetchone()
        return self._extref(row) if row else None

    @staticmethod
    def _extref(r: sqlite3.Row) -> ExternalRef:
        return ExternalRef(
            id=r["id"], entity_type=r["entity_type"], entity_id=r["entity_id"],
            provider=r["provider"], external_id=r["external_id"], external_url=r["external_url"],
            sync_enabled=bool(r["sync_enabled"]), created_at=r["created_at"],
            updated_at=r["updated_at"], metadata=_u(r["metadata_json"], {}),
        )

    # --------------------------------------------------------------- inbox
    def insert_inbox_item(self, item: InboxItem) -> InboxItem:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO inbox_items (id,manager_id,kind,ref_id,task_id,title,created_at,"
                "dismissed_at) VALUES (?,?,?,?,?,?,?,?)"
                " ON CONFLICT(manager_id,kind,ref_id) DO NOTHING",
                (item.id, item.manager_id, item.kind.value, item.ref_id, item.task_id,
                 item.title, item.created_at, item.dismissed_at),
            )
        return item

    def list_inbox_items(self, manager_id: str) -> list[InboxItem]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM inbox_items WHERE manager_id=? AND dismissed_at IS NULL"
                " ORDER BY created_at",
                (manager_id,),
            ).fetchall()
        return [
            InboxItem(
                id=r["id"], manager_id=r["manager_id"], kind=r["kind"], ref_id=r["ref_id"],
                task_id=r["task_id"], title=r["title"], created_at=r["created_at"],
                dismissed_at=r["dismissed_at"],
            )
            for r in rows
        ]

    def dismiss_inbox_items(self, ref_id: str, at: float) -> None:
        with self.transaction() as c:
            c.execute(
                "UPDATE inbox_items SET dismissed_at=? WHERE ref_id=? AND dismissed_at IS NULL",
                (at, ref_id),
            )

    # ----------------------------------------------------------- approvals
    def insert_approval(self, a: ApprovalRecord) -> ApprovalRecord:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO approvals (id,task_id,subtask_id,status,requested_worker,"
                "requested_location,estimated_cost_usd,max_cost_usd,decided_by,reason,"
                "created_at,decided_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (a.id, a.task_id, a.subtask_id, a.status.value, a.requested_worker,
                 a.requested_location, a.estimated_cost_usd, a.max_cost_usd, a.decided_by,
                 a.reason, a.created_at, a.decided_at),
            )
        return a

    def get_approval(self, approval_id: str) -> Optional[ApprovalRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM approvals WHERE id=?", (approval_id,)
            ).fetchone()
        return self._approval(row) if row else None

    def pending_approval_for(self, subtask_id: str) -> Optional[ApprovalRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM approvals WHERE subtask_id=? AND status='pending'"
                " ORDER BY created_at DESC LIMIT 1",
                (subtask_id,),
            ).fetchone()
        return self._approval(row) if row else None

    def list_approvals(self, task_id: str) -> list[ApprovalRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM approvals WHERE task_id=? ORDER BY created_at", (task_id,)
            ).fetchall()
        return [self._approval(r) for r in rows]

    def update_approval(self, approval_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.transaction() as c:
            c.execute(f"UPDATE approvals SET {cols} WHERE id=?", (*fields.values(), approval_id))

    @staticmethod
    def _approval(r: sqlite3.Row) -> ApprovalRecord:
        return ApprovalRecord(
            id=r["id"], task_id=r["task_id"], subtask_id=r["subtask_id"], status=r["status"],
            requested_worker=r["requested_worker"], requested_location=r["requested_location"],
            estimated_cost_usd=r["estimated_cost_usd"], max_cost_usd=r["max_cost_usd"],
            decided_by=r["decided_by"], reason=r["reason"], created_at=r["created_at"],
            decided_at=r["decided_at"],
        )

    # ---------------------------------------------------------- executions
    def upsert_execution(self, h: ExecutionHandle) -> ExecutionHandle:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO executions (handle_id,backend,entity_type,entity_id,workflow_id,"
                "run_id,state,created_at,updated_at,detail) VALUES (?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(handle_id) DO UPDATE SET workflow_id=excluded.workflow_id,"
                " run_id=excluded.run_id, state=excluded.state, updated_at=excluded.updated_at,"
                " detail=excluded.detail",
                (h.handle_id, h.backend, h.entity_type, h.entity_id, h.workflow_id, h.run_id,
                 h.state.value, h.created_at, h.updated_at, h.detail),
            )
        return self.get_execution(h.handle_id) or h

    def get_execution(self, handle_id: str) -> Optional[ExecutionHandle]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM executions WHERE handle_id=? OR workflow_id=?",
                (handle_id, handle_id),
            ).fetchone()
        return self._execution(row) if row else None

    def executions_for(self, entity_type: str, entity_id: str) -> list[ExecutionHandle]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM executions WHERE entity_type=? AND entity_id=? ORDER BY created_at",
                (entity_type, entity_id),
            ).fetchall()
        return [self._execution(r) for r in rows]

    def update_execution(self, handle_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.transaction() as c:
            c.execute(
                f"UPDATE executions SET {cols} WHERE handle_id=? OR workflow_id=?",
                (*fields.values(), handle_id, handle_id),
            )

    @staticmethod
    def _execution(r: sqlite3.Row) -> ExecutionHandle:
        return ExecutionHandle(
            handle_id=r["handle_id"], backend=r["backend"], entity_type=r["entity_type"],
            entity_id=r["entity_id"], workflow_id=r["workflow_id"], run_id=r["run_id"],
            state=r["state"], created_at=r["created_at"], updated_at=r["updated_at"],
            detail=r["detail"],
        )

    # -------------------------------------------------------------- events
    def insert_event(self, e: Event) -> Event:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO events (id,task_id,kind,actor,payload_json,created_at)"
                " VALUES (?,?,?,?,?,?)",
                (e.id, e.task_id, e.kind, e.actor, _j(e.payload), e.created_at),
            )
        return e

    def list_events(self, task_id: Optional[str] = None, limit: int = 100) -> list[Event]:
        with self._lock:
            if task_id:
                rows = self._conn.execute(
                    "SELECT * FROM events WHERE task_id=? ORDER BY created_at DESC LIMIT ?",
                    (task_id, max(0, limit)),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM events ORDER BY created_at DESC LIMIT ?", (max(0, limit),)
                ).fetchall()
        return [
            Event(id=r["id"], task_id=r["task_id"], kind=r["kind"], actor=r["actor"],
                  payload=_u(r["payload_json"], {}), created_at=r["created_at"])
            for r in rows
        ]
