"""Shared memory & artifact store (handoff §8).

A SQLite-backed store that owns persistent task state, logs, summaries,
artifacts, routing decisions, and quota-usage records. Large outputs (patches,
logs, traces) are stored here and referenced by id; they are NEVER returned
wholesale through MCP — callers read them back in bounded chunks
(``read_artifact_chunk`` / ``get_log_slice``), enforcing context virtualization
(§9).

The store is deliberately dependency-light (stdlib ``sqlite3``) so the core runs
without the service frameworks.
"""

from __future__ import annotations

import functools
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id      TEXT PRIMARY KEY,
    state        TEXT NOT NULL,
    agent_type   TEXT,
    privacy_class TEXT,
    summary      TEXT,
    recommended_next_action TEXT,
    risks        TEXT,        -- json list
    submission   TEXT,        -- json
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id  TEXT PRIMARY KEY,
    task_id      TEXT,
    kind         TEXT NOT NULL,
    mime         TEXT NOT NULL DEFAULT 'text/plain',
    size_chars   INTEGER NOT NULL,
    content      TEXT NOT NULL,
    created_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS logs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id      TEXT NOT NULL,
    level        TEXT NOT NULL DEFAULT 'info',
    message      TEXT NOT NULL,
    created_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS routing_decisions (
    task_id      TEXT NOT NULL,
    decision     TEXT NOT NULL,   -- json RoutingDecision
    created_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS quota_records (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    provider     TEXT NOT NULL,
    task_id      TEXT,
    est_input    INTEGER DEFAULT 0,
    est_output   INTEGER DEFAULT 0,
    act_input    INTEGER DEFAULT 0,
    act_output   INTEGER DEFAULT 0,
    requests     INTEGER DEFAULT 0,
    est_cost_usd REAL DEFAULT 0,
    act_cost_usd REAL DEFAULT 0,
    status       TEXT,
    failure_reason TEXT,
    created_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    provider     TEXT NOT NULL,
    model        TEXT,
    task_id      TEXT,
    agent_type   TEXT,
    status       TEXT NOT NULL,     -- completed | failed
    latency_ms   REAL DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cost_usd     REAL DEFAULT 0,
    confidence   REAL,              -- worker-reported, if any
    retries      INTEGER DEFAULT 0,
    created_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,     -- json
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_logs_task ON logs(task_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_task ON artifacts(task_id);
CREATE INDEX IF NOT EXISTS idx_quota_provider ON quota_records(provider);
CREATE INDEX IF NOT EXISTS idx_eval_provider ON evaluations(provider);
CREATE INDEX IF NOT EXISTS idx_quota_created ON quota_records(created_at);

-- Federated work-queue (WorkQueue lives over this same connection so queue
-- state, task state, artifacts and audit are ONE transactional store). All
-- authorization (which tier may see/claim which privacy_class) is recomputed
-- live from routing config in WorkQueue; the allowed_tiers column below is a
-- denormalized cache for UX/indexing only and is NEVER trusted for a claim.
CREATE TABLE IF NOT EXISTS work_queue (
    ticket_id       TEXT PRIMARY KEY,
    dedup_key       TEXT,               -- origin idempotency key (partial-unique)
    origin          TEXT,               -- enqueuing identity (audit)
    task_id         TEXT,               -- FK-by-convention to tasks.task_id
    payload_ref     TEXT,               -- artifact id of the worker-visible payload
    privacy_class   TEXT NOT NULL,      -- authoritative for authorization
    allowed_tiers   TEXT,               -- json cache; NOT trusted for the claim
    required_capabilities TEXT,         -- json list; matching filter, not a gate
    priority        TEXT DEFAULT 'normal',
    status          TEXT NOT NULL,      -- open|claimed|in_review|done|parked|failed
    assignee        TEXT,               -- advisory router hint; never enforced
    lease_holder    TEXT,
    lease_tier      TEXT,
    lease_token     TEXT,               -- fencing token (fresh per claim)
    lease_expires_at REAL,
    attempts        INTEGER DEFAULT 0,
    max_attempts    INTEGER DEFAULT 3,
    result_ref      TEXT,               -- artifact id of the WorkerResult json
    result_status   TEXT,               -- pending|approved|rejected
    provenance      TEXT,               -- json audit trail
    park_reason     TEXT,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    claimed_at      REAL,
    completed_at    REAL
);
CREATE TABLE IF NOT EXISTS work_queue_deps (
    ticket_id   TEXT NOT NULL,
    depends_on  TEXT NOT NULL,
    PRIMARY KEY(ticket_id, depends_on)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_wq_dedup ON work_queue(dedup_key) WHERE dedup_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_wq_status_priority ON work_queue(status, priority, created_at);
CREATE INDEX IF NOT EXISTS idx_wq_lease ON work_queue(status, lease_expires_at);
CREATE INDEX IF NOT EXISTS idx_wqdeps_ticket ON work_queue_deps(ticket_id);
"""


def _now() -> float:
    return time.time()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class ArtifactChunk:
    artifact_id: str
    offset: int
    max_chars: int
    content: str
    total_size: int
    next_offset: Optional[int]  # None when the artifact is fully read


def _synchronized(method):
    """Serialize a connection-mutating method under the store's reentrant lock.

    The connection is shared across threads (``check_same_thread=False``) and a
    SQLite transaction is a property of the *connection*, not the thread. Without
    serialization, one thread's ``commit()``/``rollback()`` acts on another
    thread's in-flight statements — a losing claimant's ``rollback`` can discard a
    winning claimant's not-yet-committed ``UPDATE``. Every write path (here and in
    :class:`WorkQueue`, which shares this same lock) therefore holds the lock
    across its whole execute→commit/rollback span, so no two writers are ever
    between a DML statement and its commit at once. The lock is an
    :class:`~threading.RLock` so nested writers (e.g. ``WorkQueue.report`` calling
    ``put_artifact``) don't self-deadlock."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class SharedMemory:
    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        # One reentrant lock guards every commit/rollback boundary on the shared
        # connection — see _synchronized. WorkQueue binds to THIS lock so the
        # queue and the rest of the store serialize against each other, not just
        # within their own layer.
        self._lock = threading.RLock()
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Wait for a lock rather than failing instantly when a *separate*
        # connection (another process opening the same file) holds a write lock;
        # WAL lets readers proceed concurrently with a writer. No-ops harmlessly
        # on an in-memory db (which cannot be shared across connections anyway).
        self._conn.execute("PRAGMA busy_timeout=5000")
        if self.db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ----------------------------------------------------------------- tasks
    @_synchronized
    def create_task(self, submission: dict[str, Any], agent_type: Optional[str] = None) -> str:
        task_id = _new_id("task")
        now = _now()
        self._conn.execute(
            "INSERT INTO tasks(task_id, state, agent_type, submission, risks, created_at, updated_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (task_id, "CREATED", agent_type, json.dumps(submission), "[]", now, now),
        )
        self._conn.commit()
        return task_id

    @_synchronized
    def update_task(self, task_id: str, **fields: Any) -> None:
        if not fields:
            return
        if "risks" in fields and not isinstance(fields["risks"], str):
            fields["risks"] = json.dumps(fields["risks"])
        fields["updated_at"] = _now()
        cols = ", ".join(f"{k}=?" for k in fields)
        self._conn.execute(
            f"UPDATE tasks SET {cols} WHERE task_id=?", (*fields.values(), task_id)
        )
        self._conn.commit()

    def get_task(self, task_id: str) -> Optional[dict[str, Any]]:
        row = self._conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["risks"] = json.loads(d.get("risks") or "[]")
        if d.get("submission"):
            d["submission"] = json.loads(d["submission"])
        return d

    def list_tasks(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT task_id, state, agent_type, summary, updated_at FROM tasks"
            " ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def task_artifacts(self, task_id: str) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT artifact_id, kind FROM artifacts WHERE task_id=? ORDER BY created_at", (task_id,)
        ).fetchall()
        return {r["kind"]: r["artifact_id"] for r in rows}

    # ------------------------------------------------------------- artifacts
    @_synchronized
    def put_artifact(
        self, task_id: Optional[str], kind: str, content: str, mime: str = "text/plain"
    ) -> str:
        artifact_id = _new_id("artifact")
        self._conn.execute(
            "INSERT INTO artifacts(artifact_id, task_id, kind, mime, size_chars, content, created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (artifact_id, task_id, kind, mime, len(content), content, _now()),
        )
        self._conn.commit()
        return artifact_id

    def read_artifact_chunk(
        self, artifact_id: str, offset: int = 0, max_chars: int = 4000
    ) -> Optional[ArtifactChunk]:
        row = self._conn.execute(
            "SELECT size_chars, content FROM artifacts WHERE artifact_id=?", (artifact_id,)
        ).fetchone()
        if row is None:
            return None
        total = row["size_chars"]
        offset = max(0, offset)
        chunk = row["content"][offset : offset + max_chars]
        next_offset = offset + len(chunk)
        return ArtifactChunk(
            artifact_id=artifact_id,
            offset=offset,
            max_chars=max_chars,
            content=chunk,
            total_size=total,
            next_offset=next_offset if next_offset < total else None,
        )

    # ------------------------------------------------------------------ logs
    @_synchronized
    def append_log(self, task_id: str, message: str, level: str = "info") -> None:
        self._conn.execute(
            "INSERT INTO logs(task_id, level, message, created_at) VALUES(?,?,?,?)",
            (task_id, level, message, _now()),
        )
        self._conn.commit()

    def get_log_slice(
        self,
        task_id: str,
        level: Optional[str] = None,
        query: Optional[str] = None,
        max_lines: int = 100,
    ) -> list[dict[str, Any]]:
        sql = "SELECT level, message, created_at FROM logs WHERE task_id=?"
        params: list[Any] = [task_id]
        if level:
            sql += " AND level=?"
            params.append(level)
        if query:
            sql += " AND message LIKE ?"
            params.append(f"%{query}%")
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(max_lines)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in reversed(rows)]

    # -------------------------------------------------------------- routing
    @_synchronized
    def record_routing_decision(self, task_id: str, decision: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO routing_decisions(task_id, decision, created_at) VALUES(?,?,?)",
            (task_id, json.dumps(decision), _now()),
        )
        self._conn.commit()

    def get_routing_decisions(self, task_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT decision FROM routing_decisions WHERE task_id=? ORDER BY created_at", (task_id,)
        ).fetchall()
        return [json.loads(r["decision"]) for r in rows]

    # ----------------------------------------------------------- evaluations
    @_synchronized
    def record_evaluation(self, record: dict[str, Any]) -> None:
        cols = (
            "provider", "model", "task_id", "agent_type", "status", "latency_ms",
            "input_tokens", "output_tokens", "cost_usd", "confidence", "retries",
        )
        placeholders = ",".join("?" for _ in cols)
        self._conn.execute(
            f"INSERT INTO evaluations({','.join(cols)}, created_at) VALUES({placeholders}, ?)",
            (*[record.get(c) for c in cols], _now()),
        )
        self._conn.commit()

    def provider_eval_aggregate(self) -> list[dict[str, Any]]:
        """Per-provider aggregate outcomes (feeds learned routing + scorecards)."""
        rows = self._conn.execute(
            "SELECT provider,"
            " COUNT(*) AS samples,"
            " SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) AS successes,"
            " AVG(latency_ms) AS avg_latency_ms,"
            " AVG(cost_usd) AS avg_cost_usd,"
            " AVG(confidence) AS avg_confidence"
            " FROM evaluations GROUP BY provider"
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ search_memory
    def search_memory(
        self, query: str, scope: str = "all", limit: int = 20
    ) -> list[dict[str, Any]]:
        """Simple substring search across summaries, artifacts, and logs (§8, §23).

        `scope` in {all, tasks, artifacts, logs}. Returns compact hit records
        (id, kind, snippet) — never full bodies.
        """
        like = f"%{query}%"
        hits: list[dict[str, Any]] = []
        if scope in ("all", "tasks"):
            for r in self._conn.execute(
                "SELECT task_id, summary FROM tasks WHERE summary LIKE ? LIMIT ?", (like, limit)
            ):
                hits.append({"type": "task", "id": r["task_id"], "snippet": _snippet(r["summary"], query)})
        if scope in ("all", "artifacts"):
            for r in self._conn.execute(
                "SELECT artifact_id, kind, content FROM artifacts WHERE content LIKE ? LIMIT ?",
                (like, limit),
            ):
                hits.append(
                    {"type": "artifact", "id": r["artifact_id"], "kind": r["kind"],
                     "snippet": _snippet(r["content"], query)}
                )
        if scope in ("all", "logs"):
            for r in self._conn.execute(
                "SELECT task_id, message FROM logs WHERE message LIKE ? LIMIT ?", (like, limit)
            ):
                hits.append({"type": "log", "id": r["task_id"], "snippet": _snippet(r["message"], query)})
        return hits[:limit]

    # ------------------------------------------------------------- quota records
    @_synchronized
    def record_quota_usage(self, record: dict[str, Any]) -> None:
        cols = (
            "provider", "task_id", "est_input", "est_output", "act_input", "act_output",
            "requests", "est_cost_usd", "act_cost_usd", "status", "failure_reason",
        )
        vals = [record.get(c) for c in cols]
        placeholders = ",".join("?" for _ in cols)
        self._conn.execute(
            f"INSERT INTO quota_records({','.join(cols)}, created_at) VALUES({placeholders}, ?)",
            (*vals, _now()),
        )
        self._conn.commit()

    def quota_usage_since(self, provider: str, since_epoch: float) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(requests),0) AS requests,"
            " COALESCE(SUM(act_input+act_output),0) AS tokens,"
            " COALESCE(SUM(act_cost_usd),0) AS cost"
            " FROM quota_records WHERE provider=? AND created_at>=? AND status!='reserved'",
            (provider, since_epoch),
        ).fetchone()
        return {"requests": row["requests"], "tokens": row["tokens"], "cost": row["cost"]}

    def total_spend_since(self, since_epoch: float) -> float:
        """All-provider committed spend (USD) since an epoch — the global budget's
        meter. Free/local rows contribute $0, so this equals paid-cloud + rented
        spend. Excludes uncommitted reservations."""
        row = self._conn.execute(
            "SELECT COALESCE(SUM(act_cost_usd),0) AS cost"
            " FROM quota_records WHERE created_at>=? AND status!='reserved'",
            (since_epoch,),
        ).fetchone()
        return float(row["cost"])

    # -------------------------------------------------------------- settings
    @_synchronized
    def set_setting(self, key: str, value: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO settings(key, value, updated_at) VALUES(?,?,?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, json.dumps(value), _now()),
        )
        self._conn.commit()

    def get_setting(self, key: str) -> Optional[dict[str, Any]]:
        row = self._conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else None


def _snippet(text: Optional[str], query: str, width: int = 120) -> str:
    if not text:
        return ""
    idx = text.lower().find(query.lower())
    if idx < 0:
        return text[:width]
    start = max(0, idx - width // 2)
    return text[start : start + width]
