import sqlite3

import pytest

from agentconnect.common.memory import SharedMemory


class _FlakyCommitConn:
    """Wraps a real sqlite3 connection and raises on the first commit() only,
    forwarding everything else (execute, rollback, in_transaction, ...) to the
    real connection. sqlite3.Connection's methods are read-only on the
    instance, so a plain monkeypatch of ``.commit`` isn't possible -- this
    proxy stands in for the whole connection instead."""

    def __init__(self, real):
        self._real = real
        self._raised = False

    def commit(self):
        if not self._raised:
            self._raised = True
            raise sqlite3.OperationalError("simulated commit failure")
        return self._real.commit()

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_synchronized_rolls_back_on_raised_commit():
    # Regression: @_synchronized held the lock across execute->commit but did
    # not roll back on an exception, so a commit that raises (e.g. SQLITE_BUSY
    # under cross-process contention) left the shared connection mid-
    # transaction with the failed writer's DML still pending. The NEXT writer
    # to take the lock would then have its own commit() durably flush BOTH the
    # orphaned write and its own -- a ghost write the failed caller believes
    # never happened.
    mem = SharedMemory()
    tid = mem.create_task({"task": "demo"})

    real_conn = mem._conn
    mem._conn = _FlakyCommitConn(real_conn)
    with pytest.raises(sqlite3.OperationalError):
        mem.append_log(tid, "first message that should not land", level="info")
    # The failed write must have been rolled back, not left pending.
    assert mem._conn.in_transaction is False

    mem.append_log(tid, "second message", level="info")
    messages = [row["message"] for row in mem.get_log_slice(tid)]
    # Only the successful write is visible -- the failed one was NOT silently
    # flushed piggybacking on the next writer's commit.
    assert messages == ["second message"]


def test_artifact_chunked_read_paginates():
    mem = SharedMemory()
    content = "x" * 10000
    aid = mem.put_artifact(None, "output", content)
    c1 = mem.read_artifact_chunk(aid, 0, 4000)
    assert c1 is not None
    assert len(c1.content) == 4000
    assert c1.total_size == 10000
    assert c1.next_offset == 4000
    c2 = mem.read_artifact_chunk(aid, c1.next_offset, 4000)
    c3 = mem.read_artifact_chunk(aid, c2.next_offset, 4000)
    assert c3.next_offset is None  # fully read


def test_log_slice_filtering():
    mem = SharedMemory()
    tid = mem.create_task({"task": "demo"})
    mem.append_log(tid, "started", level="info")
    mem.append_log(tid, "boom failure", level="error")
    errs = mem.get_log_slice(tid, level="error")
    assert len(errs) == 1
    assert "boom" in errs[0]["message"]
    hits = mem.get_log_slice(tid, query="started")
    assert len(hits) == 1


def test_search_memory_returns_snippets_not_full_bodies():
    mem = SharedMemory()
    tid = mem.create_task({"task": "demo"})
    mem.put_artifact(tid, "output", "the quick brown fox " * 500)
    hits = mem.search_memory("brown fox", scope="artifacts")
    assert hits
    assert len(hits[0]["snippet"]) < 200
