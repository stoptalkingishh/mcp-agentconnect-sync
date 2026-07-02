from agentconnect.common.memory import SharedMemory


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
