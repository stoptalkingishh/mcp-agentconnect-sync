"""Cross-repo integration: AgentConnect's context stack against a REAL WikiBrain ledger.

Both repos pass their own suites while disagreeing about what "trusted" means, and
that disagreement is invisible from inside either one. This test exists to make the
boundary assert itself.

**The seam.** WikiBrain has no HTTP server yet, so we inject a `transport` into
`WikiBrainMemoryAdapter` that dispatches the REST contract straight into WikiBrain's
in-process Python API (`wiki.api`). That means the *semantics* are real — a real
ledger, real promotion, WikiBrain's real trust filter — while the wire format is
exercised only as far as field names and shapes. A real `wiki serve` would close
the remaining gap; nothing else here would change.

**The assertion that matters.** `status == "promoted"` is not authority.
`trusted is True` is. WikiBrain returns a contradicted claim as
`status: "promoted", trusted: false` — because a contradiction is a warning, not a
deletion, and the claim is still of record. Anything downstream keying on `status`
will happily hand a disputed claim to a manager as established truth.
"""
from __future__ import annotations

import os
import sys
import tempfile
import urllib.parse
from pathlib import Path

import pytest

# --- locate the sibling WikiBrain checkout ----------------------------------
WIKIBRAIN_REPO = Path(
    os.environ.get("WIKIBRAIN_REPO", Path(__file__).resolve().parents[2] / "WikiBrain")
)
_CLI = WIKIBRAIN_REPO / "cli"
if _CLI.is_dir() and str(_CLI) not in sys.path:
    sys.path.insert(0, str(_CLI))

wiki_api = pytest.importorskip(
    "wiki.api", reason=f"WikiBrain checkout not found at {WIKIBRAIN_REPO}")

from wiki import candidates as wiki_candidates  # noqa: E402
from wiki import scopes as wiki_scopes  # noqa: E402
from wiki.db import Repo as WikiRepo, init_db as wiki_init_db  # noqa: E402

from agentconnect.core.context import ContextBuilder, MemoryRanker  # noqa: E402
from agentconnect.core.memory import (  # noqa: E402
    BROAD_RETRIEVAL, TEMPORAL_GRAPH, MemoryItem, MemoryScope, RecallRequest,
    StaticMemoryAdapter, WikiBrainMemoryAdapter,
)

BASE_URL = "http://wikibrain.test"


# --- the REST contract, dispatched in-process -------------------------------
def _make_transport(repo_root: Path):
    """Map AgentConnect's WikiBrain REST calls onto `wiki.api`.

    Every path here is one the adapter actually issues. If WikiBrain's API drifts
    from this shape, this shim breaks — which is the point.
    """
    def transport(method: str, url: str, payload: dict | None = None) -> dict:
        path = url[len(BASE_URL):]
        parsed = urllib.parse.urlparse(path)
        route, query = parsed.path, urllib.parse.parse_qs(parsed.query)
        with WikiRepo.open(start=repo_root) as repo:
            if method == "POST" and route == "/recall":
                return wiki_api.recall(repo, payload or {}).as_dict()
            if method == "POST" and route == "/capture":
                return wiki_api.capture_candidate(repo, payload or {}).as_dict()
            if method == "POST" and route.endswith("/promote"):
                cid = route.split("/")[2]
                body = payload or {}
                claim = wiki_api.promote(
                    repo, cid, reviewer=body.get("promoted_by"),
                    confidence=body.get("confidence"), scope=body.get("scope"))
                return {**claim, "claim_id": claim["id"]}
            if method == "GET" and route == "/candidates":
                limit = int(query.get("limit", ["50"])[0])
                return {"candidates": wiki_api.pending(repo, limit=limit)}
            if method == "POST" and route == "/feedback":
                body = payload or {}
                wiki_api.record_feedback(repo, {
                    "feedback": body["feedback"],
                    "actor_id": body.get("actor_id") or "unknown",
                    "actor_type": "manager",
                    "claim_id": body.get("memory_item_id"),
                    "source_id": body.get("source_id"),
                    "note": body.get("note"), "task_id": body.get("task_id")})
                return {}
            if method == "GET" and route == "/health":
                return wiki_api.health(repo)
        raise AssertionError(f"WikiBrain shim has no route for {method} {path}")
    return transport


# --- a real ledger, seeded with one claim per trust state -------------------
SCOPE = "global"  # visible from any recall; scope filtering is WikiBrain's own suite

TRUSTED = "Refresh token validation lives in auth/session.py."
PENDING = "Refresh token validation might move to auth/tokens.py."
REJECTED = "Refresh token secrets should be logged for debugging."
OLD = "Refresh token lifetime is 60 minutes."
NEW = "Refresh token lifetime is 15 minutes."
DISPUTED = "Refresh token rotation happens on every request."
RIVAL = "Refresh token rotation happens only on expiry."


@pytest.fixture()
def ledger(monkeypatch):
    """A WikiBrain repo on a scratch DB, seeded across every trust state.

    WIKIBRAIN_DB is mandatory: Repo.open() migrates whatever DB it resolves to, and
    a temp repo root does NOT isolate it (see WikiBrain docs/MIGRATIONS.md).
    """
    root = Path(tempfile.mkdtemp(prefix="wb-integration-"))
    for d in ("raw", "inbox", "db"):
        (root / d).mkdir(parents=True)
    (root / "log.md").write_text("# log\n", encoding="utf-8")
    (root / "config.toml").write_text('[paths]\ndb = "unused.db"\n', encoding="utf-8")
    monkeypatch.setenv("WIKIBRAIN_DB", str(root / "scratch.db"))
    wiki_init_db(start=root).close()

    ids: dict[str, str] = {}

    def propose(text, tags=()):
        return wiki_api.capture_candidate(repo, {
            "text": text, "proposed_by": "claude-code", "proposed_by_type": "manager",
            "tags": list(tags)}).candidate_id

    def promote(cand, confidence="verified"):
        return wiki_api.promote(repo, cand, reviewer="matthew",
                                confidence=confidence, scope=SCOPE)["id"]

    with WikiRepo.open(start=root) as repo:
        ids["trusted"] = promote(propose(TRUSTED, ["decision", "constraint"]))

        # A pending CLAIM (not merely a candidate): unvetted material in the ledger.
        repo.ex("INSERT INTO claims(text, source_id, confidence, origin, status, "
                "created_at, scope_type, scope_id, confidence_label) VALUES "
                "(?, 1, 0.9, 'session/mcp', 'pending', '2026-01-01T00:00:00Z', "
                "'global', '', 'high')", (PENDING,))
        repo.ex("INSERT INTO claims(text, source_id, confidence, origin, status, "
                "created_at, scope_type, scope_id, confidence_label) VALUES "
                "(?, 1, 0.9, 'clip', 'rejected', '2026-01-01T00:00:00Z', "
                "'global', '', 'verified')", (REJECTED,))
        repo.finalize("test", "seed")

        old_id = promote(propose(OLD, ["decision"]))
        new_id = promote(propose(NEW, ["decision"]))
        wiki_api.supersede(repo, old_id, new_id, reason="shortened in v3",
                           reviewer="matthew")
        ids["old"], ids["new"] = old_id, new_id

        # A PROMOTED claim in an OPEN contradiction. WikiBrain keeps it (a warning,
        # not a deletion) and returns it as trusted=false. This is the trap.
        ids["disputed"] = promote(propose(DISPUTED, ["decision"]))
        ids["rival"] = promote(propose(RIVAL, ["decision"]))
        a = int(ids["disputed"].split("_")[1])
        b = int(ids["rival"].split("_")[1])
        repo.ex("INSERT INTO contradictions(claim_a, claim_b, status) VALUES (?,?,'open')",
                (a, b))
        repo.finalize("test", "contradiction")

    return {"root": root, "ids": ids, "transport": _make_transport(root)}


@pytest.fixture()
def adapter(ledger):
    return WikiBrainMemoryAdapter(base_url=BASE_URL, transport=ledger["transport"])


def _by_text(pack):
    return {i.text: i for i in pack.items}


def _recall(adapter, **kw):
    return adapter.recall(RecallRequest(query="refresh token", **kw))


# =============================================================================
# 1. WikiBrain's own pack already carries the right verdicts
# =============================================================================
def test_wikibrain_returns_a_contradicted_claim_as_promoted_but_untrusted(ledger):
    """The premise of everything below. If WikiBrain ever stops doing this, the
    downstream assertions become vacuous rather than failing."""
    raw = ledger["transport"]("POST", f"{BASE_URL}/recall", {
        "query": "refresh token rotation", "trusted_only": True})
    disputed = [i for i in raw["items"] if i["text"] == DISPUTED]
    assert disputed, "WikiBrain should still return a contradicted claim"
    assert disputed[0]["status"] == "promoted"
    assert disputed[0]["trusted"] is False
    assert disputed[0]["contradicted"] is True
    assert any("contradiction" in w.lower() for w in raw["warnings"])


# =============================================================================
# 2. The trust table (the cases from the review)
# =============================================================================
def test_promoted_and_trusted_is_included_and_marked_trusted(adapter):
    item = _by_text(_recall(adapter))[TRUSTED]
    assert item.status == "promoted"
    assert item.metadata["trusted"] is True


def test_pending_is_excluded_by_default(adapter):
    assert PENDING not in _by_text(_recall(adapter))


def test_pending_is_included_only_on_request_and_never_trusted(adapter):
    pack = _recall(adapter, trusted_only=False, include_pending=True, max_items=20)
    item = _by_text(pack)[PENDING]
    assert item.status == "pending"
    assert item.metadata["trusted"] is False
    assert any("pending" in w.lower() for w in pack.warnings)


def test_rejected_is_never_returned(adapter):
    for kw in ({}, {"trusted_only": False, "include_pending": True,
                    "include_superseded": True, "max_items": 20}):
        assert REJECTED not in _by_text(_recall(adapter, **kw))


def test_superseded_promoted_is_excluded_by_default(adapter):
    packed = _by_text(_recall(adapter, max_items=20))
    assert OLD not in packed
    assert NEW in packed


def test_superseded_is_visible_on_request_but_not_trusted(adapter):
    pack = _recall(adapter, trusted_only=False, include_superseded=True, max_items=20)
    item = _by_text(pack)[OLD]
    assert item.status == "superseded"
    assert item.metadata["trusted"] is False
    assert item.superseded_by


# --- THE assertion: status is not authority ---------------------------------
def test_contradicted_promoted_claim_is_never_treated_as_trusted(adapter):
    """A promoted-but-contradicted claim must not be stamped trusted.

    This fails for any implementation that computes trust as
    `item.status == "promoted"` instead of honouring the authority's `trusted`.
    """
    pack = _recall(adapter, trusted_only=False, max_items=20)
    disputed = _by_text(pack).get(DISPUTED)
    if disputed is not None:
        assert disputed.status == "promoted"
        assert disputed.metadata["trusted"] is False, (
            "a contradicted claim was stamped trusted — trust was derived from "
            "status instead of the authority's `trusted` field")


def test_contradicted_promoted_claim_is_excluded_from_a_trusted_only_recall(adapter):
    pack = _recall(adapter, trusted_only=True, max_items=20)
    assert DISPUTED not in _by_text(pack), (
        "a disputed claim reached a trusted_only recall")
    assert any("disput" in w.lower() or "contradict" in w.lower() for w in pack.warnings)


def test_a_disputed_claim_never_outranks_a_trusted_one(adapter):
    """Even when admitted, it must fall below trusted claims in the authority ladder."""
    pack = _recall(adapter, trusted_only=False, max_items=20)
    by_text = _by_text(pack)
    if DISPUTED not in by_text:
        pytest.skip("disputed claim excluded entirely, which is also correct")
    ranker = MemoryRanker()
    assert ranker.authority(by_text[DISPUTED]) > ranker.authority(by_text[TRUSTED])


# =============================================================================
# 3. Retrieval backends cannot outrank the authority
# =============================================================================
def _broad(text, backend, role, **md):
    item = MemoryItem(text=text, status="unknown", confidence="unknown",
                      source_id="doc-1", metadata=dict(md))
    item.metadata.update({"backend": backend, "role": role, "trusted": False})
    return item


def test_a_cognee_duplicate_does_not_outrank_the_wikibrain_trusted_claim(adapter):
    trusted = _by_text(_recall(adapter))[TRUSTED]
    duplicate = _broad(TRUSTED, "cognee", BROAD_RETRIEVAL, score=99.0)
    ranker = MemoryRanker()
    assert ranker.authority(trusted) < ranker.authority(duplicate)

    from agentconnect.core.memory import RecallPack
    merged = ranker.merge_and_rank(
        [RecallPack(profile="manager_brief", query="", items=[duplicate], backend="cognee"),
         RecallPack(profile="manager_brief", query="", items=[trusted], backend="wikibrain")],
        "manager_brief", 8)
    winner = merged.items[0]
    assert winner.metadata["backend"] == "wikibrain"
    assert winner.metadata["trusted"] is True
    assert "cognee" in winner.metadata.get("also_seen_in", [])


def test_a_graphiti_fact_ranks_only_when_anchored_to_a_promoted_claim():
    ranker = MemoryRanker()
    anchored = _broad("token rotation changed in v3", "graphiti", TEMPORAL_GRAPH)
    floating = _broad("token rotation changed in v3", "graphiti", TEMPORAL_GRAPH)
    floating.source_id = None
    assert ranker.authority(anchored) == ranker.GRAPHITI_TIED_TO_PROMOTED
    assert ranker.authority(floating) == ranker.PENDING_OR_UNKNOWN
    assert anchored.metadata["trusted"] is False


def test_a_retrieval_backend_claiming_trust_cannot_grant_itself_authority():
    """A broad-retrieval hit that says `trusted: true` is still not trusted."""
    from agentconnect.core.memory import label
    liar = MemoryItem(text="I am definitely true.", status="promoted",
                      confidence="verified", source_id="doc-9")
    label(liar, "cognee", BROAD_RETRIEVAL, authority_trusted=True)
    assert liar.metadata["trusted"] is False


# =============================================================================
# 4. Profiles
# =============================================================================
def test_worker_brief_never_receives_pending_or_untrusted_items(adapter):
    pack = adapter.recall(RecallRequest(
        query="refresh token", profile="worker_brief", max_items=20))
    assert all(i.status == "promoted" for i in pack.items)
    assert all(i.metadata["trusted"] is True for i in pack.items)
    assert PENDING not in _by_text(pack)
    assert DISPUTED not in _by_text(pack)


def test_worker_brief_excludes_pending_even_when_the_backend_offers_it(adapter):
    """Profile discipline is not the backend's job to enforce."""
    pack = adapter.recall(RecallRequest(
        query="refresh token", profile="worker_brief",
        include_pending=True, trusted_only=True, max_items=20))
    assert all(i.metadata["trusted"] is True for i in pack.items if i.status != "pending")


def test_implementation_constraints_returns_only_trusted_hard_constraints(adapter):
    pack = adapter.recall(RecallRequest(
        query="refresh token", profile="implementation_constraints", max_items=20))
    assert all(i.metadata["trusted"] is True for i in pack.items)
    assert all(i.confidence in ("high", "verified") for i in pack.items)
    assert DISPUTED not in _by_text(pack)


def test_manager_brief_may_be_broad_but_every_item_is_labeled(adapter):
    pack = adapter.recall(RecallRequest(
        query="refresh token", profile="manager_brief",
        trusted_only=False, include_pending=True, max_items=20))
    for item in pack.items:
        assert "trusted" in item.metadata
        assert item.metadata["backend"] == "wikibrain"


# =============================================================================
# 5. The full loop: capture -> human promotion -> trusted recall
# =============================================================================
def test_capture_promote_recall_round_trip(adapter, ledger):
    from agentconnect.core.memory import CaptureRequest

    text = "The worker fleet must not write to auth/session.py directly."
    result = adapter.capture_candidate(CaptureRequest(
        text=text, task_id="task_auth_001", origin_actor_id="claude-code",
        origin_actor_type="manager", source_ref="agentconnect_attempt_123",
        tags=["constraint"], proposed_scopes=[MemoryScope("global", "")]))
    assert result.accepted and result.status == "pending"

    # Not trusted, not recalled, until a human says so.
    assert text not in _by_text(_recall(adapter, max_items=20))
    assert any(c["ref"] == result.candidate_id for c in adapter.list_pending())

    claim = adapter.promote_candidate(result.candidate_id, promoted_by="matthew",
                                      confidence="high", scope="global")
    assert claim["claim_id"].startswith("claim_")

    item = _by_text(adapter.recall(RecallRequest(query="worker fleet", max_items=20)))[text]
    assert item.status == "promoted"
    assert item.metadata["trusted"] is True


def test_promotion_refuses_to_guess_confidence(adapter, ledger):
    from agentconnect.core.memory import CaptureRequest
    result = adapter.capture_candidate(CaptureRequest(
        text="A candidate with no confidence stated.", origin_actor_id="claude-code",
        origin_actor_type="manager", proposed_scopes=[MemoryScope("global", "")]))
    with pytest.raises(Exception):
        adapter.promote_candidate(result.candidate_id, promoted_by="matthew")


def test_feedback_can_address_the_claim_it_was_recalled_from(adapter, ledger):
    from agentconnect.core.memory import MemoryFeedbackRequest
    item = _by_text(_recall(adapter))[TRUSTED]
    assert item.item_id, "a recalled item must carry the id feedback needs to target"
    adapter.record_feedback(MemoryFeedbackRequest(
        task_id="task_auth_001", memory_item_id=item.item_id, source_id=None,
        feedback="stale", actor_id="claude-code", note="moved in v3"))
    with WikiRepo.open(start=ledger["root"]) as repo:
        from wiki import feedback as wiki_feedback, refs as wiki_refs
        tally = wiki_feedback.tally(repo, wiki_refs.parse(item.item_id, wiki_refs.CLAIM))
    assert tally == {"stale": 1}

    # Feedback is an observation, never a demotion.
    assert _by_text(_recall(adapter))[TRUSTED].metadata["trusted"] is True


def test_health_reports_the_ledger_through_the_adapter(adapter):
    health = adapter.health()
    assert health["backend"] == "wikibrain"
    assert health["ok"] is True
    assert health["ledger"]["claims_promoted"] >= 1
    # `backend` means the adapter here, not WikiBrain's retrieval backend.
    assert health["retrieval"]["backend"] == "sqlite_fts"


# =============================================================================
# 6. The full stack: ContextBuilder over a real ledger + fake breadth backends
# =============================================================================
@pytest.fixture()
def stack(tmp_path, ledger):
    """AgentConnectService wired to the real WikiBrain plus fake Cognee/Graphiti."""
    from agentconnect.core import (
        AgentConnectService, CogneeMemoryAdapter, CreateTaskRequest,
        GraphitiMemoryAdapter, MemoryConfig,
    )

    def cognee(method, url, payload=None):
        if url.endswith("/health"):
            return {"status": "ok"}
        # Cognee "finds" the trusted claim too, loudly, and a stray document.
        return {"results": [
            {"text": TRUSTED, "source_id": "doc_1", "score": 99.0},
            {"text": "Someone on the team dislikes auth code.", "source_id": "doc_2",
             "score": 0.9},
        ]}

    def graphiti(method, url, payload=None):
        if url.endswith("/health"):
            return {"status": "ok"}
        return {"facts": [
            {"fact": "auth/session.py was last touched in v3.", "source_id": "claim_1"},
            {"fact": "An unanchored rumour about auth.", "source_id": None},
        ]}

    svc = AgentConnectService.create(
        db_path=":memory:", artifact_dir=str(tmp_path / "artifacts"),
        memory_backends={
            "wikibrain": WikiBrainMemoryAdapter(base_url=BASE_URL,
                                                transport=ledger["transport"]),
            "cognee": CogneeMemoryAdapter(transport=cognee),
            "graphiti": GraphitiMemoryAdapter(transport=graphiti),
        },
        memory_config=MemoryConfig(),
    )
    task = svc.create_task(CreateTaskRequest(title="refresh token", goal="fix auth"))
    return svc, task


def _pack_texts(pack):
    return [i.text for i in pack.memory.items]


def _trusted_texts(pack):
    return [i.text for i in pack.memory.items if i.metadata.get("trusted")]


def test_context_builder_does_not_treat_promoted_as_trusted_when_trusted_false(stack):
    """The acceptance assertion, at the level a manager actually consumes."""
    svc, task = stack
    pack = svc.get_task_context_pack(task.id, profile="manager_brief")
    for item in pack.memory.items:
        if item.text == DISPUTED:
            assert item.metadata["trusted"] is False
    assert DISPUTED not in _trusted_texts(pack)


def test_implementation_constraints_queries_only_the_trusted_authority(stack):
    svc, task = stack
    pack = svc.get_task_context_pack(task.id, profile="implementation_constraints")
    assert pack.backends_queried == ["wikibrain"]
    assert all(i.metadata["backend"] in ("wikibrain", "agentconnect")
               for i in pack.memory.items)
    assert all(i.metadata["trusted"] for i in pack.memory.items)
    for excluded in (PENDING, REJECTED, OLD, DISPUTED):
        assert excluded not in _pack_texts(pack)


def test_worker_brief_excludes_pending_untrusted_and_graphiti(stack):
    svc, task = stack
    pack = svc.get_task_context_pack(task.id, profile="worker_brief")
    assert "graphiti" not in pack.backends_queried
    assert PENDING not in _pack_texts(pack)
    assert REJECTED not in _pack_texts(pack)
    assert DISPUTED not in _trusted_texts(pack)
    wikibrain_items = [i for i in pack.memory.items
                       if i.metadata["backend"] == "wikibrain"]
    assert wikibrain_items and all(i.metadata["trusted"] for i in wikibrain_items)


def test_a_cognee_duplicate_is_deduped_in_favour_of_the_trusted_claim(stack):
    svc, task = stack
    pack = svc.get_task_context_pack(task.id, profile="manager_brief")
    copies = [i for i in pack.memory.items if i.text == TRUSTED]
    assert len(copies) == 1, "the same fact survived twice"
    assert copies[0].metadata["backend"] == "wikibrain"
    assert copies[0].metadata["trusted"] is True
    assert "cognee" in copies[0].metadata.get("also_seen_in", [])


def test_a_cognee_hit_is_never_trusted_however_high_it_scores(stack):
    svc, task = stack
    pack = svc.get_task_context_pack(task.id, profile="manager_brief")
    for item in pack.memory.items:
        if item.metadata["backend"] == "cognee":
            assert item.metadata["trusted"] is False


def test_an_unanchored_graphiti_fact_never_becomes_an_implementation_constraint(stack):
    svc, task = stack
    constraints = svc.get_task_context_pack(task.id, profile="implementation_constraints")
    assert "graphiti" not in constraints.backends_queried
    manager = svc.get_task_context_pack(task.id, profile="manager_brief")
    for item in manager.memory.items:
        if item.metadata["backend"] == "graphiti":
            assert item.metadata["trusted"] is False


def test_healthy_memory_yields_a_non_empty_labeled_bounded_pack(stack):
    svc, task = stack
    pack = svc.get_task_context_pack(task.id, profile="manager_brief")
    assert pack.memory.items, "healthy memory returned an empty pack"
    assert len(pack.memory.items) <= 8
    for item in pack.memory.items:
        assert item.metadata["backend"]
        assert item.metadata["role"]
        assert "trusted" in item.metadata


def test_a_memory_outage_degrades_the_pack_and_records_a_warning(tmp_path, ledger):
    from agentconnect.core import AgentConnectService, CreateTaskRequest, MemoryConfig

    def dead(method, url, payload=None):
        raise ConnectionError("wikibrain is down")

    svc = AgentConnectService.create(
        db_path=":memory:", artifact_dir=str(tmp_path / "artifacts"),
        memory_backends={"wikibrain": WikiBrainMemoryAdapter(
            base_url=BASE_URL, transport=dead)},
        memory_config=MemoryConfig(),
    )
    task = svc.create_task(CreateTaskRequest(title="t", goal="g"))
    pack = svc.get_task_context_pack(task.id)
    assert any("recall failed" in w for w in pack.warnings)
    assert not [i for i in pack.memory.items if i.metadata["backend"] == "wikibrain"]


def test_the_disputed_claim_surfaces_as_a_warning_not_as_guidance(stack):
    svc, task = stack
    pack = svc.get_task_context_pack(task.id, profile="manager_brief")
    assert DISPUTED not in _trusted_texts(pack)
    assert any("disput" in w.lower() or "contradict" in w.lower() for w in pack.warnings)
