# Status — stabilization boundary

AgentConnect is past architecture-building. This file records what is true, what is
deliberately not built, and what the tests do and do not prove. It is the document to
read before proposing work.

**Current state**

| | |
|---|---|
| Checkpoint | **`28048ed`** on `origin/main` |
| Gate | `.venv/bin/python -m pytest -q` — **697 passing** |
| Execution backend | `DirectExecutionBackend` (in-process, shipped default) |
| Memory backends | none wired by default; adapters exist for WikiBrain, Cognee, Graphiti |
| Temporal | optional; `agentconnect-core` installs and runs with no workflow server |
| Linear | optional; unconfigured means completion simply fires no hook |

**Feature work is frozen.** Accept only bug fixes found by running the loop,
documentation corrections, and the small CLI ergonomics needed to run it.

## Memory boundary

Implemented and validated.

* AgentConnect controls **access**, WikiBrain controls **trust**, Cognee adds breadth,
  Graphiti adds temporal reasoning. The `ContextBuilder` decides what a manager or
  worker actually sees.
* `trusted` is the authority signal. `status == "promoted"` is **not** authority. A
  missing `trusted` means untrusted — it fails closed. The verdict may only downgrade.
* **Only the trusted authority enforces `trusted_only`.** Retrieval backends may return
  untrusted breadth; AgentConnect labels, ranks, and filters *after* retrieval. Passing
  `trusted_only` to a non-authoritative engine silently erases breadth and produces a
  falsely reassuring empty context. This is a correctness rule, not a style preference.
* Scopes are resolved broadest-first (`global`, `project:`, `repo:`, `task:`, and
  `manager:`/`worker:`/`model:` where a profile declares them). An unresolvable scope is
  dropped and *reported*, never sent empty.

## Proprietary-agent loop

Implemented and validated end to end, by an automated test and by a manual dogfood run
(see `docs/OPERATOR_GUIDE.md`).

`launch` prepares a workspace, instructions, a claim, and a scoped session token.
`shell` runs the agent in a sanitized environment. Durable work enters the ledger, the
audit reads it without writing, and the operator completes.

The five rules the loop depends on are the **operational contract** in
`docs/BACKPLANE.md`. Each names the code that enforces it and the test that keeps it
enforced.

## Known test-fidelity limits

Worth stating plainly, because a green suite invites more confidence than it has earned.

* **No test exercises real WikiBrain over real HTTP.** `tests/test_agent_loop_e2e.py`
  runs a real HTTP server on a real port that serves *canned* responses — it proves the
  adapter's httpx path, not the ledger. `tests/test_wikibrain_integration.py` drives real
  `wiki.api` **in-process** through a transport shim — it proves the semantics, not the
  wire. Nothing has both halves. Closing that needs `wiki serve`, which belongs to
  WikiBrain.
* **Cognee and Graphiti are exercised only through transport doubles.** Field names and
  shapes are asserted; no real service has ever answered.
* **Temporal is tested against the in-process time-skipping test server**, never a
  deployed cluster.
* **The compliance layer is not a sandbox.** It makes AgentConnect the normal path and
  makes bypasses visible. It does not contain a hostile process. An agent that edits its
  own environment, or opens the SQLite file directly, is stopped by nothing here. That is
  the documented scope, not an oversight.

## Known deferred work

* `wiki serve` — WikiBrain's HTTP transport. **Deferred, and not AgentConnect's task.**
  Tracked here only as a known integration gap. Do not build it from this side.
* Container / microVM isolation for `agentconnect shell` (the `--container` seam is
  designed for and deliberately unbuilt).
* `TaskWorkflow`, `ManagerHandoffWorkflow`, `WorkerPipelineWorkflow`.
* Mem0 / Supermemory adapters; soft user-preference memory. Both explicitly excluded.
* Contradiction *detection* between promoted claims.

## What would reopen work

Only these, and each needs a concrete reproduction:

1. a bug found by running the loop;
2. a field-shape mismatch against real WikiBrain;
3. a trust or scope mismatch;
4. a migration issue.
