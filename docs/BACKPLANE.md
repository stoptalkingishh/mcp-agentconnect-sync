# AgentConnect backplane — as built

Status of the three handoff specs against the code, as of 2026-07-10.
Read this before trusting a spec: **the specs say what we intend, this file says
what exists.**

* `docs/BACKPLANE_SPEC.md` — protocol-neutral task ledger (27 sections)
* `docs/BACKPLANE_SPEC_TEMPORAL.md` — Temporal-first durable execution (amends the above)
* `docs/BACKPLANE_SPEC_ADAPTERS.md` — memory adapter + external local-model-manager boundary

Gate: `.venv/bin/python -m pytest -q` → **496 passed** (116 of them backplane tests, all offline).

## The one rule

```
MCP tool call / HTTP request / CLI command / Linear webhook / Temporal activity
        ↓
AgentConnectService          ← the only place work state changes
        ↓
SQLite ledger + filesystem artifacts
```

Temporal makes *execution* durable. AgentConnect makes *work state* durable.
Linear makes work *visible*. Memory is *external context*, never truth. These
layers do not collapse into each other.

## Packages

| Package | What it is |
|---|---|
| `agentconnect-core` | `agentconnect.core.*` — models, storage, artifacts, service, routing, workers, handoff, execution backends, memory adapters, local-compute contract |
| `agentconnect-api` | FastAPI adapter (`agentconnect-api`) |
| `agentconnect-cli` | `agentconnect` CLI |
| `agentconnect-mcp` | MCP adapter (`agentconnect-mcp`) — 13 manager tools + 4 memory tools |
| `agentconnect-linear` | Push sync + webhook ingest |
| `agentconnect-temporal` | Workflows, activities, `TemporalExecutionBackend`, `agentconnect-temporal-worker` |

The pre-existing `agentconnect-router`, `-model-manager`, and `-runtime` packages
are untouched. They are the older MCP-router product; the backplane does not
depend on them. Per the adapters spec, `agentconnect-model-manager` is exactly
the kind of thing that should sit *behind* `LocalComputeProvider` over HTTP
rather than inside the backplane.

> ⚠️ `packages/agentconnect-temporal` (this repo) is **not** the standalone
> `agentconnect-temporal` repo at `/home/mini/agentconnect-temporal`. That one
> wraps the router-era `AgentRuntime` protocol. Same name, different layer.

## What is built

**Ledger (spec 1, phases 1–7).** Tasks, constraints, claims (leased, exclusive on
`primary_manager`, expiring), decisions (lockable; superseding a locked one needs
a live `human_owner` claim), attempts, artifacts (chunked reads that never split
a UTF-8 character), reviews (the manager-to-manager primitive), subtasks, worker
runs, approvals, external refs, inbox items, events. Deterministic handoff
summaries with **no LLM**.

**Routing (spec 1, §20).** Deterministic: six hard gates (`healthy`,
`privacy_allowed`, `capability_match`, `sandbox_supported`, `budget_allowed`,
`approval_granted`), then weighted scoring, ties broken by worker id. The
explanation is stored twice — inline on the subtask and as a
`route_explanation` artifact — including every rejected worker and the gate it
failed. `echo_worker` and `raw_model_worker` ship; a worker is a
`{harness, model, tools, sandbox, privacy_tiers}` tuple, never a model.

**Execution (spec 2).** `ExecutionBackend` seam with two implementations.
`DirectExecutionBackend` runs inline (tests, local smoke; the default, so
`pip install agentconnect-core` is a working backplane with no server).
`TemporalExecutionBackend` starts a workflow and returns. `SubtaskWorkflow`,
`ReviewWorkflow`, and `ApprovalWorkflow` are deterministic — they reference
activities by *string name* so core never enters the workflow sandbox, and they
carry artifact **ids**, never bodies. Activities are idempotent: routing is
deterministic, `run_worker` refuses to re-run a terminal subtask, a decided
approval is left alone.

**Linear (spec 1 §14–15, spec 2).** Push sync is idempotent on the `ExternalRef`.
Comments are pointers, never payloads. Webhooks parse `/agentconnect approve
cloud`, `/agentconnect approve rented-gpu max_cost=3.00`, `/agentconnect deny`,
match the approval against the route class routing actually chose, and signal the
running workflow. A Linear status change is *recorded as an event and does not
overwrite task status* — Linear is a mirror.

**Memory (spec 3, Part A).** `MemoryAdapter` with `NoopMemoryAdapter` (default),
`StaticMemoryAdapter`, `HttpMemoryAdapter`. `recall_memory`,
`capture_memory_candidate`, `record_memory_feedback`, `get_task_context_pack` on
the service, MCP, HTTP, and CLI. Capture **never promotes** — a backend that
claims it did is downgraded and logged. Visibility policy (`trusted_only`,
`include_pending`, `max_items`) is re-applied service-side, so a sloppy backend
cannot smuggle pending items into a manager's context. Memory failure is a
warning, never a failed task.

**Local compute (spec 3, Part B).** `LocalComputeProvider` +
`HttpLocalComputeProvider` (speaking `/health`, `/models`, `/models/loaded`,
`/route/estimate`, `/generate`, `/runs/{id}/cancel`) +
`LocalModelManagerWorkerAdapter`. The manager's answer is nested into the route
explanation as `local_estimate`. An outage rejects the worker at the `healthy`
gate; it never crashes the API or MCP.

## What is not built

* `TaskWorkflow`, `ManagerHandoffWorkflow`, `WorkerPipelineWorkflow` — the spec
  defers the pipeline; the other two add nothing the ledger does not already do
  synchronously.
* Rented-GPU lifecycle workflow (spec 2, phase 8).
* Real worker adapters beyond echo/raw-model: LiteLLM, OpenAI-compatible,
  Deep Agents, OpenClaw, sandboxed shell. `RawModelWorker` takes any
  `(prompt) -> text` callable, so these are adapters, not surgery.
* `WikiBrainMemoryAdapter`, `CogneeMemoryAdapter`, `GraphitiMemoryAdapter`.
* A2A (spec 1 §22 — explicitly "do not implement until the rest is stable").
* CLI remote mode (`--api-url`). The CLI runs against the DB directly.
* Contradiction *detection* between decisions. Superseding is explicit
  (`supersedes=[...]`), because deciding two strings contradict each other is
  either undecidable or nondeterministic, and §16 forbids the latter.

## Running it

```bash
# Everything, no server, no GPU, no credentials:
export AGENTCONNECT_DB_PATH=~/.agentconnect/agentconnect.db
agentconnect tasks create --title "Refactor auth" --goal "dedupe expiry"
agentconnect subtasks submit TASK_ID --title "scan" --instructions "read only"
agentconnect tasks handoff TASK_ID

# Durable execution: needs a Temporal server + at least one worker process.
temporal server start-dev &
TEMPORAL_ADDRESS=localhost:7233 agentconnect-temporal-worker &

# Adapters (point them all at the same AGENTCONNECT_DB_PATH):
agentconnect-api            # :8790
agentconnect-mcp            # stdio, or AGENTCONNECT_MCP_TRANSPORT=streamable-http

# Linear mirror:
export LINEAR_API_KEY=... LINEAR_TEAM_ID=...
agentconnect linear sync TASK_ID
```

Environment: `AGENTCONNECT_DB_PATH`, `AGENTCONNECT_ARTIFACT_DIR`,
`AGENTCONNECT_MAX_COST_USD`, `AGENTCONNECT_WORKERS`, `TEMPORAL_ADDRESS`,
`AGENTCONNECT_TEMPORAL_TASK_QUEUE`, `LINEAR_API_KEY`, `LINEAR_TEAM_ID`.

The API server and the MCP server *start* workflows; the Temporal worker
*executes* them. Without a worker process, subtasks sit in `running` forever.
