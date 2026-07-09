# AgentConnect backplane — as built

Status of the five handoff specs against the code, as of 2026-07-10.
Read this before trusting a spec: **the specs say what we intend, this file says
what exists.**

* `docs/BACKPLANE_SPEC.md` — protocol-neutral task ledger (27 sections)
* `docs/BACKPLANE_SPEC_TEMPORAL.md` — Temporal-first durable execution (amends the above)
* `docs/BACKPLANE_SPEC_ADAPTERS.md` — memory adapter + external local-model-manager boundary
* `docs/BACKPLANE_SPEC_MEMORY_STACK.md` — Temporal + WikiBrain + Cognee + Graphiti
* `docs/BACKPLANE_SPEC_COMPLIANCE.md` — easiest useful Level 4: launch, shell, audit

Gate: `.venv/bin/python -m pytest -q` → **647 passed**, all offline.

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

And within memory:

> AgentConnect controls access. WikiBrain controls trust. Cognee improves
> breadth. Graphiti improves temporal reasoning. Temporal runs workflows.
> Linear shows humans what matters.

And for the agents themselves:

> Agents may think and work inside their own harness. But durable work must enter
> AgentConnect. **If it is not recorded in AgentConnect, it did not happen.**

## Packages

| Package | What it is |
|---|---|
| `agentconnect-core` | `agentconnect.core.*` — models, storage, artifacts, service, routing, workers, handoff, execution backends, memory adapters, local-compute contract |
| `agentconnect-api` | FastAPI adapter (`agentconnect-api`) |
| `agentconnect-cli` | `agentconnect` CLI |
| `agentconnect-mcp` | MCP adapter (`agentconnect-mcp`) — 13 manager tools + 4 memory tools |
| — | compliance layer lives in `core.sessions`, `core.workspace`, `core.audit` |
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

**Memory (spec 3 Part A, spec 4).** `MemoryAdapter` with `NoopMemoryAdapter`
(default), `StaticMemoryAdapter`, `HttpMemoryAdapter`, and the three real
backends: `WikiBrainMemoryAdapter` (a `TrustedMemoryAdapter`),
`CogneeMemoryAdapter` and `GraphitiMemoryAdapter` (both `IndexingMemoryAdapter`).
Each carries a `role` — `trusted_authority`, `broad_retrieval`, `temporal_graph`
— and every recalled item is stamped with `backend` / `role` / `trusted`, so a
Cognee search hit can never be read as a recorded decision.

`ContextBuilder` is the whole read path. `MemoryRouter` decides which backends a
*profile* may even ask (nine profiles; `implementation_constraints` asks only the
trusted authority, `worker_brief` never reaches the temporal graph).
`MemoryRanker` merges, dedupes by normalized text, and orders by a fixed
authority: ledger > WikiBrain verified > WikiBrain promoted > Graphiti tied to a
promoted claim > Cognee > pending/unknown. A retrieval engine surfacing a
sentence three times never outranks a librarian promoting it once. Ledger truth
(task constraints, locked decisions, configured hard policies) and recalled
memory live in one ranked list but are never confused: `memory_is_external_context`
rides on every pack.

> ### ⚠️ Only the trusted authority enforces `trusted_only`
>
> Retrieval backends such as Cognee and Graphiti may return untrusted breadth.
> AgentConnect labels, ranks, and filters those results **after** retrieval.
> Passing `trusted_only` directly to non-authoritative retrieval engines is
> **incorrect**: they have no notion of promotion, so the flag silently erases
> breadth and produces falsely reassuring empty context. This is deliberate, not
> an oversight — do not "clean it up" into a bug.
>
> Enforced by `ContextBuilder` (`is_authority and not include_pending`) and pinned
> by `test_trusted_only_is_never_pushed_down_to_a_retrieval_engine`.

> ### ⚠️ `status: "promoted"` is not authority. `trusted` is.
>
> WikiBrain returns a claim in an open contradiction as `status: "promoted",
> trusted: false` — a contradiction is a warning, not a deletion, so the claim
> stays of record. Anything keying on `status` will hand a disputed claim to a
> manager as established truth. A missing `trusted` therefore means **untrusted**
> and is never inferred from the status. The verdict may only ever *downgrade*: a
> retrieval engine sending `trusted: true` cannot grant itself authority.
>
> Enforced by `memory.label(..., authority_trusted=...)` and `memory.is_disputed`;
> pinned by `tests/test_wikibrain_integration.py` against a real WikiBrain ledger.

Write path: agent → `capture_memory_candidate` → **pending** in WikiBrain →
human/librarian `promote_memory_candidate` → promoted claim → fanned out to
Cognee and Graphiti. Capture **never promotes** — a backend that claims it did is
downgraded and logged; Cognee and Graphiti *refuse* agent writes outright.
Promotion is deliberately absent from the MCP surface, so an agent cannot promote
its own suggestion. An indexing failure does not undo a promotion: the trusted
authority is the record, the indexes are caches of it.

How each kind of agent gets memory:

* **Managers** (Claude Code, Codex, Linear Agent — proprietary, unmodifiable)
  **pull**: they call the `get_task_context_pack` MCP tool. They cannot see
  WikiBrain, Cognee, or Graphiti at all; only the AgentConnect MCP server is
  mounted for them. This is the entire reason the MCP adapter exists.
* **Workers** (bounded, usually with no MCP client) get memory **pushed**: the
  `recall_context` activity builds a `worker_brief` and attaches it to
  `subtask.metadata["context_pack"]` before `run_worker` runs.

Visibility policy (`trusted_only`, `include_pending`, `include_superseded`,
`max_items`) is re-applied service-side, so a sloppy backend cannot smuggle
pending items into a manager's context. Memory is queried only from Temporal
*activities*, never workflow code, and a memory outage degrades the pack with a
warning rather than failing the subtask. Soft user-preference memory is
deliberately **not** in the core: hard preferences are AgentConnect config
(`hard_policies`) or scoped WikiBrain claims.

Config lives in `config/memory.yaml` (`AGENTCONNECT_MEMORY_CONFIG`), backend URLs
in `WIKIBRAIN_URL`, `COGNEE_URL`, `GRAPHITI_URL`. Absent config means memory is
simply off and context packs are task state only.

**Local compute (spec 3, Part B).** `LocalComputeProvider` +
`HttpLocalComputeProvider` (speaking `/health`, `/models`, `/models/loaded`,
`/route/estimate`, `/generate`, `/runs/{id}/cancel`) +
`LocalModelManagerWorkerAdapter`. The manager's answer is nested into the route
explanation as `local_estimate`. An outage rejects the worker at the `healthy`
gate; it never crashes the API or MCP.

**Compliance layer (spec 5).** `agentconnect launch claude --task task_123 --claim`
prepares a managed session: it verifies the task, materializes a workspace (a git
worktree on `agentconnect/task_123/<slug>` where possible; otherwise bind, copy,
or empty), claims the task, mints a scoped token, and writes `AGENTCONNECT.md`,
`CLAUDE.md`/`CODEX.md`, `.env.agentconnect` (0600), `.mcp.json`, and
`workspace.json`. Then `agentconnect shell --task task_123 -- claude` runs the
agent in that workspace.

This is a **compliance wrapper, not a hardened sandbox**. It makes AgentConnect
the normal path and makes bypasses visible; it does not stop a hostile agent. The
`--container` seam exists in `shell` and is deliberately unimplemented.

*Environment.* Sanitization is an **allowlist** (`PATH HOME SHELL TERM LANG LC_ALL
USER LOGNAME TMPDIR TZ` plus the `AGENTCONNECT_*` session vars), never a bare
`env -i` — an empty environment breaks too many tools to be useful. Backend
credentials are not removed so much as never copied in, because they were never on
the list. The denylist polices the one explicit opt-in path
(`AGENTCONNECT_SHELL_ALLOW_ENV`), where an operator could otherwise re-admit a
credential by name; anything matching `*_API_KEY`, `*_SECRET`, `*_TOKEN` and
friends is refused there with an error rather than quietly obeyed.

*Credentials.* An agent gets exactly one: a short-lived `act_…` session token,
scoped to its session, entity, and mode. Manager mode buys ten actions, reviewer
mode six, readonly four. `promote_memory_candidate`, `temporal_signal`,
`secrets_read`, `grant_approval`, and `complete_task` are in **no** mode's list,
so the deny is structural rather than a special case. Only the token's SHA-256 is
stored; the plaintext exists just long enough to write the env file. Ending the
shell revokes it, so a leaked `.env.agentconnect` is inert.

*Ids are inferred.* `AGENTCONNECT_TASK_ID` and `AGENTCONNECT_MANAGER_ID` are in the
environment, and every MCP tool falls back to them. An agent that cannot mistype an
id cannot record its work against the wrong task.

*Audit is the teeth.* `agentconnect audit task_123` asks one question several ways:
*you did something — where is it in the ledger?* It checks the workspace, the
session, the claim, an attempt recorded during **this** session, changed files
registered as artifacts (structurally via `metadata["files"]`, or named in an
artifact summary), resolved subtasks, completed reviews, a decision behind any
durable change, a fresh handoff, Linear agreement, and status consistency. Memory
capture is *advisory* — it warns, it never blocks, because memory is optional.

The audit **writes nothing**. That is load-bearing: `get_handoff_summary` persists
the summary as a side effect, so auditing through it would repair the staleness it
reports, and the first audit would fail where the second passed. An audit that
changes what it measures is not an audit. `complete_task` therefore regenerates the
handoff *first*, then audits.

*Completion.* `complete` runs the audit, marks the ledger succeeded, and only then
fires completion hooks — which is how Linear learns. A hook that raises is logged,
not fatal: a tracker outage cannot undo a completion. `/agentconnect complete` in a
Linear comment routes through the same audit, and moving the Linear issue to Done
records an event and changes nothing. The issue body says so:
`**Canonical status:** AgentConnect-managed`.


## What is not built

* `TaskWorkflow`, `ManagerHandoffWorkflow`, `WorkerPipelineWorkflow` — the spec
  defers the pipeline; the other two add nothing the ledger does not already do
  synchronously.
* Rented-GPU lifecycle workflow (spec 2, phase 8).
* Real worker adapters beyond echo/raw-model: LiteLLM, OpenAI-compatible,
  Deep Agents, OpenClaw, sandboxed shell. `RawModelWorker` takes any
  `(prompt) -> text` callable, so these are adapters, not surgery.
* Container/microVM isolation for `agentconnect shell` (spec 5 §16 — the `--container`
  seam is designed for, and deliberately not built; host shell only).
* Mem0 / Supermemory adapters (spec 4 §15 — explicitly not in the core stack yet).
* Soft user-preference memory (spec 4 §3 — deliberately excluded).
* Contradiction *detection* between promoted claims. `memory_comment("conflict", ...)`
  can render one; nothing raises one, for the same reason decisions do not.
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
`AGENTCONNECT_TEMPORAL_TASK_QUEUE`, `LINEAR_API_KEY`, `LINEAR_TEAM_ID`,
`AGENTCONNECT_MEMORY_CONFIG`, `WIKIBRAIN_URL`, `COGNEE_URL`, `GRAPHITI_URL`.

```bash
# Memory is opt-in. With no config file and no *_URL set, packs are task state only.
agentconnect memory pending                 # the librarian's queue
agentconnect memory promote candidate_1 --by matthew   # human-only; not an MCP tool

# Run a proprietary agent through the backplane:
agentconnect launch claude --task task_123 --claim --repo ~/code/myrepo
agentconnect shell --task task_123 -- claude
agentconnect audit task_123
agentconnect complete task_123 --by matthew   # refuses unless the audit passes
```

The API server and the MCP server *start* workflows; the Temporal worker
*executes* them. Without a worker process, subtasks sit in `running` forever.
