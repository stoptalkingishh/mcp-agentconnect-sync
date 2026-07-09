# AgentConnect Implementation Handoff — Temporal-First Durable Agent Backplane

> Authored by the project owner, 2026-07-10, as an amendment to `docs/BACKPLANE_SPEC.md`.
> Where the two disagree, **this one is newer**. The ledger design, data model, Linear mapping,
> routing procedure, and worker adapter interface carry over unchanged; what changes is *how work
> executes*. See `docs/BACKPLANE.md` for the as-built status.

## Goal

Implement AgentConnect as a **Temporal-first, local-first agent work backplane**.

Manager harnesses (Claude Code, Codex, Linear Agent, OpenClaw, future frontier agents) attach to
the same durable task state, claim work, delegate subtasks, request reviews, wait for approvals,
run workers, store artifacts, and resume from compact handoff summaries.

Temporal is the durable execution engine for long-running workflows, retries, approval waits,
worker calls, review loops, and failure recovery.

AgentConnect remains the canonical task/artifact/policy state. Linear remains the human-visible
issue and approval surface.

## Core architecture

```
Human / Linear
      ↓
AgentConnect Linear adapter
      ↓
AgentConnect API / MCP / CLI
      ↓
AgentConnectService
      ↓
Temporal workflows
      ↓
Activities:
  - route subtask
  - run worker
  - update Linear
  - save artifact
  - request approval
  - record attempt
  - complete review
```

## Source-of-truth boundaries

**AgentConnect owns:** tasks, subtasks, manager claims, review tickets, decisions, attempts,
artifacts, route history, worker runs, handoff summaries, approval records, external Linear refs.

**Temporal owns:** durable workflow execution, workflow timers, retries, activity scheduling,
long-running waits, workflow signals, queries, updates, crash recovery.

**Linear owns:** human visibility, issue workflow, comments, approvals, assignments,
labels/status.

**Worker harnesses own:** execution internals, temporary tool context, model-specific loops,
ephemeral local state.

## Non-goals

* Do not make Temporal the source of truth for task records.
* Do not store all artifacts inside Temporal workflow history.
* Do not make Linear the canonical database.
* Do not make MCP the only interface.
* Do not build direct free-form manager-to-manager chat first.
* Do not make memory/knowledge retrieval part of this first implementation.

## Required package structure

```
packages/
  agentconnect-core/      models.py service.py storage.py artifacts.py claims.py
                          reviews.py decisions.py attempts.py handoff.py
  agentconnect-api/       app.py routes_tasks.py routes_subtasks.py routes_reviews.py
                          routes_artifacts.py routes_linear.py routes_temporal.py
  agentconnect-mcp/       server.py tools.py
  agentconnect-cli/       main.py
  agentconnect-linear/    client.py mapping.py sync.py webhooks.py
  agentconnect-temporal/  client.py
                          workflows/{task,subtask,review,approval,manager_handoff,worker_pipeline}_workflow.py
                          activities/{route_subtask,run_worker,save_artifact,update_linear,
                                      request_approval,record_attempt,complete_review,notify_manager}.py
  agentconnect-router/    routing.py privacy.py budget.py worker_registry.py
                          worker_adapters/{base,echo,raw_model,litellm,local_model_manager}.py
```

## Storage

SQLite first. `~/.agentconnect/agentconnect.db`, `~/.agentconnect/artifacts/`. Overrides:
`AGENTCONNECT_DB_PATH`, `AGENTCONNECT_ARTIFACT_DIR`.

Temporal stores workflow execution history. AgentConnect stores task state and artifacts.
**Do not store large artifact bodies in Temporal.** Workflows store artifact IDs, not contents.

## Core execution abstraction

Even though the first serious backend is Temporal, define a generic execution interface so tests
run without Temporal.

```python
class ExecutionBackend:
    def start_subtask(self, subtask_id: str) -> ExecutionHandle: ...
    def start_review(self, review_id: str) -> ExecutionHandle: ...
    def start_approval(self, approval_id: str) -> ExecutionHandle: ...
    def get_status(self, handle_id: str) -> ExecutionStatus: ...
    def cancel(self, handle_id: str) -> None: ...
    def signal(self, handle_id: str, name: str, payload: dict) -> None: ...
```

Backends: `DirectExecutionBackend`, `TemporalExecutionBackend`. Use `DirectExecutionBackend` only
for tests and local smoke flows.

## Temporal workflow rules

Workflow code must be **deterministic**. Do not call LLMs, model APIs, Linear APIs, filesystem IO,
network IO, or random/time APIs directly inside workflows. Put external work in Activities.

Good:
```
Workflow:
  call route_subtask activity
  if approval needed:
    call request_approval activity
    wait for approval signal
  call run_worker activity
  call save_artifact activity
  call update_linear activity
```

Bad: calling OpenAI directly, calling Linear directly, reading an artifact file directly,
inspecting wall-clock time directly.

## Core workflows

### 1. SubtaskWorkflow
Run a bounded worker subtask durably.

1. Load subtask from AgentConnect DB.
2. Call `route_subtask` activity.
3. If route requires approval: mark subtask `needs_approval`, call `request_approval`, wait for approval signal/update.
4. Call `run_worker` activity.
5. Call `save_artifact` activity.
6. Record worker run result.
7. Update subtask status.
8. Update Linear with compact result.
9. Return subtask result summary.

Signals: `approval_granted`, `approval_denied`, `cancel_requested`, `manager_note_added`.
Queries: `status`, `progress`, `selected_route`, `current_wait_reason`.

### 2. ReviewWorkflow
Coordinate manager-to-manager or manager-to-worker review.

1. Load review ticket.
2. Update Linear with review request.
3. Wait for `review_claimed` or `review_completed` signal.
4. If completed: save review artifact, mark review completed, update Linear.
5. If timeout: mark review stale or needs_attention.

Signals: `review_claimed`, `review_completed`, `cancel_requested`.
Queries: `status`, `assigned_to`, `artifact_refs`.

### 3. ApprovalWorkflow
Wait for human approval for paid/rented/cloud/risky execution.

1. Create approval record.
2. Update Linear issue with approval request.
3. Wait for approval signal from Linear webhook or API.
4. Record approval/denial in AgentConnect.
5. Signal waiting workflow if needed.

Signals: `approval_granted`, `approval_denied`, `approval_expired`.

### 4. WorkerPipelineWorkflow
Run a multi-step agent work plan (repo map → plan → patch candidate → test worker → retry on
failure within policy → request review → save final handoff). Deferred until SubtaskWorkflow and
ReviewWorkflow are stable.

## Activities

Thin calls into AgentConnect service, router, workers, and Linear.

```
route_subtask(subtask_id) -> RouteExplanation
request_approval(subtask_id, route_explanation) -> ApprovalRecord
run_worker(subtask_id, route_explanation) -> WorkerResult
save_artifact(task_id, worker_result) -> Artifact
record_attempt(task_id, attempt) -> Attempt
update_linear(task_id, update) -> None
complete_review(review_id, result) -> Review
notify_manager(manager_id, message) -> None
```

Activities must be idempotent where possible. Idempotency keys:
`subtask_id + activity_name + attempt_number`, `review_id + activity_name`,
`approval_id + activity_name`.

## Linear integration

Task → Issue. Review → comment/sub-issue. Approval → status/label/comment requiring human action.
Artifact → comment with artifact link, **not** full content.

Labels: `agentconnect`, `manager:*`, `worker:{local,cloud,rented}`, `privacy:*`, `needs-review`,
`needs-approval`, `blocked`.

Webhook handling: receive → parse status/comment/assignment change → convert to AgentConnect event
→ **if the event corresponds to a running Temporal workflow, send a signal or update to it** →
record event in the AgentConnect DB.

Approval comments: `/agentconnect approve cloud`, `/agentconnect approve rented-gpu max_cost=3.00`,
`/agentconnect deny`.

## MCP tools

`create_task, open_task, get_handoff_summary, claim_task, release_task, record_decision,
record_attempt, request_review, submit_subtask, get_status, list_artifacts, read_artifact_chunk,
explain_route`

When `submit_subtask` is called, MCP creates a Subtask record and **starts a Temporal
SubtaskWorkflow. Do not run the worker directly inside the MCP request.** Return quickly with
`subtask_id`, `workflow_id`, `status`, route pending/running/needs_approval, next check tool.

## HTTP API

The `docs/BACKPLANE_SPEC.md` §11 set, plus:

```
GET    /workflows/{workflow_id}
POST   /workflows/{workflow_id}/signal
```

## Worker router

Unchanged from `docs/BACKPLANE_SPEC.md` §20: filter ineligible → score eligible → pick highest →
store route explanation → if none eligible, `needs_approval` when a paid/rented route is possible,
otherwise `failed`.

## Worker adapters

Unchanged base class. Start with `echo_worker` and `raw_model_worker`; then LiteLLM, local
OpenAI-compatible endpoint, Local Model Manager, Deep Agents, OpenClaw, sandboxed shell,
rented GPU.

## Handoff summary

Deterministic. Include task title, goal, status, current manager claim, Linear issue link,
constraints, locked decisions, recent attempts, important artifacts, open reviews, open subtasks,
**running Temporal workflows, waiting approvals**, suggested next action.

## Testing

task creation · Linear sync mapping · claim exclusivity · decision recording · artifact chunking ·
subtask starts Temporal workflow · Temporal workflow records route explanation · approval wait
workflow · Linear approval webhook signals workflow · echo worker execution through Temporal
activity · worker result artifact saved · review workflow claim/complete · manager switch flow ·
MCP submit_subtask returns quickly · Temporal workflow cancellation · retry behavior on activity
failure · idempotency of Linear update activity

## Acceptance criteria

1. Start Temporal dev server.
2. Start AgentConnect API.
3. Create task through CLI/API.
4. Sync task to Linear.
5. Start MCP server.
6. Claude Code claims task through MCP.
7. Claude records a locked decision.
8. Claude submits a subtask through MCP.
9. AgentConnect starts a Temporal SubtaskWorkflow.
10. Workflow routes to echo worker.
11. Echo worker result becomes artifact.
12. Linear issue receives compact update with artifact link.
13. Claude requests Codex review.
14. AgentConnect starts ReviewWorkflow.
15. Codex claims/completes review through API/CLI.
16. Review artifact is stored.
17. Linear receives review result summary.
18. Handoff summary includes task state, decisions, artifacts, reviews, running/completed workflows.
19. Claude releases task.
20. Codex claims task and continues from same state.

## Implementation phases

1. SQLite task ledger, artifact store, `AgentConnectService`, HTTP API, CLI, deterministic handoff.
2. Temporal client, Temporal worker process, SubtaskWorkflow, basic activities, `echo_worker`.
3. Linear push-only sync, compact updates, `external_ref` mapping.
4. MCP adapter; `submit_subtask` starts a Temporal workflow; artifact chunk reads; status queries.
5. ReviewWorkflow, manager inbox, claim/complete review, manager switch workflow.
6. Linear webhooks, approval command parsing, ApprovalWorkflow, signals into Temporal workflows.
7. Real worker adapters: LiteLLM, local OpenAI-compatible endpoint, Local Model Manager; Deep Agents/OpenClaw later.
8. Rented GPU lifecycle workflow: provision, health check, run, collect artifact, teardown, failure cleanup.

## Design principle

**Temporal makes execution durable. AgentConnect makes work state durable. Linear makes work
visible. Do not collapse these layers.**
