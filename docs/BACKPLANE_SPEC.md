# AgentConnect Implementation Handoff — Linear-First Agent Task Backplane

> Authored by the project owner, 2026-07-10. This is the canonical statement of intent for the
> backplane. Where this document and older docs disagree about *what AgentConnect is*, this wins.
> Where this document and the code disagree about *what exists today*, the code wins — see
> `docs/BACKPLANE.md` for the as-built status against these phases.

## 1. Goal

Refactor or implement AgentConnect as a local-first, protocol-neutral task/ticket backplane for
interchangeable agent managers and interchangeable worker harnesses.

AgentConnect should let manager harnesses such as Claude Code, Codex, Linear Agent, OpenClaw, or
future frontier agents:

* attach to the same durable task state
* claim and release work
* record decisions
* request reviews from other managers
* delegate subtasks to interchangeable workers
* retrieve artifacts in chunks
* resume from compact handoff summaries
* expose human-visible workflow through Linear

The core product is not "an MCP router." The core product is:

> A persistent task, artifact, decision, review, routing, and handoff backplane for agent managers
> and workers.

MCP is one adapter. HTTP API and CLI are also required. A2A should be designed for later but not
required in the first implementation.

## 2. Product thesis

Frontier model harnesses are good managers. Local and cheaper models are useful bounded workers.

AgentConnect should sit between them:

```
Claude Code / Codex / Linear Agent / future manager
        ↓
MCP / HTTP API / CLI / future A2A
        ↓
AgentConnect task ledger
        ↓
local models / cheap APIs / worker harnesses / shell workers
```

The manager can change. The task state remains.
The worker harness can change. The task contract remains.

Linear gives humans visibility, approvals, and workflow control. AgentConnect remains the source
of truth.

## 3. Non-goals

* Do not build a Claude subscription workaround.
* Do not build a general model gateway. Use LiteLLM, OpenRouter, or direct provider adapters where appropriate.
* Do not build a full agent framework if existing worker harnesses can be adapted.
* Do not try to preserve hidden manager reasoning or full chat history between managers.
* Do not make Linear the canonical database.
* Do not require A2A support from proprietary managers.
* Do not make MCP the only interface.
* Do not implement free-form manager-to-manager chat as the main primitive.

## 4. High-level architecture

```
┌────────────────────────────────────────────┐
│ Human / Linear                             │
│ - issues                                   │
│ - comments                                 │
│ - approvals                                │
│ - assignment                               │
│ - status visibility                        │
└──────────────────────┬─────────────────────┘
                       ▼
┌────────────────────────────────────────────┐
│ AgentConnect Linear Adapter                │
│ - sync AgentConnect tasks to Linear issues │
│ - ingest Linear comments/status changes    │
│ - expose approval and review workflow      │
└──────────────────────┬─────────────────────┘
                       ▼
┌────────────────────────────────────────────┐
│ AgentConnect Core                          │
│ - task ledger                              │
│ - manager claims                           │
│ - decision log                             │
│ - attempt log                              │
│ - review tickets                           │
│ - artifact registry                        │
│ - handoff summaries                        │
│ - route history                            │
│ - worker runs                              │
└───────────────┬───────────────┬────────────┘
                ▼               ▼
┌──────────────────────┐   ┌──────────────────────┐
│ Manager Adapters      │   │ Worker Router         │
│ - MCP                 │   │ - privacy policy      │
│ - HTTP API            │   │ - budget policy       │
│ - CLI                 │   │ - capability matching │
│ - future A2A          │   │ - route explanation   │
└──────────────────────┘   └──────────┬───────────┘
                                       ▼
                         ┌────────────────────────┐
                         │ Worker Harnesses        │
                         │ - echo worker           │
                         │ - raw model worker      │
                         │ - LiteLLM worker        │
                         │ - local model manager   │
                         │ - Deep Agents later     │
                         │ - OpenClaw later        │
                         │ - sandbox shell later   │
                         └────────────────────────┘
```

## 5. Core design rule

AgentConnect is task-first.

MCP, HTTP API, CLI, Linear, and future A2A are adapters over the same internal service layer.
**Do not duplicate logic per protocol.**

```
MCP tool call / HTTP request / CLI command / Linear webhook / future A2A request
        ↓
AgentConnectService
        ↓
same storage, same task model, same policy, same artifacts
```

## 6. Source-of-truth boundaries

**AgentConnect owns:** tasks, subtasks, manager claims, reviews, decisions, attempts, artifacts,
route history, worker runs, handoff summaries.

**Linear owns:** human visibility, issue workflow, comments, approval surface, assignment surface,
project/team planning view.

**Worker harnesses own:** execution internals, temporary local context, tool loops,
model-specific behavior.

Worker internals must not be treated as durable truth unless recorded back into AgentConnect as
attempts, artifacts, decisions, or review results.

## 7. Recommended package layout

```
packages/
  agentconnect-core/    src/agentconnect/core/
      models.py service.py storage.py artifacts.py claims.py decisions.py
      attempts.py reviews.py subtasks.py handoff.py errors.py
  agentconnect-router/  src/agentconnect/router/
      policy.py privacy.py budget.py routing.py worker_registry.py
      worker_adapters/{base,echo,raw_model,litellm,local_model_manager,deepagents,openclaw,shell_sandbox}.py
  agentconnect-api/     src/agentconnect/api/
      app.py routes_tasks.py routes_artifacts.py routes_reviews.py
      routes_managers.py routes_subtasks.py routes_linear.py
  agentconnect-mcp/     src/agentconnect/mcp/    server.py tools.py
  agentconnect-cli/     src/agentconnect/cli/    main.py
  agentconnect-linear/  src/agentconnect/linear/ client.py sync.py webhooks.py mapping.py
```

If the current monorepo already has usable package names, adapt without breaking existing imports.

## 8. Storage

SQLite for the first version. Filesystem-backed artifact storage.

```
~/.agentconnect/agentconnect.db
~/.agentconnect/artifacts/
```

Environment overrides: `AGENTCONNECT_DB_PATH`, `AGENTCONNECT_ARTIFACT_DIR`.

Keep the storage interface abstract enough to support Postgres later.

## 9. Core data model

### 9.1 Task
`id, title, goal, status, priority, created_by, created_at, updated_at, current_manager,
handoff_summary, linear_issue_id, linear_issue_url, metadata_json`

Statuses: `queued, in_progress, blocked, needs_review, needs_approval, succeeded, failed, cancelled`

### 9.2 Constraint
`id, task_id, text, created_by, created_at`

### 9.3 Claim
`id, task_id, manager_id, role, expires_at, created_at, released_at`

Roles: `primary_manager, reviewer, planner, implementer, observer, human_owner, worker_delegate`

Rules: only one active `primary_manager` claim per task; a claim expires automatically after
`expires_at`; expired claims must not block new claims; released claims remain in history.

### 9.4 Decision
`id, task_id, made_by, decision, rationale, locked, created_at`

Locked decisions must appear prominently in handoff summaries. If a manager attempts to record a
decision contradicting a locked decision, reject unless the caller has `human_owner` or admin
authority.

### 9.5 Attempt
`id, task_id, actor_id, actor_type, summary, outcome, created_at, artifact_refs_json`

Actor types: `manager, worker, human, system`

### 9.6 Artifact
`id, task_id, type, path, summary, created_by, created_at, metadata_json`

Types: `summary, report, patch, test_log, file_snapshot, review, worker_output, plan,
route_explanation, other`

Artifact bodies must be retrievable in chunks. Do not dump large artifacts into MCP or Linear
responses. Return summaries and artifact IDs.

### 9.7 Review
`id, task_id, requested_by, assigned_to, status, criteria_json, artifact_refs_json,
result_artifact_id, created_at, updated_at`

Statuses: `open, claimed, in_progress, completed, rejected, cancelled`

Reviews are the main manager-to-manager coordination primitive. Do not implement generic
free-form manager chat first.

### 9.8 Subtask
`id, parent_task_id, title, instructions, status, privacy_tier, preferred_worker,
assigned_worker, created_at, updated_at, result_artifact_id, route_reason_json, metadata_json`

Statuses: `queued, running, succeeded, failed, cancelled, needs_approval`
Privacy tiers: `public, public_redacted, repo_sensitive, secret_sensitive, local_only`

### 9.9 WorkerRun
`id, subtask_id, worker_id, harness, model, status, route_reason_json, started_at, finished_at,
input_artifact_id, output_artifact_id, metrics_json, error`

### 9.10 ExternalRef
`id, entity_type, entity_id, provider, external_id, external_url, sync_enabled, created_at,
updated_at, metadata_json`

Initial provider: `linear`. Future: `github, jira, local_markdown`.

## 10. Internal service layer

A protocol-neutral `AgentConnectService`. Every adapter must call it. No adapter may bypass it.

```python
class AgentConnectService:
    def create_task(self, request: CreateTaskRequest) -> Task: ...
    def get_task(self, task_id: str) -> TaskDetail: ...
    def list_tasks(self, filters: TaskFilters) -> list[TaskSummary]: ...
    def claim_task(self, task_id: str, manager_id: str, role: str, ttl_seconds: int) -> Claim: ...
    def release_task(self, task_id: str, manager_id: str) -> None: ...
    def get_handoff_summary(self, task_id: str, manager_id: str | None = None) -> HandoffSummary: ...
    def regenerate_handoff_summary(self, task_id: str) -> HandoffSummary: ...
    def record_decision(self, task_id: str, request: RecordDecisionRequest) -> Decision: ...
    def record_attempt(self, task_id: str, request: RecordAttemptRequest) -> Attempt: ...
    def create_artifact(self, task_id: str, request: CreateArtifactRequest) -> Artifact: ...
    def read_artifact_chunk(self, artifact_id: str, offset: int, limit: int) -> ArtifactChunk: ...
    def list_artifacts(self, task_id: str) -> list[ArtifactSummary]: ...
    def request_review(self, task_id: str, request: ReviewRequest) -> Review: ...
    def get_manager_inbox(self, manager_id: str) -> list[InboxItem]: ...
    def claim_review(self, review_id: str, manager_id: str) -> Review: ...
    def complete_review(self, review_id: str, request: ReviewResultRequest) -> Review: ...
    def submit_subtask(self, task_id: str, request: SubtaskRequest) -> Subtask: ...
    def get_subtask(self, subtask_id: str) -> SubtaskDetail: ...
    def cancel_subtask(self, subtask_id: str) -> None: ...
    def explain_route(self, subtask_id: str) -> RouteExplanation: ...
```

## 11. HTTP API

FastAPI over `AgentConnectService`. Minimum endpoints:

```
POST   /tasks
GET    /tasks
GET    /tasks/{task_id}
GET    /tasks/{task_id}/handoff
POST   /tasks/{task_id}/claim
POST   /tasks/{task_id}/release
POST   /tasks/{task_id}/decisions
POST   /tasks/{task_id}/attempts
POST   /tasks/{task_id}/artifacts
GET    /tasks/{task_id}/artifacts
GET    /artifacts/{artifact_id}
GET    /artifacts/{artifact_id}/chunk?offset=0&limit=8000
POST   /tasks/{task_id}/reviews
GET    /managers/{manager_id}/inbox
POST   /reviews/{review_id}/claim
POST   /reviews/{review_id}/result
POST   /tasks/{task_id}/subtasks
GET    /subtasks/{subtask_id}
POST   /subtasks/{subtask_id}/cancel
GET    /subtasks/{subtask_id}/route
POST   /linear/sync
POST   /linear/webhook
GET    /linear/tasks/{task_id}
```

Stable public IDs with prefixes: `task_ claim_ decision_ attempt_ artifact_ review_ subtask_ run_ external_`

## 12. CLI

```
agentconnect tasks create --title ... --goal ...
agentconnect tasks list
agentconnect tasks show TASK_ID
agentconnect tasks handoff TASK_ID
agentconnect tasks claim TASK_ID --manager claude-code --role primary_manager
agentconnect tasks release TASK_ID --manager claude-code
agentconnect decisions add TASK_ID --by claude-code --decision ... --rationale ...
agentconnect attempts add TASK_ID --actor claude-code --summary ... --outcome ...
agentconnect artifacts add TASK_ID --type report --file path.txt --summary ...
agentconnect artifacts list TASK_ID
agentconnect artifacts read ARTIFACT_ID --offset 0 --limit 8000
agentconnect reviews request TASK_ID --to codex --by claude-code --artifact ARTIFACT_ID --criteria ...
agentconnect inbox codex
agentconnect reviews claim REVIEW_ID --manager codex
agentconnect reviews complete REVIEW_ID --file result.md
agentconnect subtasks submit TASK_ID --title ... --instructions ... --privacy repo_sensitive
agentconnect subtasks show SUBTASK_ID
agentconnect subtasks route SUBTASK_ID
agentconnect linear sync TASK_ID
agentconnect linear issue TASK_ID
agentconnect linear webhook-test PAYLOAD_FILE
```

The CLI may call the HTTP API or instantiate the service directly in local mode.

## 13. MCP adapter

Minimal MCP server exposing only manager-useful tools:

```
create_task  open_task  get_handoff_summary  claim_task  release_task
record_decision  record_attempt  request_review  submit_subtask
get_status  list_artifacts  read_artifact_chunk  explain_route
```

Do not expose every admin endpoint through MCP. MCP responses must be compact and
action-oriented.

**Bad:** full logs, full patch, full worker trace, full route JSON, every artifact body.
**Good:** task ID, status, short summary, artifact IDs, next action, route summary, explicit
command to read chunks if needed.

## 14. Linear-first external visibility layer

Linear is the first external tracker integration. AgentConnect remains canonical. Linear is the
human-visible mirror and control surface.

### 14.1 Mapping
```
AgentConnect Task      → Linear Issue
AgentConnect Subtask   → Linear sub-issue, linked issue, or compact issue comment
AgentConnect Review    → Linear sub-issue, linked issue, or review comment
AgentConnect Artifact  → Linear comment with AgentConnect artifact link
AgentConnect Decision  → Linear comment with "Decision:" marker
AgentConnect Approval  → Linear label/status/comment requiring human action
AgentConnect Claim     → Linear label or issue field showing current manager
```

### 14.2 Issue body
Create or update the Linear issue description with: AgentConnect Task ID, Status, Current manager,
Privacy, Priority, Goal, Constraints, Current handoff, Important artifacts, Open reviews, Open
subtasks.

### 14.3 Suggested labels
```
agentconnect
manager:claude-code  manager:codex  manager:linear-agent
worker:local  worker:cloud  worker:rented
privacy:public  privacy:repo-sensitive  privacy:secret-sensitive  privacy:local-only
needs-review  needs-approval  blocked
```

### 14.4 Sync modes

**Mode 1 — push-only sync.** AgentConnect creates and updates Linear issues.
*Acceptance:* create task → run sync → issue appears with title, goal, status, handoff summary, labels.

**Mode 2 — webhook ingest.** Linear comments/status/label changes become AgentConnect events.
*Acceptance:* human comments `/agentconnect approve cloud` → webhook reaches AgentConnect → approval event recorded.

**Mode 3 — agent-native workflow.** Linear assignment or labels create manager inbox items.
*Acceptance:* assign issue to `codex-manager` → AgentConnect creates inbox item → Codex adapter can claim task or review through API/CLI.

## 15. Linear approval workflow

Linear is the human approval surface for expensive or risky actions.

1. Manager submits subtask.
2. Routing says paid cloud or rented GPU is needed.
3. AgentConnect marks subtask `needs_approval`.
4. Linear issue gets label `needs-approval`.
5. AgentConnect comments with: requested route, estimated cost/risk, privacy tier, reason local/free workers were rejected.
6. Human approves via status/label/comment.
7. Linear webhook updates AgentConnect.
8. AgentConnect unlocks route and records approval.

```
/agentconnect approve cloud
/agentconnect approve rented-gpu max_cost=3.00
/agentconnect deny
```

## 16. Handoff summary behavior

Deterministic handoff summaries first. **Do not require an LLM.**

Include: task title, goal, current status, current manager claim, constraints, locked decisions,
recent attempts, important artifacts, open reviews, open subtasks, suggested next step if known,
Linear issue link if synced.

```
Task: Refactor auth session handling
Status: in_progress
Current manager: claude-code
Linear: LIN-123
Goal:
Reduce duplicated token expiry logic without changing public login behavior.
Constraints:
- No schema changes
- Preserve middleware contract
Locked decisions:
- Keep refresh token validation in auth/session.py because middleware assumes this location.
Recent attempts:
- Claude Code mapped auth flow.
- Local worker found duplicated expiry checks in auth/session.py and auth/tokens.py.
Important artifacts:
- artifact_flow_map_001: Auth flow map
- artifact_dup_report_002: Duplicate expiry logic report
Open items:
- Patch shared expiry helper
- Run auth test subset
```

Later, optional LLM-assisted summaries may be added. The deterministic version must always work.

## 17. Manager coordination model

Mediated coordination through AgentConnect. Do not start with direct manager-to-manager chat.

Primitives: `claim_task, release_task, request_review, claim_review, complete_review,
record_decision, record_attempt, get_manager_inbox, get_handoff_summary`.

Example flow: Claude Code requests Codex review → AgentConnect creates review ticket → Linear
shows review request → Codex adapter sees review in inbox → Codex claims review → Codex writes
review artifact → AgentConnect stores result → Linear gets compact review summary → Claude Code
reads result through MCP.

This lets managers "talk" through durable task state instead of dumping context into each other.

## 18. Worker adapter interface

```python
class WorkerAdapter:
    @property
    def worker_id(self) -> str: ...
    def capabilities(self) -> WorkerCapabilities: ...
    def estimate(self, subtask: Subtask, context: WorkerContext) -> WorkerEstimate: ...
    def run(self, subtask: Subtask, context: WorkerContext) -> WorkerResult: ...
    def cancel(self, run_id: str) -> None: ...
    def health(self) -> WorkerHealth: ...
```

Normalized worker result:

```json
{
  "status": "succeeded",
  "summary": "Found duplicated token expiry logic in auth/session.py and auth/tokens.py.",
  "artifacts": [
    {"artifact_id": "artifact_dup_report_001", "type": "report",
     "description": "Line-by-line duplicate validation candidates"}
  ],
  "metrics": {"tokens_in": 8420, "tokens_out": 1100,
              "wall_time_seconds": 38, "estimated_cost_usd": 0.002},
  "warnings": ["Worker was readonly; no files modified."]
}
```

Start with `echo_worker` and `raw_model_worker`. Add later: LiteLLM worker, Local Model Manager
worker, Deep Agents worker, OpenClaw worker, sandboxed shell worker, A2A worker adapter.

## 19. Separate worker harness from model

Do not hardcode worker identity as a model.

Bad: `worker = deepagents-qwen`

Better:
```json
{"worker_harness": "deepagents", "model": "qwen2.5-coder-14b-local",
 "tools": ["read_files", "grep", "write_artifact"],
 "sandbox": "readonly", "privacy_tier": "local_only"}
```

This allows: same harness/different model; same model/different harness; same task/different
privacy route; same manager/multiple worker types.

## 20. Routing

Deterministic routing first.

Inputs: subtask type, privacy tier, preferred worker, available worker capabilities, local
availability, budget policy, quota policy, estimated cost, required sandbox.

Procedure:
1. Filter ineligible workers.
2. Score eligible workers.
3. Select highest score.
4. Store route explanation as JSON.
5. If no eligible worker exists, mark subtask `needs_approval` or `failed` depending on policy.

```json
{
  "selected_worker": "local_qwen_worker",
  "selected_harness": "raw_model",
  "selected_model": "qwen2.5-coder-14b",
  "hard_gates": ["privacy_allowed", "capability_match", "budget_allowed"],
  "score_terms": {"privacy_fit": 1.0, "cost": 1.0, "availability": 0.8, "capability": 0.7},
  "rejected_workers": [
    {"worker": "cheap_cloud_deepseek",
     "reason": "repo_sensitive task cannot use public cloud worker"}
  ]
}
```

## 21. Security and safety defaults

* local-first
* no shell worker enabled by default
* no secrets mounted into workers by default
* artifacts treated as untrusted until reviewed
* secret-sensitive tasks do not sync raw content to Linear
* cloud/rented execution requires explicit approval
* large logs and full artifacts are not pushed to Linear
* MCP returns artifact refs, not full payloads

Subtasks declare sandbox expectations: `filesystem: none|readonly|workspace_write`,
`network: true|false`, `shell: true|false`. Initial workers should be readonly or artifact-only.

## 22. A2A posture

Design for A2A later. Do not require it now. MCP is the practical near-term interface for Claude
Code-style managers. HTTP API/CLI are the practical integration path for Codex adapters, Linear
workflows, and debugging.

Future A2A maps onto the same internal service model (`A2A request → AgentConnectService → same
task ledger`). Potential roles: task continuity agent, delegation router agent, artifact context
agent, manager review broker. Do not implement until the core task ledger, Linear sync, MCP
adapter, and worker routing are stable.

## 23. Example end-to-end flow

1. **Task creation** — `agentconnect tasks create --title "Refactor auth session handling" --goal "Reduce duplicate token expiry logic without changing behavior"` → `task_auth_refactor_001`; Linear sync creates issue `AUTH-123`.
2. **Claude Code claims through MCP** — `claim_task(task_id, manager_id="claude-code", role="primary_manager")`; AgentConnect records claim and updates Linear.
3. **Claude records decision** — `record_decision(decision="Keep refresh token validation in auth/session.py.", rationale="Middleware assumes this location.", locked=true)`; compact Linear comment posted.
4. **Claude delegates subtask** — `submit_subtask(title="Find duplicated expiry checks", instructions="Inspect auth files and return file paths and line ranges only. Do not edit files.", privacy_tier="repo_sensitive", preferred_worker="local")`; routed to local/echo worker; worker returns artifact; Linear gets compact update with artifact link.
5. **Claude requests Codex review** — `request_review(assigned_to="codex", artifact_refs=[...], criteria=["Find correctness issues","Check auth edge cases","Ignore style-only changes"])`; Linear shows review request; Codex reads `agentconnect inbox codex`, claims, completes.
6. **Manager switch** — Claude calls `release_task`; Codex runs `agentconnect tasks claim ... --role primary_manager` then `agentconnect tasks handoff ...` and continues from the same state.

## 24. Tests

task creation · task retrieval · task listing · constraint recording · claim creation · claim
exclusivity · claim expiry · claim release · decision recording · locked decision visibility in
handoff · attempt recording · artifact write/read/chunking · review request · manager inbox ·
review claim · review completion · subtask submission · echo worker execution · deterministic
route selection · route explanation persistence · MCP tools invoking service layer · HTTP
endpoints invoking service layer · CLI smoke tests · Linear issue mapping · Linear push-only sync
· Linear webhook parsing · approval command parsing · manager switch flow

Use `echo_worker` for deterministic worker tests.

## 25. Acceptance criteria

1. Start AgentConnect API.
2. Create task through CLI or API.
3. Sync task to Linear as an issue.
4. Start AgentConnect MCP server.
5. Claude Code claims task through MCP.
6. Claude records one locked decision.
7. Claude submits a readonly subtask.
8. Echo worker produces an artifact.
9. Artifact can be read in chunks through MCP and CLI.
10. Linear issue gets compact update with artifact link.
11. Claude requests Codex review.
12. Review appears in Codex inbox through API/CLI.
13. Codex completes review with a review artifact.
14. Linear receives compact review result.
15. Handoff summary includes task state, locked decisions, review result, artifacts, and next steps.
16. Claude releases task.
17. Codex claims task and can continue from the same state.

## 26. Implementation phases

1. **Core task ledger** — SQLite storage, filesystem artifact store, `AgentConnectService`, Task/Constraint/Claim/Decision/Attempt/Artifact models, deterministic handoff summary.
2. **HTTP API and CLI** — FastAPI app, task/artifact/claim/decision/review/subtask endpoints, CLI commands.
3. **Linear push-only integration** — client, task-to-issue mapping, issue create/update, labels/status mapping, `ExternalRef` storage, manual sync command.
4. **MCP adapter** — minimal manager tools, compact response formatting, artifact chunk reads, MCP-to-service integration tests.
5. **Worker routing MVP** — `WorkerAdapter` base class, `echo_worker`, optional `raw_model_worker`, subtask routing, route explanations, worker run records.
6. **Review and manager switching workflow** — `request_review`, manager inbox, `claim_review`, `complete_review`, claim/release flow, handoff after manager switch, Linear review visibility.
7. **Linear webhook ingest** — webhook endpoint, status change handling, comment command handling, approval parsing, assignment-to-inbox behavior.
8. **Real worker adapters** — LiteLLM, local OpenAI-compatible endpoint, Local Model Manager, Deep Agents, OpenClaw, sandbox shell.
9. **Future A2A adapter** — A2A server, Agent Card, task continuity skill, delegation router skill, artifact context skill, review broker skill.

## 27. Design principle

The durable boundary is:

```
task spec
→ normalized worker/manager result
→ artifact refs
→ decisions/attempts/reviews
→ handoff summary
```

Do not depend on preserving hidden reasoning, full chat history, or internal harness state.

Managers and workers are replaceable. The AgentConnect task ledger is the stable middle layer.
