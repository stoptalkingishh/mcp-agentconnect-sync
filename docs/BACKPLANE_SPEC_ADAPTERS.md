# AgentConnect Handoff — Memory Adapter Interface and External Local Model Manager Boundary

> Authored by the project owner, 2026-07-10. Third handoff, layered on `BACKPLANE_SPEC.md` and
> `BACKPLANE_SPEC_TEMPORAL.md`. See `docs/BACKPLANE.md` for as-built status.

## Goal

Integrate with a pluggable memory layer and an optional external local model manager **without
making either one part of AgentConnect's core.**

AgentConnect remains responsible for: task state, manager claims, review tickets, decisions,
attempts, artifacts, Temporal workflows, Linear sync, routing policy, worker execution
coordination.

AgentConnect must **not** become: a memory engine, a vector database, a knowledge graph system, a
local model runtime manager, a VRAM/load-balancing service, an Ollama/vLLM/SGLang replacement.

Instead, AgentConnect defines stable adapter interfaces.

---

# Part A — Memory Adapter Interface

## Design principle

AgentConnect controls **when** memory is read or written. The memory backend controls **how**
memory is stored, indexed, retrieved, and governed.

Managers and subagents should not freely query memory backends directly. They ask AgentConnect for
scoped context packs.

```
Manager / subagent
        ↓
AgentConnect recall_memory(...)
        ↓
MemoryAdapter
        ↓
WikiBrain / Cognee / Graphiti / Mem0 / other backend
```

This preserves policy control and keeps manager context bounded.

## Required behavior

Support backends such as `wikibrain`, `cognee`, `graphiti`, `mem0`, `supermemory`, `none` — but
the core depends only on a generic `MemoryAdapter`.

First implementation: `NoopMemoryAdapter`, `StaticMemoryAdapter` (tests), `HttpMemoryAdapter`
skeleton. Later: `WikiBrainMemoryAdapter`, `CogneeMemoryAdapter`, `GraphitiMemoryAdapter`.

## Memory is optional

AgentConnect must work with memory disabled. With no backend configured:

* `recall_memory` → empty context pack with warning
* `capture_memory_candidate` → no-op or stored local pending event
* `record_memory_feedback` → no-op

**No core workflow may fail just because memory is unavailable.**

## Core interface

```python
MemoryProfile = Literal[
    "manager_brief", "worker_brief", "reviewer_brief", "implementation_constraints",
    "user_preferences", "known_failures", "model_performance",
]
MemoryStatus = Literal[
    "promoted", "pending", "rejected", "superseded", "contradicted", "archived", "unknown",
]
MemoryConfidence = Literal["low", "medium", "high", "verified", "unknown"]

@dataclass
class MemoryScope:
    scope_type: str  # global | user | project | repo | task | manager | worker | model | tool
    scope_id: str

@dataclass
class RecallRequest:
    query: str
    task_id: str | None = None
    profile: MemoryProfile = "manager_brief"
    scopes: list[MemoryScope] = field(default_factory=list)
    max_items: int = 8
    trusted_only: bool = True
    include_pending: bool = False
    include_superseded: bool = False
    include_sources: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryItem:
    text: str
    status: MemoryStatus
    confidence: MemoryConfidence
    source_id: str | None = None
    source_url: str | None = None
    scope: MemoryScope | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    superseded_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class RecallPack:
    profile: MemoryProfile
    query: str
    items: list[MemoryItem]
    warnings: list[str] = field(default_factory=list)
    backend: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class CaptureRequest:
    text: str
    task_id: str | None = None
    proposed_scopes: list[MemoryScope] = field(default_factory=list)
    origin_actor_id: str | None = None
    origin_actor_type: str | None = None  # manager | worker | human | system
    source_ref: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class CaptureResult:
    accepted: bool
    candidate_id: str | None = None
    status: MemoryStatus = "pending"
    message: str | None = None
    backend: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryFeedbackRequest:
    task_id: str | None
    memory_item_id: str | None
    source_id: str | None
    feedback: str  # useful | irrelevant | stale | wrong | too_broad | missing_context
    actor_id: str | None = None
    note: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

class MemoryAdapter(Protocol):
    @property
    def backend_name(self) -> str: ...
    def recall(self, request: RecallRequest) -> RecallPack: ...
    def capture_candidate(self, request: CaptureRequest) -> CaptureResult: ...
    def record_feedback(self, request: MemoryFeedbackRequest) -> None: ...
    def health(self) -> dict[str, Any]: ...
```

## AgentConnect service methods

```python
class AgentConnectService:
    def recall_memory(self, request: RecallRequest) -> RecallPack: ...
    def capture_memory_candidate(self, request: CaptureRequest) -> CaptureResult: ...
    def record_memory_feedback(self, request: MemoryFeedbackRequest) -> None: ...
    def get_task_context_pack(
        self, task_id: str, profile: MemoryProfile = "manager_brief",
        max_memory_items: int = 8,
    ) -> TaskContextPack: ...
```

`get_task_context_pack` combines: AgentConnect task state, constraints, locked decisions, recent
attempts, important artifacts, open reviews, running workflows, waiting approvals, and the scoped
memory recall pack. **Memory must be clearly labeled as external recalled context.**

## MCP tools

Add memory tools only after the adapter exists: `recall_memory`, `capture_memory_candidate`,
`record_memory_feedback`, `get_task_context_pack`. Responses must be bounded. **Do not expose raw
backend search dumps through MCP.**

`capture_memory_candidate` **never promotes directly** — it returns `status: "pending"`.

## HTTP API

```
POST /memory/recall
POST /memory/capture
POST /memory/feedback
GET  /memory/health
GET  /tasks/{task_id}/context-pack
```

The API calls `AgentConnectService`, not the backend directly.

## CLI

```
agentconnect memory recall --task TASK_ID --query "..." --profile manager_brief
agentconnect memory capture --task TASK_ID --text "..."
agentconnect memory feedback --item ITEM_ID --feedback useful
agentconnect memory health
agentconnect tasks context-pack TASK_ID
```

## Temporal integration

Memory calls happen inside Activities, never inside deterministic workflow code.

```
recall_memory_activity(task_id, query, profile, max_items) -> RecallPack
capture_memory_candidate_activity(task_id, text, origin_actor_id, origin_actor_type) -> CaptureResult
record_memory_feedback_activity(...) -> None
```

If memory recall fails: continue the workflow, record a warning, **do not fail the task by
default.**

## Memory access policy

```yaml
memory:
  enabled: true
  backend: wikibrain
  trusted_only_default: true
  max_items_default: 8
  allow_pending_injection: false
  capture_candidates_default: true
```

Defaults: `trusted_only=true`, `include_pending=false`, `include_superseded=false`, `max_items=8`.
Managers may request pending memory only through explicit tool/API parameters, and pending memory
must be labeled.

## Backend-specific adapters

* **NoopMemoryAdapter** — memory disabled: empty recall, non-accepting capture, no-op feedback, `health: disabled`.
* **StaticMemoryAdapter** — tests; items from config or fixture.
* **HttpMemoryAdapter** — generic HTTP memory service (`base_url`, `api_key_env`).
* **WikiBrainMemoryAdapter** — `brain_recall` / `brain_capture` (always pending) / feedback endpoint.
* **CogneeMemoryAdapter** — map results into `RecallPack`. If Cognee lacks pending/promoted
  semantics, the adapter labels status `unknown` or implements an AgentConnect-side pending table.

## Acceptance criteria (Part A)

1. AgentConnect runs with memory disabled.
2. AgentConnect runs with `StaticMemoryAdapter`.
3. MCP `recall_memory` returns a bounded `RecallPack`.
4. `capture_memory_candidate` never promotes directly.
5. `get_task_context_pack` includes task state plus memory items.
6. Temporal workflows can call `recall_memory_activity`.
7. Memory backend failure does not crash a subtask workflow by default.
8. Tests verify `trusted_only` filtering behavior.
9. Tests verify pending memory is not injected unless explicitly requested.
10. CLI can recall and capture memory.

---

# Part B — External Local Model Manager Boundary

## Design principle

The local model manager is a **separate optional subsystem**. AgentConnect knows how to ask local
compute for estimates and execution. It does not own local runtime details.

```
AgentConnect:        task policy, Temporal workflows, Linear, artifacts, manager state
Local Model Manager: model inventory, local runtime selection, VRAM/RAM admission, loading, queueing
```

Reasons to separate: local hardware is user-specific; model/runtime choices are user-specific;
Ollama/vLLM/SGLang/llama.cpp/LM Studio evolve independently; VRAM/load management is its own
problem; users may already have bespoke local inference setups; AgentConnect should remain
task/workflow focused.

**AgentConnect includes a client adapter, not the full model manager.**

## Worker adapter shape

```
AgentConnect SubtaskWorkflow
        ↓
run_worker_activity
        ↓
LocalModelManagerWorkerAdapter
        ↓
external local model manager API
        ↓
Ollama / vLLM / SGLang / llama.cpp / LM Studio / custom
```

## Local compute provider interface

```python
@dataclass
class LocalModel:
    id: str
    runtime: str
    capabilities: list[str]
    context_tokens: int
    loaded: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class LocalEstimateRequest:
    task_type: str
    privacy_tier: str
    required_capabilities: list[str]
    context_tokens: int
    max_output_tokens: int
    latency_preference: str = "normal"
    quality_preference: str = "good_enough"
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class LocalEstimate:
    eligible: bool
    selected_model: str | None = None
    runtime: str | None = None
    loaded: bool = False
    estimated_queue_seconds: float | None = None
    estimated_tokens_per_second: float | None = None
    estimated_quality: float | None = None
    reason: dict[str, Any] = field(default_factory=dict)

@dataclass
class LocalRunRequest:
    model: str | None
    task_type: str
    prompt: str
    context: str | None = None
    max_output_tokens: int = 2048
    temperature: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class LocalRunResult:
    status: str
    output: str
    model: str | None
    runtime: str | None
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

class LocalComputeProvider(Protocol):
    def inventory(self) -> list[LocalModel]: ...
    def loaded(self) -> list[LocalModel]: ...
    def estimate(self, request: LocalEstimateRequest) -> LocalEstimate: ...
    def run(self, request: LocalRunRequest) -> LocalRunResult: ...
    def health(self) -> dict[str, Any]: ...
```

## Local model manager HTTP API expectation

```
GET  /health
GET  /models
GET  /models/loaded
POST /route/estimate
POST /generate
POST /runs/{run_id}/cancel
```

`/route/estimate` input:
```json
{"task_type": "code_search", "privacy_tier": "repo_sensitive",
 "required_capabilities": ["code", "summarization"], "context_tokens": 12000,
 "max_output_tokens": 2000, "latency_preference": "normal",
 "quality_preference": "good_enough"}
```

`/route/estimate` output:
```json
{"eligible": true, "selected_model": "qwen2.5-coder-14b-q4", "runtime": "vllm",
 "loaded": true, "estimated_queue_seconds": 8, "estimated_tokens_per_second": 42,
 "estimated_quality": 0.72,
 "reason": {"hard_gates": ["context_fits", "capability_match", "privacy_local"],
            "score_terms": {"already_loaded": 1.0, "capability": 0.8,
                            "latency": 0.7, "quality": 0.7}}}
```

## MVP local model strategy

Do not require a local model manager for MVP. Support first: `echo_worker`,
`openai_compatible_worker`, `litellm_worker`. Then: `local_model_manager_worker`.

```yaml
workers:
  - id: local-qwen
    type: openai_compatible
    base_url: http://localhost:8000/v1
    model: qwen2.5-coder-14b
    privacy: local_only
    capabilities: [code, summarize]
    context_tokens: 32768

  - id: local-manager
    type: local_model_manager
    base_url: http://localhost:8090
    privacy: local_only
```

## AgentConnect global router behavior

AgentConnect's global router decides **whether local compute is eligible**. It does not choose
local model internals.

AgentConnect asks: *Can local compute handle this?*
Local manager answers: *Yes, with this model/runtime/queue estimate* — or *No, context too large or
no capable model available.*

AgentConnect then decides: run local, try cheap cloud, request approval, fail, or queue.

## Route explanation integration

```json
{"selected_worker": "local-manager", "worker_type": "local_model_manager",
 "hard_gates": ["privacy_allowed", "local_available"],
 "score_terms": {"privacy_fit": 1.0, "cost": 1.0, "availability": 0.8},
 "local_estimate": {"selected_model": "qwen2.5-coder-14b-q4", "runtime": "vllm",
                    "loaded": true, "estimated_queue_seconds": 8,
                    "reason": {"hard_gates": ["context_fits", "capability_match"],
                               "score_terms": {"already_loaded": 1.0, "capability": 0.8}}}}
```

## Temporal behavior

Workflows call local models through Activities; workflow code never calls the local model manager
directly. Activity failure is retried per policy. If the local manager is unavailable: record a
worker failure, update the route explanation, and let the router choose a fallback if policy
allows.

## Acceptance criteria (Part B)

1. AgentConnect can run without a local model manager.
2. AgentConnect can use a static OpenAI-compatible local worker.
3. AgentConnect can call a mocked local model manager `/route/estimate`.
4. AgentConnect can call mocked `/generate` and store the result artifact.
5. Route explanation includes the local manager estimate.
6. Temporal workflow executes `local_model_manager_worker` through an Activity.
7. Local manager outage does not crash the AgentConnect API/MCP.
8. Tests verify the local model manager is optional.

---

## Final architecture rule

**AgentConnect defines the contract. Users may bring their own local model manager. AgentConnect
integrates with it — it does not own it.**
