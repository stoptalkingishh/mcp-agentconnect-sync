# Agent Runtime

> **Status:** the first slice is implemented in `packages/agentconnect-runtime`.
> A LangGraph act/tool loop executes tasks in a confined workspace with
> filesystem + shell tools and returns `WorkerResult`. The model is reached via
> the `ModelSource` protocol (`generate(GenerateRequest) -> GenerateResponse`),
> so the stub backend, residency manager, and HTTP clients all plug in.
> All first-slice surfaces are implemented: filesystem, shell, test-runner,
> and browser tools plus the remote transport (`create_worker_app` /
> `HttpAgentRuntime`).

This repo is the deterministic control plane around agents, not the agent worker itself.

The runtime layer should use **LangChain + LangGraph** as one stack:

- **LangChain** for the higher-level agent API, integrations, prompt/tool wiring, and model access
- **LangGraph** for the execution runtime, loops, branching, persistence, and durable state

That keeps the runtime aligned with the current ecosystem without making this repo own the low-level agent loop.

## What This Runtime Does

The runtime is the thing that actually executes work after the router assigns a task. It should:

1. Receive a `TaskSubmission`
2. Load workspace context
3. Call the model through the configured backend
4. Use tools for files, shell, tests, and optional browser/data access
5. Persist state and artifacts
6. Return a structured `WorkerResult`

The router remains responsible for:

- classification
- privacy and redaction
- budget and quota
- provider selection
- shared-memory storage
- compact summaries and artifact refs

## Why LangChain + LangGraph

The current LangChain docs present the stack as a unified agent platform, and LangGraph as the execution runtime underneath. The practical implication is that the high-level developer API and the stateful runtime now belong together.

For this repo, that is useful because:

- the runtime can stay focused on task execution rather than framework plumbing
- loops, retries, and human checkpoints are first-class
- the agent stack can evolve without changing the router contract

## Suggested Shape

```text
packages/
  agentconnect-runtime/
    pyproject.toml
    src/agentconnect/runtime/
      __init__.py
      agent.py
      graph.py
      tools/
        __init__.py
        filesystem.py
        shell.py
        tests.py
        browser.py
      workspace.py
      prompts.py
      state.py
      results.py
      transport.py
```

## Module Roles

### `agent.py`

The top-level entrypoint for task execution.

- accepts `TaskSubmission`
- builds the LangChain/LangGraph run
- returns `WorkerResult`

### `graph.py`

Defines the LangGraph state machine.

- nodes for planning, acting, observing, revising, and finalizing
- edges for retries, branching, and approval checkpoints
- typed state for task context and tool outputs

### `tools/`

The worker-side tools.

- filesystem edits
- shell commands and test runs
- browser or data access when needed
- any future domain-specific tools

### `workspace.py`

Manages the task checkout or working directory.

- repo state
- diffs
- changed files
- artifact generation

### Mid-run resumability (opt-in)

A crashed agentic run can resume **mid-reasoning** instead of re-dispatching from
scratch. Off by default (ephemeral, in-memory); enable it per run:

```python
LangGraphAgentRuntime(model_source,
    RuntimeConfig(checkpoint_root="/var/lib/agentconnect/checkpoints"))
```

- **Needs the `[resumable]` extra** (`pip install 'agentconnect-runtime[resumable]'`) —
  a LangGraph `SqliteSaver`. Without it, setting `checkpoint_root` raises a clear error.
- Everything for one task lives under `‹checkpoint_root›/‹task_id›/`: the durable
  `workspace/` and a `checkpoint.sqlite`. The graph state is checkpointed after each
  super-step; `thread_id = task_id`.
- **Resume:** re-dispatch the same `task_id`. The runtime reads `graph.get_state(cfg)`;
  if a node is pending it re-invokes with `None`, resuming from the **exact pending
  node** — prior nodes are **not** re-run. On success the whole durable dir is removed;
  on a crash it survives (the `finally` is skipped) for the next attempt.
- **Guarantee & caveat:** exactly-once for every node that completed *and was
  checkpointed*; only the single node in flight at crash time can re-run (LangGraph's
  at-least-once bound). `act` has no side effects, so a re-run there is free; a `tool`
  killed mid-side-effect is the irreducible case — `write_file` is idempotent,
  `shell`/`run_tests` are not, so make workspace side effects idempotent if that matters.
- **Interaction:** the work-queue lease-reaper can then *resume* a re-dispatched ticket
  (same `task_id`) rather than restart it.

### `prompts.py`

Assembles the runtime prompts.

- role instructions
- tool guidance
- task-specific context
- output contract

### `state.py`

Typed state passed through LangGraph.

- task identity
- workspace metadata
- tool outputs
- iteration count
- confidence / risks / evidence refs

### `results.py`

Converts final runtime state into the shared contract.

- `WorkerResult`
- evidence refs
- changed artifacts
- recommended next action

### `transport.py`

How the runtime is invoked remotely.

- `create_worker_app(runtime)` — FastAPI app: `POST /run` (task_id + `TaskSubmission` → `WorkerResult`), `GET /can_accept` (capacity/liveness probe), lock-guarded capacity counter
- `HttpAgentRuntime(base_url, tls=...)` — `AgentRuntime` client over HTTP; authentication is the mTLS client certificate
- TLS terminates at the server launcher (uvicorn `ssl_cert_reqs=CERT_REQUIRED`), never in the app factory; the wire never carries `RuntimeConfig`

The router can PUSH an agentic task to one of these workers instead of running the
loop in-process (`HttpAgentRuntime` is a drop-in `AgentRuntime`), gated by the same
fail-closed trust predicate as the pull queue. See
[REMOTE_DISPATCH.md](REMOTE_DISPATCH.md).

## Bring Your Own Runtime (public extension point)

The entire runtime contract is one structural protocol — implement it and the router
drives your code with no rewrite of your agent:

```python
class AgentRuntime(Protocol):
    def run(self, task: TaskSubmission, task_id: str = "task_local") -> WorkerResult: ...
```

If you already have a compiled **LangGraph** (or CrewAI, or hand-rolled) agent, wrap it
in a class whose `run()` invokes your graph and marshals the outcome into a
`WorkerResult`. Two ways to plug it in — both inherit AgentConnect's privacy×tier
routing, multi-harness managers, and federation, none of which the agent framework
provides:

- **In-process** — inject `local_runtime_factory=lambda source, config: YourRuntime(...)`
  into `RouterService.create(...)`. Agentic tasks then run through your runtime instead
  of the built-in `LangGraphAgentRuntime`. (`source` is a `ModelSource` bound to the
  routing decision; use it or ignore it if your graph owns its own model.)
- **Federated** — serve your runtime with `create_worker_app(your_runtime)` (a `POST
  /run` that takes `{task_id, submission}` → `WorkerResult`), register it in
  `remote_workers.yaml`, and the router pushes to it over mTLS.

Either way, a manager harness (Claude Code, Codex, Cursor, opencode) drives it end-to-end
through the vanilla `submit_task(execution="agentic")` MCP tool.

## How It Connects To The Router

The router already defines the control-plane contract:

- input: `TaskSubmission`
- worker output: `WorkerResult`
- stored result: shared memory artifacts and logs

The runtime should be wired behind the router, not inside it.

Suggested flow:

1. Claude submits a task to the router via MCP.
2. Router classifies, redacts, scores, and selects a backend.
3. Router dispatches to the runtime.
4. LangChain/LangGraph executes the work.
5. Runtime stores artifacts and returns `WorkerResult`.
6. Router persists the result and returns a compact `TaskSummary`.

## What Should Not Move Into The Runtime

Keep these in the router:

- privacy classification
- secrets resolution
- budget enforcement
- quota reservation
- provider selection
- mTLS and rental policy

Keep these out of the runtime unless they are explicitly task-local:

- global spend policy
- cloud provider credentials
- provider health scoring
- router memory records

## First Implementation

The first version should be narrow:

- one worker process
- one workspace
- one tool layer
- one LangGraph execution loop
- one model source at a time

That gives you a working agent runtime without overbuilding the first pass.

## Recommendation

Use LangChain + LangGraph as the runtime stack, but keep this repo’s architecture split intact:

- router = policy and routing
- runtime = execution
- model manager = local inference
- shared memory = state and artifacts

That gives you the ecosystem support you want without losing the deterministic boundary this repo is built around.
