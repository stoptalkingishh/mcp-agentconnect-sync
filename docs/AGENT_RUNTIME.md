# Agent Runtime

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

Defines how the runtime is invoked.

- local subprocess
- HTTP worker service
- future distributed execution

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
