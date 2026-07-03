# agentconnect-runtime

Skeleton package for the AgentConnect worker runtime.

This package is the execution layer that will sit behind the router and use
LangChain + LangGraph for tool-using agent execution. It is intentionally
lightweight for now: the repo has the control plane and contracts, but not the
full worker implementation yet.

Planned responsibilities:

- task execution
- workspace management
- tool use
- graph-based agent loops
- `WorkerResult` assembly

See the repository-level [docs/AGENT_RUNTIME.md](../../docs/AGENT_RUNTIME.md).
