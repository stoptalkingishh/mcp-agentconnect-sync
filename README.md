# mcp-agentconnect

A two-service agent infrastructure that lets **Claude Code** (or any manager
agent) delegate work through **MCP** while a deterministic control plane routes
each task to the best available model provider — local GPU, free-tier cloud, or
paid cloud — without wasting scarce quota or leaking sensitive context.

The point is **not** to bypass token costs. It is **hierarchical context
management**: the manager agent receives compact summaries and artifact
references, and reads full logs / diffs / outputs only when it explicitly needs
to — everything large lives in shared memory (see [context virtualization](#context-virtualization)).

```
                 ┌──────────────────────────────────────────────┐
   Claude Code   │            Agent Router MCP (Machine A)       │
   / manager ───▶│  classify → privacy/redact → eligible →       │
      agent      │  score → select → reserve → dispatch → store  │
                 │  (task DB · shared memory · quota · policy)    │
                 └───────────────┬───────────────┬──────────────┘
                                 │               │
                        secret_ref│resolved      │ HTTP /status /generate …
                          at call │ time         ▼
                 ┌───────────────▼──┐   ┌────────────────────────────────┐
                 │  Secrets Manager  │   │ Local Model Manager (Machine B)│
                 │ (1Password/Vault) │   │ residency · admission · GPU    │
                 └───────────────────┘   │ vLLM / llama.cpp / Ollama      │
                                         └────────────────────────────────┘
```

## Design rule (the one that matters)

> Router owns **decisions**. Model Manager owns local **execution**. Secrets
> Manager owns **credentials**. Agents own **task work**. Shared Memory owns
> **state and artifacts**. Claude owns high-level **management and selective
> inspection**.

Concretely: the inference machine never becomes the global policy engine, agents
never make infrastructure decisions, and **secrets never enter model-visible
context**.

## Repository layout

```
config/                      # policy & registry (edit these, not code)
  providers.yaml             #   §6  provider & model registry
  profiles.yaml              #   §17 capability profiles + agent defaults
  routing.yaml               #   §9,§12,§13,§16,§18 privacy, scoring, residency, limits
  secrets.example.yaml       #   §7  secret_ref → resolver mapping (copy to secrets.yaml)

src/agentconnect/
  common/                    # framework-free core (pydantic + pyyaml only)
    schemas.py               #   all data contracts
    state.py                 #   §19 deterministic task state machine
    memory.py                #   §8  shared memory + artifact store (SQLite)
    quota.py                 #   §15 quota reservation + reconciliation
    privacy.py               #   §13,§14 privacy classes + redaction layer
    providers.py             #   §6  provider registry access
    secrets.py               #   §7  secret resolution (only the gateway uses it)
    config.py / tokens.py    #   config loaders, token estimation
  router/                    # global control plane (Agent Router MCP)
    routing.py               #   §10,§11,§12 deterministic routing engine
    gateway.py               #   §7  secret-aware provider gateway
    service.py               #   §11 orchestration (submit_task … full flow)
    mcp_server.py            #   §23 MCP tools
    local_client.py          #   §5  client to the Local Model Manager
  model_manager/             # local inference control plane (appliance)
    residency.py             #   §16 residency + admission control
    backends.py              #   backend abstraction + deterministic StubBackend
    app.py                   #   §22 HTTP API (FastAPI)

tests/                       # 29 unit + e2e tests, run offline (stub backend)
examples/demo.py             # end-to-end walkthrough, no GPU required
docs/ARCHITECTURE.md         # detailed design notes + section map
```

## Quick start

```bash
pip install -e ".[all]"        # or: pip install -e ".[dev]" for just tests
pytest -q                      # 29 passing, fully offline
python examples/demo.py        # end-to-end: submit tasks, see compact summaries
```

### Run the two services

```bash
# Machine B — Local Model Manager (defaults to the deterministic stub backend)
agentconnect-model-manager               # serves http://0.0.0.0:8080

# Machine A — Agent Router MCP (stdio transport for Claude Code)
#   point it at the manager, or omit MODEL_MANAGER_URL to run one in-process
export MODEL_MANAGER_URL=http://machine-b:8080
export LOCAL_R9700_API_TOKEN=...         # only if the manager enforces auth
agentconnect-router
```

Register the router with Claude Code (`.mcp.json`):

```json
{
  "mcpServers": {
    "agentconnect": { "command": "agentconnect-router" }
  }
}
```

## MCP tools (§23)

| Tool | Returns |
|------|---------|
| `submit_task(task, agent_type, profile, privacy_class, …)` | compact `TaskSummary` + artifact refs |
| `get_task_status(task_id)` | status summary |
| `get_task_artifacts(task_id)` | `{kind: artifact_id}` |
| `read_artifact_chunk(artifact_id, offset, max_chars)` | bounded chunk + `next_offset` |
| `get_log_slice(task_id, level, query, max_lines)` | bounded log slice |
| `search_memory(query, scope, limit)` | snippets (never full bodies) |
| `get_router_status()` / `get_provider_status()` | policy + live provider/quota state |
| `promote_task(task_id)` / `cancel_task(task_id)` | control |

## Context virtualization

Workers write full output (patches, logs, traces) to shared memory and return
only status + summary + refs + risks + next action. The router's default output
policy (`config/routing.yaml`) caps MCP payloads and forbids returning full logs,
full repo files, or full traces inline — the manager pulls detail on demand with
`read_artifact_chunk` / `get_log_slice`.

## Privacy & secrets (fail-closed)

- Tasks are classified into `public / low_sensitive / repo_sensitive /
  secret_sensitive / restricted` (§13). A redaction pass (§14) scrubs
  keys/JWTs/DB-URLs/PII before any external call.
- `secret_sensitive` content is **blocked from every LLM** — the router rejects
  it before routing.
- Provider configs hold **secret references only**. The gateway is the sole
  component that resolves a secret, at call time, and never returns/logs it.

## Status vs. the phased plan (§25)

Implemented end-to-end (offline, tested): the deterministic router (Phases 1 &
5 core), shared memory + context virtualization (Phase 2), residency + admission
(Phase 3), the provider gateway + secrets + quota ledger + privacy/redaction
(Phase 4). Cloud calls degrade to a deterministic stub until real credentials are
supplied; Phase 6 (learning/eval) is scaffolded via the quota/usage records but
not yet scored. See `docs/ARCHITECTURE.md` for the section-by-section map.

## License

MIT
