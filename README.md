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

Three independently-installable packages in one repo (a PEP 420 namespace, so
every import path stays `agentconnect.*`):

```
config/                      # policy & registry (edit these, not code)
  providers.yaml             #   §6  provider & node registry (local nodes = mTLS, no secret)
  profiles.yaml              #   §17 capability profiles + agent defaults
  routing.yaml               #   §9,§12,§13,§16 privacy tiers, scoring, residency, rental
  secrets.example.yaml       #   §7  CLOUD secret_ref → resolver (local nodes carry none)

packages/
  agentconnect-core/         # shared, framework-free (pydantic + pyyaml only)
    …/common/{schemas,state,memory,quota,privacy,providers,secrets,config,tokens}.py
  agentconnect-router/       # the PRIMARY product — Agent Router MCP control plane
    …/router/{routing,gateway,service,mcp_server,local_client,provisioning}.py
  agentconnect-model-manager/# optional satellite — local inference appliance
    …/model_manager/{residency,backends,app,tls}.py

tests/                       # 43 unit + e2e tests, run offline (stub backend + mTLS)
examples/demo.py             # end-to-end walkthrough, no GPU required
docs/ARCHITECTURE.md         # detailed design notes + section map
```

## Quick start

```bash
pip install -e packages/agentconnect-core \
            -e packages/agentconnect-router \
            -e packages/agentconnect-model-manager
pytest -q                      # 43 passing, fully offline
python examples/demo.py        # end-to-end: submit tasks, see compact summaries
```

The Router is the product and installs **without** the Model Manager:

```bash
pip install -e packages/agentconnect-core -e packages/agentconnect-router
agentconnect-router            # cloud-only standalone; local-only tasks report "no local node"
```

### Run the two services (mutual TLS, no shared secret)

```bash
# Machine B — Local Model Manager (serves HTTPS + requires a client cert)
export MODEL_MANAGER_TLS_CERT=/certs/server.crt
export MODEL_MANAGER_TLS_KEY=/certs/server.key
export MODEL_MANAGER_TLS_CA=/certs/ca.crt          # trust anchor for router client certs
export MODEL_MANAGER_ALLOWED_CLIENTS=agentconnect-router-01   # optional identity allowlist
agentconnect-model-manager                         # serves https://0.0.0.0:8443 (mTLS)

# Machine A — Agent Router MCP (stdio transport for Claude Code)
export MODEL_MANAGER_URL=https://machine-b:8443
export AGENTCONNECT_LOCAL_CA=/certs/ca.crt
export AGENTCONNECT_LOCAL_CLIENT_CERT=/certs/router.crt
export AGENTCONNECT_LOCAL_CLIENT_KEY=/certs/router.key
agentconnect-router
```

Identity is the certificate — **no bearer token or shared secret crosses the
wire**. For single-box dev, omit `MODEL_MANAGER_URL` and the Router embeds an
in-process manager (needs no transport at all).

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
| `get_provider_scorecards()` | learned per-provider quality/latency (Phase 6) |
| `set_budget(amount_usd, period)` / `get_budget_status()` | global spend budget + pacing |
| `promote_task(task_id)` / `cancel_task(task_id)` | control |

## Context virtualization

Workers write full output (patches, logs, traces) to shared memory and return
only status + summary + refs + risks + next action. The router's default output
policy (`config/routing.yaml`) caps MCP payloads and forbids returning full logs,
full repo files, or full traces inline — the manager pulls detail on demand with
`read_artifact_chunk` / `get_log_slice`.

## Three-tier compute (owned · rented · cloud)

A rented GPU running **your** open-weights model is a distinct tier — private
inference for very large models without owning the hardware. It plugs in as *just
another Model Manager node*, reached over the same mTLS transport:

| Tier | Hardware | Model | Privacy tier | Cost |
|---|---|---|---|---|
| Local node | your box | small/mid, yours | `local_only` | free (owned) |
| Rented node | rented GPU | **very large, yours** | `private_rented` | hourly rent + spin-up |
| Cloud API | provider's | *their* model | `external` / `external_paid` | per-token |

Renting has **two credential planes**: inference traffic is mTLS (no secret); the
rental vendor's control-plane API key is the only secret and lives in the secrets
manager, used solely to rent/terminate the box. A `repo_sensitive` task may run on
a rented node only with explicit opt-in (`allow_rented`) **and** when the node
meets its trust policy (ephemeral, encrypted, your image, no external logging).

## Spend budget & direct user authorization (fail-closed on money)

Set one number and a period — `set_budget(amount_usd, "daily"|"weekly"|"monthly")` —
and the router **paces** spend against it: as cumulative spend runs ahead of the
even-burn line (or nears the cap) it steers toward free/local via a scoring penalty,
and it hard-blocks paid/rented when the period budget is exhausted. Spend is metered
across all real money (paid cloud + rented GPU) from the one `quota_records` ledger.

Two deliberate safety properties:

- **Mandatory, no silent default.** There is no default amount. Until the user
  explicitly sets a budget, paid cloud and rented GPU are **ineligible** — the system
  still runs fully on free-tier/owned-local, but real money is off until asked for.
- **Deterministic human gate on every charge.** Money never depends on the stochastic
  agent. The router calls a `SpendAuthorizer` *directly*: `request_budget` prompts the
  user to set a budget when none exists, and `confirm_charge` asks the user to approve
  **each** paid/rented charge before it happens. The default is `Deny` (fail-closed);
  wire `CallbackSpendAuthorizer` to your app's native confirmation UI, or use
  `Console`/`AutoApprove` for CLI/trusted automation. This bounds the "stochastic blast
  zone": the agent can propose work, but the user approves the spend.

## Privacy & secrets (fail-closed)

- Tasks are classified into `public / low_sensitive / repo_sensitive /
  secret_sensitive / restricted` (§13). A redaction pass (§14) scrubs
  keys/JWTs/DB-URLs/PII before any external call.
- `secret_sensitive` content is **blocked from every LLM/node** — the router
  rejects it before routing.
- **Local + rented inference nodes carry no secret at all** — they authenticate
  via mutual TLS (identity = client certificate). Only the third-party secrets
  manager holds secrets, and only cloud API keys + rental control-plane keys.
- Cloud provider configs hold **secret references only**. The gateway is the sole
  component that resolves a cloud secret, at call time, and never returns/logs it.

## Status vs. the phased plan (§25)

A **global spend budget** with even-burn pacing and a **direct-to-user spend
authorizer** (mandatory budget, per-charge confirmation — money never rides on the
agent) sit on top of all six phases.

**All six phases implemented end-to-end (offline, tested — 72 tests):** the
deterministic router (Phases 1 & 5), shared memory + context virtualization
(Phase 2), residency + **real concurrency admission** (Phase 3), the provider
gateway + secrets + quota ledger + privacy/redaction (Phase 4), and
**evaluation + learning** (Phase 6): outcomes are recorded per provider and a
bounded learned-quality signal tilts future routing. Plus the cross-cutting work:
**mutual-TLS inter-service transport**, the **three-package split** (Router
installs without the Manager), the **rented-GPU node tier** with a **RunPod vendor
adapter**, warm-node reuse + idle reaping, and a **real OpenAI-compatible inference
backend** (vLLM / llama.cpp / Ollama). Cloud calls and the local backend degrade to
deterministic stubs until real endpoints/credentials are supplied. CI runs the
suite on 3.10–3.12 and verifies the Router builds standalone. See
`docs/ARCHITECTURE.md` for the section-by-section map.

## License

MIT
