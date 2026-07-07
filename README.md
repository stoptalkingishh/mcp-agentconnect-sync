# mcp-agentconnect

**TL;DR** — Give **Claude Code** (or any agent) a set of MCP tools that hand tasks
off to a control plane. It classifies each task, keeps sensitive context out of
the wrong models, and routes to the cheapest capable model — your local GPU, a
free cloud tier, or paid cloud — then returns a **compact summary + artifact
references** instead of dumping everything back into context. Runs end-to-end
**today with no GPU** (a built-in stub model); plug in one model server for real
output.

## Get set up (≈2 minutes, no GPU)

```bash
git clone <this-repo> && cd mcp-agentconnect
python -m venv .venv && source .venv/bin/activate
pip install -e packages/agentconnect-core \
            -e packages/agentconnect-router \
            -e packages/agentconnect-model-manager \
            -e packages/agentconnect-runtime
pytest -q          # 333 passing, fully offline — confirms the install works
```

That installs the `agentconnect-router` command. You're done — nothing else is
required to try it.

## Use it

**1. See it work — no config, no GPU:**

```bash
python examples/demo.py             # submit tasks; watch classify → route → summarize
python examples/federation_demo.py  # a friend's box drains a shared queue, privacy enforced
```

**2. Wire it into Claude Code** — add to your `.mcp.json`:

```json
{ "mcpServers": { "agentconnect": { "command": "agentconnect-router" } } }
```

Claude Code now has tools like `submit_task`, `queue_add`, and
`read_artifact_chunk` ([full list below](#mcp-tools-23)). Out of the box, tasks
route to a **built-in stub model** (deterministic echo), so the entire pipeline
works with zero infrastructure.

**3. Get real model output** — point the Model Manager at any OpenAI-compatible
server (Ollama, vLLM, llama.cpp, SGLang). On a single box the Router embeds the
manager, so no TLS/second process is needed:

```bash
export MODEL_MANAGER_BACKEND=openai
export MODEL_BACKEND_URL=http://localhost:11434/v1   # e.g. Ollama
agentconnect-router
```

That model server is the **only** external thing you supply. Free/paid cloud
providers, a separate GPU box over mutual TLS, and remote-worker dispatch are all
optional — see [Production](#production-two-machines-over-mutual-tls) and the
sections below.

---

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

Four independently-installable packages in one repo (a PEP 420 namespace, so
every import path stays `agentconnect.*`):

```
config/                      # policy & registry (edit these, not code)
  providers.yaml             #   §6  provider & node registry (local nodes = mTLS, no secret)
  profiles.yaml              #   §17 capability profiles + agent defaults
  routing.yaml               #   §9,§12,§13,§16 privacy tiers, scoring, residency, rental
  secrets.example.yaml       #   §7  CLOUD secret_ref → resolver (local nodes carry none)

packages/
  agentconnect-core/         # shared, framework-free (pydantic + pyyaml only)
    …/common/{schemas,state,memory,quota,privacy,providers,secrets,config,tokens,workqueue}.py
  agentconnect-router/       # the PRIMARY product — Agent Router MCP control plane
    …/router/{routing,gateway,service,mcp_server,local_client,provisioning}.py
  agentconnect-model-manager/# optional satellite — local inference appliance
    …/model_manager/{residency,backends,app,tls}.py
  agentconnect-runtime/      # worker runtime — LangGraph act/tool execution loop
    …/runtime/{agent,graph,tools,workspace,prompts,state,results,transport}.py

tests/                       # 333 unit + e2e tests, run offline (stub backend + mTLS)
examples/demo.py             # end-to-end router walkthrough, no GPU required
examples/federation_demo.py  # federated work queue: friend contributes compute, privacy enforced
docs/ARCHITECTURE.md         # detailed design notes + section map
docs/WORK_QUEUE.md           # federated pull-based work queue design
docs/REMOTE_DISPATCH.md      # router-driven remote-worker push dispatch
```

## Production: two machines over mutual TLS

For a real deployment you split the Router (Machine A, decisions) from the Local
Model Manager (Machine B, a GPU inference appliance). They authenticate by
**client certificate — no bearer token or shared secret crosses the wire**:

```bash
# Machine B — Local Model Manager (serves HTTPS + requires a client cert)
export MODEL_MANAGER_TLS_CERT=/certs/server.crt
export MODEL_MANAGER_TLS_KEY=/certs/server.key
export MODEL_MANAGER_TLS_CA=/certs/ca.crt          # trust anchor for router client certs
export MODEL_MANAGER_ALLOWED_CLIENTS=agentconnect-router-01   # optional identity allowlist
export MODEL_MANAGER_BACKEND=openai                # front a real vLLM/llama.cpp/Ollama server
export MODEL_BACKEND_URL=http://localhost:8000/v1
agentconnect-model-manager                         # serves https://0.0.0.0:8443 (mTLS)

# Machine A — Agent Router MCP (stdio transport for Claude Code)
export MODEL_MANAGER_URL=https://machine-b:8443
export AGENTCONNECT_LOCAL_CA=/certs/ca.crt
export AGENTCONNECT_LOCAL_CLIENT_CERT=/certs/router.crt
export AGENTCONNECT_LOCAL_CLIENT_KEY=/certs/router.key
agentconnect-router
```

The Router also runs **cloud-only, without the Model Manager** at all — install
just `agentconnect-core` + `agentconnect-router`; local-only tasks then report
"no local node" and everything else routes to cloud providers.

## MCP tools (§23)

| Tool | Returns |
|------|---------|
| `submit_task(task, agent_type, profile, privacy_class, …)` | compact `TaskSummary` + artifact refs |
| `queue_add(task, privacy_class, …)` | ticket + status |
| `queue_next(worker_id, capabilities)` | list of claimable tickets |
| `queue_claim(worker_id, ticket_id)` | claimed ticket or error |
| `queue_report(worker_id, ticket_id, lease_token, result)` | ticket/result status |
| `queue_status(ticket_id, …)` | ticket metadata + audit trail |
| `queue_list(status, privacy_class, …)` | payload-free ticket listing (operator view) |
| `queue_pending(limit)` | in_review backlog (operator view) |
| `queue_stats()` | queue counts + capability requirements (operator view) |
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

## Federated work queue (pull-based, open surface)

Beyond routing, AgentConnect offers a **pull-based work queue** where separate agents and
untrusted external compute can discover and claim work — all while respecting the
trust × privacy boundary. Workers pull tickets for the privacy class they are authorized
for; results from untrusted workers land in a review gate before becoming truth.

| Tier | Who | Can claim | Auto-trusted |
|---|---|---|---|
| `local_only` | your box | all classes | yes — auto-approve results |
| `private_rented` | rented GPU, your model | public, low_sensitive | no — results require review |
| `external` | free cloud API | public, low_sensitive (redacted) | no — results require review |
| `external_paid` | paid cloud API | public, low_sensitive (redacted) | no — results require review |

- **Atomicity & fencing:** One shared SQLite store; claim is a single guarded `UPDATE`. Leases expire; the reaper requeues, and fresh tokens prevent stale workers from submitting results.
- **Result verification:** Results from untrusted tiers land `in_review` until a `local_only` reviewer approves.
- **Dependencies:** Tickets can depend on other tickets. A child is claimable only when all parents are `done`. Privacy monotonicity is enforced: a lower-class ticket cannot depend on a higher-class parent (prevents laundering).
- **Idempotency:** Deduplicated by key; terminal states (done, failed) are never reopened. Already-reported tickets are refused.

See [docs/WORK_QUEUE.md](docs/WORK_QUEUE.md) for the full design, MCP API, and examples.

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

### Agentic execution on the rented tier

`execution="agentic"` runs the worker runtime's act/tool loop in-process, feeding
each tool observation back to the model. Those observations must never reach an
**untrusted external** model, so agentic runs only on (a) an owned-local resident
model, or (b) a **trusted, opted-in rented private node** — a box running your
weights ephemerally with no external logging (exactly the rented tier's purpose).
Cloud (`external`/`external_paid`) agentic is always rejected; `secret_sensitive`
never routes at all. The guard fails closed even if routing selected rented for a
public task without `allow_rented` or without a trust-satisfying node.

On the rented tier the loop dispatches every step through the rented path, not the
gateway: the node is **acquired once** before the loop, **reused across all steps**,
its rental window **billed exactly once** at spin-up, and it is **released** (for
the idle reaper) in a `finally`. Token usage is summed across steps for the
evaluation record only — the single window bill is the whole money path (a rented
node is `type="local"`, so no cloud quota is reserved or reconciled).

### Router-driven remote-worker dispatch (push)

Instead of running the agentic loop in-process, the router can PUSH the whole task
to a **registered remote worker** over mutual TLS (`HttpAgentRuntime` → `POST /run`),
automatically preferring any worker whose **attested tier** is trusted for the task's
privacy class (the same fail-closed `WorkQueue.may_claim` predicate the pull queue
uses) and that reports capacity — else it falls back to in-process. The worker runs
its own model and **self-reports token usage**; there is no router-side spend. Register
workers in `config/remote_workers.yaml` (ships empty → feature off). See
[docs/REMOTE_DISPATCH.md](docs/REMOTE_DISPATCH.md).

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

### Batteries-included web approval host

Don't want to build a confirmation channel? Turn on the reference one and you get a
browser approvals page plus (optionally) phone push. There are three levels — pick one.

**Level 1 — browser only (local machine):**

```bash
pip install "agentconnect-router[web]"
export AGENTCONNECT_SPEND_AUTHORIZER=web
agentconnect-router
```

Open `http://127.0.0.1:8770/`. When the agent tries a paid/rented task, the call blocks,
a `SPEND APPROVAL NEEDED …` line is logged with a link, and the page shows **Approve /
Deny** buttons (or, if no budget is set yet, an amount box). Click it and the task
proceeds. No response within 5 minutes = denied.

**Level 2 — phone push with one-tap approve (recommended): ntfy.** Three steps:

1. Install the free **ntfy** app (iOS/Android) and subscribe to a topic name you choose,
   e.g. `agentconnect-yourname` (any hard-to-guess string).
2. Make the approval server reachable from your phone. On a laptop the quickest way is a
   tunnel, e.g. `cloudflared tunnel --url http://localhost:8770` — copy the public
   `https://…` URL it prints.
3. Run the router pointed at both:

   ```bash
   pip install "agentconnect-router[web]"
   export AGENTCONNECT_SPEND_AUTHORIZER=web
   export AGENTCONNECT_APPROVAL_URL=https://YOUR-TUNNEL.trycloudflare.com  # from step 2
   export AGENTCONNECT_NOTIFY=ntfy
   export AGENTCONNECT_NTFY_URL=https://ntfy.sh/agentconnect-yourname      # your topic
   agentconnect-router
   ```

   Now every paid/rented charge pushes a notification to your phone with **Approve** and
   **Deny** buttons — tapping one POSTs straight to the server. (One-tap works because the
   ntfy app makes the call; that's why the server must be reachable at
   `AGENTCONNECT_APPROVAL_URL`.)

**Level 3 — Slack / Discord.** Set `AGENTCONNECT_NOTIFY=slack` (or `discord`) with
`AGENTCONNECT_SLACK_WEBHOOK` / `AGENTCONNECT_DISCORD_WEBHOOK`. You get a rich message with
a one-tap **link** to a per-item page where you confirm (incoming webhooks can't do true
in-message buttons — only ntfy can). Combine channels with a comma:
`AGENTCONNECT_NOTIFY=ntfy,slack`.

**All environment variables** (mode is set by `AGENTCONNECT_SPEND_AUTHORIZER`):

| Variable | Default | Meaning |
|---|---|---|
| `AGENTCONNECT_SPEND_AUTHORIZER` | `deny` | `deny` (fail-closed) · `web` · `console` · `auto` (trusted/tests) |
| `AGENTCONNECT_APPROVAL_HOST` / `_PORT` | `127.0.0.1` / `8770` | where the approvals server binds |
| `AGENTCONNECT_APPROVAL_URL` | `http://host:port` | public base URL used in notifications (set to your tunnel) |
| `AGENTCONNECT_APPROVAL_TOKEN` | *(none)* | optional bearer token required on `/api/*` |
| `AGENTCONNECT_APPROVAL_TIMEOUT` | `300` | seconds to wait before failing closed (deny) |
| `AGENTCONNECT_NOTIFY` | *(none)* | comma list: `ntfy` · `slack` · `discord` · `webhook` |
| `AGENTCONNECT_NTFY_URL` | — | your ntfy topic URL, e.g. `https://ntfy.sh/…` |
| `AGENTCONNECT_SLACK_WEBHOOK` / `_DISCORD_WEBHOOK` | — | incoming-webhook URLs |
| `AGENTCONNECT_APPROVAL_WEBHOOK` | — | raw JSON POST target (for `webhook` mode) |

**Security:** the approvals endpoint controls money. It binds loopback by default; before
exposing it (e.g. via a tunnel) set `AGENTCONNECT_APPROVAL_TOKEN` and keep it over HTTPS.
Without the `[web]` extra installed, `web` mode logs a warning and falls back to `deny`.

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

**All six phases implemented end-to-end (offline, tested — 333 tests):** the
deterministic router (Phases 1 & 5), shared memory + context virtualization
(Phase 2), residency + **real concurrency admission** (Phase 3), the provider
gateway + secrets + quota ledger + privacy/redaction (Phase 4), and
**evaluation + learning** (Phase 6): outcomes are recorded per provider and a
bounded learned-quality signal tilts future routing. Plus the cross-cutting work:
**mutual-TLS inter-service transport**, the **three-package split** (Router
installs without the Manager), the **rented-GPU node tier** with a **RunPod vendor
adapter**, warm-node reuse + idle reaping, and a **real OpenAI-compatible inference
backend** (vLLM / llama.cpp / Ollama). **New in this phase:** agentic execution on
trusted rented private nodes (opt-in, acquire-once-bill-once), capability matching
(ticket-to-worker filter, not a gate), and broker-side operator view (queue
visibility + in_review backlog with approve/reject web host). Cloud calls and the
local backend degrade to deterministic stubs until real endpoints/credentials are
supplied. CI runs the suite on 3.10–3.12 and verifies the Router builds standalone.
See `docs/ARCHITECTURE.md` for the section-by-section map.

## License

MIT
