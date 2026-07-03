# Architecture

This document maps the [handoff specification](../README.md) to the code and
explains the boundaries that keep the system deterministic and safe. Section
numbers (§) refer to the original handoff.

## Four packages, one shared core

```
packages/
  agentconnect-core/          ← shared core: policy-agnostic (pydantic+pyyaml)  → agentconnect.common.*
  agentconnect-router/        ← Machine A: Agent Router MCP — control plane (§4.1) → agentconnect.router.*
  agentconnect-model-manager/ ← Machine B: Local Model Manager — appliance (§4.2)  → agentconnect.model_manager.*
  agentconnect-runtime/       ← Agent runtime — LangChain/LangGraph worker stack  → agentconnect.runtime.*
```

They share the `agentconnect` **PEP 420 namespace** (no top-level `__init__.py`),
so import paths are unchanged even though the code lives in three distributions.
The **Router is the primary product** and depends only on `agentconnect-core`; the
Model Manager is an optional install (`agentconnect-router[embedded]`). The core
depends on nothing heavy; the MCP SDK and FastAPI are imported lazily inside
`router/mcp_server.py` and `model_manager/app.py`.

### Inter-service transport: mutual TLS, no shared secret (§7)

The Router↔Manager link authenticates with **X.509 client/server certificates**
signed by a private CA — identity is the certificate, so no bearer token or shared
secret crosses the wire. `HttpLocalClient` builds an `ssl.SSLContext` that pins the
manager's server cert to the CA and presents the router's client cert; the manager
launches uvicorn with `ssl_cert_reqs=CERT_REQUIRED + ssl_ca_certs`, so the
handshake fails for any client not signed by the trusted CA. Cert/key material is
referenced by **path** (env-expandable) in `providers.yaml` `tls`, never inlined.
`ClientIdentityMiddleware` adds an optional per-CN/SAN allowlist (defense in depth;
see `model_manager/tls.py` for the uvicorn ASGI-TLS caveat). The `SecretResolver`
now serves **cloud providers only**.

### Rented-GPU node tier (Goal 4)

A rented box runs the **same** `agentconnect-model-manager` and is reached over the
**same mTLS transport** — it is just another node with a `private_rented` privacy
tier, an hourly cost model (`quota.py` `rented_gpu`, budgeted by
`rental.max_daily_usd`), and a lifecycle (`router/provisioning.py`:
`NodeProvisioner` + offline `StubProvisioner`). Two credential planes: inference =
mTLS (no secret); the rental vendor control-plane key = the only secret, in the
secrets manager, used solely to rent/terminate. A `repo_sensitive` task reaches a
rented node only with explicit `allow_rented` **and** a satisfied trust policy;
`secret_sensitive` never. Scoring adds `rental_setup_penalty` (spin-up/min-window)
and `rental_cost_penalty` (hourly vs. budget), so the router rents only when the
work justifies amortizing the window — the model-switch "is the setup worth it?"
logic, one level up.

### Responsibility split (§26 — the most important rule)

| Concern | Owner | Where |
|---|---|---|
| What should happen (decisions) | Agent Router | `router/routing.py`, `router/service.py` |
| How the task is executed | Agent Runtime | `runtime/agent.py`, `runtime/graph.py` |
| Local execution (residency, generation) | Local Model Manager | `model_manager/residency.py` |
| Credentials | Secrets Manager (via gateway) | `common/secrets.py`, `router/gateway.py` |
| State + artifacts | Shared Memory | `common/memory.py` |
| Task work | Agents | worker return contract in `common/schemas.py` |

Guardrails enforced by structure:
- The Model Manager imports **nothing** from `router/` — it cannot become the
  policy engine.
- The Agent Runtime should not decide global policy; it executes tasks under
  router control.
- Only `router/gateway.py` constructs a `SecretResolver`; secrets never reach the
  agent/MCP layer.

## The deterministic routing flow (§10, §11)

`RouterService.submit_task` (`router/service.py`) implements the numbered flow:

1. **Receive + assign id** → `SharedMemory.create_task`
2. **Classify** → resolve a capability profile (`profiles.yaml` `agent_defaults`)
3. **Privacy class** → `privacy.classify` (§13)
4. **Redaction pass** → `privacy.redact`, stored as a `sanitized_payload`
   artifact (§14). `secret_sensitive` → **fail closed** (REJECTED, never routed).
5. **Quality + token estimate** → `tokens.estimate_io_tokens`
6. **Fetch local status** → `LocalClient.status()` (§5)
7. **Eligibility** → `RoutingEngine.eligibility` applies HARD constraints (§12)
8. **Score** → `RoutingEngine.score`, sort descending
9. **Select** → highest score; build an explainable `RoutingDecision`
10. **Reserve** → `QuotaLedger.reserve` (cloud) / admission (local) (§15)
11. **Dispatch** → `ProviderGateway.call` (§7)
12. **Store** full output in shared memory; **return** compact summary + refs
13. **Log** the routing decision (already recorded, queryable)

Every step is a pure function of `(task, config, live status)`. There is no
randomness in the control plane (§10) — only inside model generation.

### Scoring (§12)

```
score = capability_fit + expected_quality + latency_fit + privacy_fit
      + availability + residency_bonus
      − quota_scarcity_penalty − queue_delay_penalty − model_switch_penalty
      − cost_penalty − opportunity_cost
```

Weights live in `config/routing.yaml` → `scoring.weights`. **Hard constraints
always override scoring** — an ineligible provider never receives a score, it
lands in `rejected_options` with a machine-readable reason. This is why a
`repo_sensitive` task can never be scored onto a cloud provider even if the cloud
score would be higher.

The objective (§12) is *lowest opportunity cost that still satisfies quality,
privacy, latency and context* — not "always local" or "always cloud". Example:
a **public** task the resident model can handle stays local, because using a
scarce free-tier quota for it incurs `opportunity_cost` while the resident model
incurs none.

## Privacy classes → provider tiers (§13)

`config/routing.yaml` → `privacy.classes` is the authoritative table:

| Task class | Allowed provider tiers |
|---|---|
| `public` | local, external, external_paid |
| `low_sensitive` | local, external, external_paid *(after redaction)* |
| `repo_sensitive` | local only |
| `secret_sensitive` | **none** — blocked from any LLM |
| `restricted` | local only |

The redaction layer (`common/privacy.py`) detects API keys, JWTs, SSH keys,
DB URLs, bearer tokens, emails, internal hostnames, and local paths. If a hard
secret is present the payload is marked **not cloud-safe** and the router keeps
it local or blocks it (§14: "if redaction is too destructive, route local").

## Model residency (§16) & profiles (§17)

Agents request a **profile** (`resident_ok`, `coding_patch`, …), never a raw
model name. `RoutingEngine.resolve_local_model` resolves the profile against the
Manager's live status:

- `prefer_resident_model` → if the loaded model satisfies the profile, use it
  (no switch penalty).
- A specialist that isn't loaded pays the full `model_switch_penalty` **unless**
  the switch policy allows it: urgent priority, or enough same-model work has
  accumulated (`min_batch_size_for_switch`), and the current queue is empty.

This prevents the model-thrashing anti-pattern in §16.

## Local Model Manager API (§22)

`model_manager/app.py` exposes `/status /models /queue /metrics /can_accept
/generate /load /unload`. `ResidencyManager` (`model_manager/residency.py`) owns
admission control: context-cap checks against `max_model_len`, switch detection,
and **real active-sequence limiting** — a `BoundedSemaphore(max_active_sequences)`
makes `generate()` block/queue when the GPU is full rather than oversubscribe it
(§18). The backend is pluggable (`ModelBackend`): `StubBackend` (deterministic,
offline) and `OpenAICompatibleBackend` (real vLLM / llama.cpp / Ollama / SGLang /
TGI, selected via `MODEL_MANAGER_BACKEND=openai` + `MODEL_BACKEND_URL`).

Authentication is **mutual TLS** — no shared bearer token (see the transport
section above and `model_manager/tls.py`).

## Evaluation & learning (§25 Phase 6)

Every dispatch records an outcome (`common/memory.py` `evaluations`: provider,
model, status, latency, tokens, cost, confidence). `common/evaluation.py`
aggregates these into per-provider **scorecards** and a bounded `[-1, 1]`
**learned-quality signal** (success rate primary, relative latency secondary; zero
until `learned_min_samples` observations). The service refreshes the signal before
each routing pass and the engine folds it in as one more scoring term
(`learned_quality` weight in `routing.yaml`) — so observed quality tilts close
calls without overriding hard constraints or dominating. Surfaced via the
`get_provider_scorecards` MCP tool. This keeps learning *inside* the deterministic
frame: outcomes adjust a prior; they never make routing stochastic.

## Shared memory & context virtualization (§8, §9)

`common/memory.py` is a SQLite store for tasks, artifacts, logs, routing
decisions, and quota records. Large outputs are stored here and **never returned
inline** through MCP. The manager reads detail on demand with
`read_artifact_chunk` (paginated via `next_offset`) and `get_log_slice`
(level/query filtered). `config/routing.yaml` → `mcp_output_policy` caps payload
sizes and forbids returning full logs/files/traces.

## Quota (§15)

`common/quota.py` implements estimate → check → reserve → call → reconcile →
release. Reservations are held in-process (fast, deterministic) so concurrent
agents can't oversubscribe a shared free tier; committed usage is persisted to
shared memory and is what daily-limit math reads. Paid providers additionally
enforce `max_daily_spend_usd`.

## Global spend budget + direct user authorization

Two layers govern real money (paid cloud + rented GPU); free/owned-local are always $0.

**Budget & pacing** (`common/budget.py`, `common/memory.py`). `BudgetManager` reads a
single user-set budget (`settings["budget"]`: amount + daily/weekly/monthly) and meters
it against `total_spend_since(period_start)` — the all-provider sum of committed
`act_cost_usd`. It exposes `remaining`, `paced_allowance` (straight-line even-burn),
`pressure` ∈ [0,1] (max of cap-proximity and ahead-of-pace), and `can_afford`. The
service refreshes a snapshot into the engine each pass (`set_budget_state`), which adds a
hard eligibility gate (`period_budget_exhausted`) and a soft `budget_pressure_penalty`
scoring term for paid/rented. **Mandatory, no silent default:** there is no default
amount; until the user sets one, paid/rented are rejected `budget_not_configured` and the
system runs on free/local only.

**Direct user authorization** (`common/authorization.py`). Money decisions are kept off
the stochastic agent. The service calls a `SpendAuthorizer` directly: `request_budget`
(prompt the user to set a budget when none exists — deterministic, then the router
re-routes) and `confirm_charge` (approve *each* paid/rented charge before it happens).
The default `DenyingSpendAuthorizer` is fail-closed (no channel wired → no spend);
deployments wire `CallbackSpendAuthorizer` to a native UI, or use `Console`/`AutoApprove`.
A **batteries-included web host** ships too: `common/approval.py` (`ApprovalQueue` +
`WebApprovalAuthorizer`, stdlib-only, blocking-with-timeout) plus `router/approval_web.py`
(a FastAPI approve/deny dashboard + `start_web_approval`, the `[web]` extra). Enable with
`AGENTCONNECT_SPEND_AUTHORIZER=web`; the router blocks the `submit_task` call, logs an
approval URL, and the user clicks Approve/Deny in a browser — loopback + optional bearer
token, fail-closed on timeout. **Push notifiers** (`common/notifiers.py`: ntfy/Slack/
Discord/raw-webhook + `MultiNotifier`, selected via `AGENTCONNECT_NOTIFY`) send the pending
charge to a phone/chat — ntfy renders true one-tap POST Approve/Deny actions; Slack/Discord
deep-link to a per-item action page (`/a/{id}`) since incoming webhooks can't do callbacks.
The charge gate runs *before* the `QUEUED` transition so a decline is a legal
`ELIGIBLE_PROVIDERS_COMPUTED → REJECTED` (the same reorder fixed a latent illegal
`QUEUED → REJECTED` on quota-reservation denial). `set_budget`/`get_budget_status` MCP
tools expose it; `get_router_status().budget.action_required` flags "set_budget".

## Mapping to the phased plan (§25)

| Phase | Status | Notes |
|---|---|---|
| 1 Minimal local router | ✅ | router + task DB + artifact store + local status + local routing |
| 2 Shared memory & virtualization | ✅ | chunked reads, summaries, state machine |
| 3 Model residency | ✅ | inventory, load/unload, switch policy, admission |
| 4 Cloud gateway | ✅ | registry, secrets, quota ledger, privacy gates, redaction |
| 5 Workload-aware routing | ✅ | scarcity/opportunity/queue/switch/rental scoring, budget enforcement |
| 6 Evaluation & learning | ✅ | outcome ledger, per-provider scorecards, bounded learned-quality signal in routing |

Cross-cutting completions: mutual-TLS transport, the three-package split, the
rented-GPU tier (RunPod adapter + `NodePool` warm reuse + idle reaping), a real
OpenAI-compatible backend, real concurrency admission, **federated work queue** (pull-based
open surface with trust × privacy boundary), and CI (3.10–3.12 + standalone-router job).

Cloud generation and the local backend degrade to deterministic stubs when
endpoints/credentials are absent, so the full pipeline is exercisable offline;
supply real endpoints/keys (env or `config/secrets.yaml`) to make live calls.

## Federated work queue (pull-based surface)

Beyond push-based routing, AgentConnect exposes a **pull-based work queue** where external
agents and untrusted compute can discover, claim, and complete work. The queue lives in
the same SQLite store (one transactional `SharedMemory._conn`) alongside task/artifact
state, so all state is atomic and audited.

The key invariant is the **trust × privacy boundary** (`common/workqueue.py` module
docstring + `docs/WORK_QUEUE.md`):

- A worker's attested tier (not self-declared) determines which privacy classes it may claim.
- Tickets are stored `parked` if their class admits no tier (e.g., `secret_sensitive`).
- Results from untrusted tiers land `in_review` until a `local_only` reviewer approves.
- Leases are fenced with per-claim tokens; the reaper requeues and regenerates them.
- Dependencies are enforced; privacy monotonicity prevents laundering output downward.

This is not a generic queue — it is a bounded surface for federation with fail-closed
guarantees. See `docs/WORK_QUEUE.md` for the full design, MCP API, and threat model.

## Extending

- **New provider**: add to `config/providers.yaml` + a `secret_ref` mapping in
  `config/secrets.yaml`. No code change for routing.
- **New agent type**: add capabilities in `RouterService._capabilities_for` and a
  default profile in `config/profiles.yaml` → `agent_defaults`.
- **Real inference backend**: use `OpenAICompatibleBackend` (set
  `MODEL_MANAGER_BACKEND=openai` + `MODEL_BACKEND_URL`) or implement `ModelBackend`.
- **New rental vendor**: implement `NodeProvisioner` (see `RunPodProvisioner`) and
  register it in `provisioning.provisioner_for`. The vendor control-plane key is
  the only secret; it resolves through the secrets manager like a cloud key.
- **Real secrets manager**: the `op://` refs are the integration point; add a
  resolver kind in `common/secrets.py` or run the `op` CLI.
