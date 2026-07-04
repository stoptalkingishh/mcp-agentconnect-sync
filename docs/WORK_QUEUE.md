# Federated Work Queue

A **pull-based, open work surface** where separate agents and untrusted external compute can discover, claim, and complete work while maintaining AgentConnect's privacy and trust guarantees.

This is not a generic ticketing system. It is a **work-queue + lease semantics + authorization boundary** layered onto the existing SQLite store, exposed via MCP tools and an optional HTTP pull endpoint. The differentiator — the entire point — is **which participant may see and claim which work**, and **whether we trust the result they return**.

> **See it run:** `python examples/federation_demo.py` walks the whole loop offline — a friend's external box claims public work (landing `in_review` until a local reviewer approves), is refused a repo_sensitive ticket, and a secret_sensitive ticket is parked, unclaimable by anyone.

## The Design

### The Pull Model vs. Push

Traditional distributed systems **push** work: the control plane assigns work to known workers. Pull is the inverse: workers discover and claim work they are authorized for. This enables **federation**:

- Internal agents (trusted) can consume all work.
- Friend's compute (semi-trusted) can consume only lower-sensitivity work.
- Cloud workers (untrusted) can consume only public-safe or redacted work.
- Each participant never sees work outside their authorization.

The queue is the only coordination point; no direct connections between workers and routers are required.

### One Store, Atomicity

All queue state — tickets, leases, results, audit — lives in the **same SQLite `_conn`** as tasks, artifacts, and logs. A claim is a single guarded `UPDATE ... WHERE status='open'`:

```sql
UPDATE work_queue
   SET status='claimed', lease_holder=?, lease_token=?, ...
 WHERE ticket_id=? AND status='open'
   AND privacy_class IN (?, ?, ...)      -- authorization gate
   AND NOT EXISTS (...)                   -- dependency gate
```

SQLite serializes writers, so exactly one concurrent poller wins each ticket. Everything is transactional; no in-flight state.

## The Trust × Privacy Boundary

A **worker tier** is an attested `ProviderPrivacyTier`:

- `local_only` — your own box, fully trusted.
- `private_rented` — rented GPU running your own model, trusted.
- `external` — cloud API, untrusted (free tier).
- `external_paid` — cloud API, untrusted (paid tier).

A **ticket's privacy class** is one of:

- `public` — any tier may claim.
- `low_sensitive` — any tier, but the payload must already be redacted.
- `repo_sensitive` — `local_only` only.
- `secret_sensitive` — NO tier at all; never claimable.
- `restricted` — `local_only` only.

The authorization rule is the **sole gatekeeper**:

> A worker of tier **T** may claim a ticket of class **P** iff **T** is in `routing.yaml` `privacy.classes[P]`.

This is the identical mapping the router uses to decide which provider tiers may receive a class. It is **recomputed live on every claim** from the current `RoutingConfig`, never trusting a denormalized column — even if an attacker poisons `allowed_tiers`, the claim's `WHERE privacy_class IN (...)` filter re-gates against live config.

### Fail-Closed Guarantees

**Tier cannot self-elevate.** The tier used for authorization is ALWAYS the *attested* tier from the caller's authenticated identity (MCP config, HTTP certificate CN). A tier in the request body is ignored; a lower tier in the body is rejected.

**Unknown identity → rejected.** Identity not in the MCP `worker_id→tier` map (or HTTP client cert not recognized) → `403` / `ERROR: unknown_worker`.

**Empty tier set → claims nothing.** An unmapped tier (or tier not in any config class list) yields an empty admissible-class set → zero-length `IN (?)` clause → claims nothing.

**secret_sensitive is never claimable.** Not even by `local_only`. If `queue_add` is called with `secret_sensitive`, the ticket is stored `parked` with reason `secret_sensitive_not_pullable`, visible for audit but never offered.

**Unreferenced low_sensitive is parked.** If a `low_sensitive` ticket is enqueued without redaction (`cloud_safe=False`), it is `parked` with reason `redaction_required_not_cloud_safe` — mirrors the router's `cloud_safe` gate.

## Lease & Fencing Protocol

### The Claim

`queue_next(worker_id, capabilities, max=1)` atomically claims up to `max` authorized, unblocked tickets:

1. Resolve `worker_id` → attested tier (fail-closed if unknown).
2. Compute `admissible_classes` for the tier.
3. Scan `work_queue` ordered by priority/created_at, filtering by:
   - `status='open'`
   - `privacy_class IN (admissible_classes)`
   - `required_capabilities ⊆ worker_capabilities`
   - No unsatisfied dependencies (checked in the `WHERE` clause with a `NOT EXISTS` join)
4. For each candidate, attempt an atomic `UPDATE` that claims it. First writer wins; others continue scanning. Winner returns the ticket with:
   - `ticket_id` (opaque id, safe to hand to the worker)
   - `payload_ref` (artifact id of the redacted payload; the HTTP pull surface resolves it to the body inline via the lease-gated `payload_for`, so a remote worker gets the runnable text in one round trip)
   - `lease_token` (fresh UUID per claim, required for report/heartbeat)
   - `lease_expires_at` (Unix timestamp)
   - `attempts` (incremented at claim, never decremented)

**Internal ids are never handed out.** The `task_id` and submission are internal (the `task_id` is a live handle into the raw, un-redacted submission); only `ticket_id` and `payload_ref` cross to the worker. Dependency satisfaction is tested by parent `status='done'`, never by exposing parent content.

### Lease Renewal (Heartbeat)

`queue_update(worker_id, ticket_id, lease_token, extend_seconds=120)` extends the lease:

```sql
UPDATE work_queue SET lease_expires_at=?
 WHERE ticket_id=? AND status='claimed'
   AND lease_holder=? AND lease_token=? AND lease_expires_at>?
```

Only the token holder can renew. Token mismatches, expired leases, or wrong status → `ERROR: lease_lost`.

### The Reaper

An offline loop (not a background thread — determinism) calls `WorkQueue.reap_expired(now)` periodically (e.g., every 30s in tests, as often as the calling loop polls):

```sql
UPDATE work_queue SET status='open', lease_*=NULL WHERE status='claimed' AND lease_expires_at<? AND attempts<max_attempts
UPDATE work_queue SET status='parked', park_reason='max_attempts_exhausted' WHERE status='claimed' AND lease_expires_at<? AND attempts>=max_attempts
```

Tickets with attempts remaining are requeued; exhausted tickets are `parked`.

### Lease Fencing

Each claim mints a fresh `lease_token` (UUID). Report and heartbeat require an exact match:

```sql
WHERE lease_holder=? AND lease_token=? AND lease_expires_at>now
```

If a slow worker's lease expires, the reaper requeues the ticket. A second attempt by another worker issues a fresh token. The slow worker's stale report is refused even under the same `lease_holder` identity — **the token itself is the fence**. This defeats lease theft and resurrected stragglers.

## Verification Gate

On a valid `queue_report(worker_id, ticket_id, lease_token, result)`, the result is branched on the **attested reporter tier**:

**Trusted (`local_only`):** A *successful* result (`WorkerResult.status == 'completed'`) immediately becomes truth: `status='done'`, `result_status='approved'`, `completed_at=now`. If linked to a task, transition to `COMPLETE`. Record `evaluation(status='completed')`. A *failure* report (any other `WorkerResult.status`, e.g. `failed`/`error`) is never recorded as truth even on the trusted fast-path — it requeues (`status='open'`, lease dropped) while attempts remain, else fails terminally (`status='failed'`), and records `evaluation(status='failed')`.

**Untrusted (`private_rented`, `external`, `external_paid`):** Result lands **PENDING**. `status='in_review'`, `result_status='pending'`; linked task advanced only to `REVIEW_READY` (not `APPROVED`). Truth is withheld.

### Promotion

`queue_approve(reviewer_id, ticket_id)` (reviewer must be `local_only`):

- `in_review → done`, `result_status='approved'`, `completed_at=now`
- Linked task `REVIEW_READY → COMPLETE`
- Record `evaluation(status='completed')`

### Rejection

`queue_reject(reviewer_id, ticket_id, reason="")` (reviewer must be `local_only`):

- If `attempts < max_attempts`: requeue (`status='open'`, clear lease). The ticket is claimable again.
- If `attempts >= max_attempts`: `status='failed'`, ticket is terminal.
- Record `evaluation(status='failed')`
- Linked task `REJECTED → FAILED`

The guarantee: **a report from untrusted compute can NEVER silently become truth**. It must pass an explicit review gate to be accepted.

## Idempotency & Monotonicity

### Dedup

`queue_add` accepts an optional `dedup_key` for **idempotent re-adding**:

```sql
CREATE UNIQUE INDEX idx_wq_dedup ON work_queue(dedup_key) WHERE dedup_key IS NOT NULL
```

The partial-unique index (NULLs are distinct) ensures:
- Adding the same `dedup_key` twice returns the existing ticket unchanged.
- A `done` ticket is never reopened.

### Privacy Monotonicity

`queue_link(ticket_id, depends_on)` adds a dependency with a privacy check:

- Child's admissible-tier set ⊆ parent's admissible-tier set.
- A `public` child of a `repo_sensitive` parent is rejected (`ERROR: privacy_downgrade`).
- Prevents laundering sensitive output through a dependency edge to a lower class.

## Blocked Tickets (Derived State)

`status='open'` with unsatisfied dependencies is **derived** as `blocked` (never stored):

- Shown in `queue_status` for UX/audit.
- Not claimable (the `NOT EXISTS` gate in the claim's `WHERE` clause prevents it).
- Becomes claimable when all parents reach `done`.

## Rented/Private-Tier Agentic Carve-Out (SECURITY-CRITICAL)

### Agentic Execution Permission Model

Agentic execution (the act/tool loop where each step's tool observations feed back to the model) has a **strict permission model** because tool observations are sensitive context. They must never reach untrusted external models.

By default, **agentic runs only on owned-local resident models** — your hardware, your weights. Extending agentic to the **rented private tier** (a trusted, ephemeral, encrypted node running your weights with no external logging) is permitted when:

1. **The provider is local-owned OR a trusted rented private node:**
   - Owned-local: `cfg.type == "local"` and not a rented node.
   - Rented private: `RoutingEngine._is_rented(cfg)` returns true AND `cfg.trust` satisfies the `repo_sensitive` trust policy (§16): ephemeral spin-up, encrypted storage, your image, no external logging.

2. **The task explicitly opts in:** `submission.constraints.allow_rented == True`.

3. **Cloud is always rejected:** `external` and `external_paid` tiers are never permitted for agentic, regardless of classification or opt-in. Tool observations never leave to an untrusted external provider.

4. **Secret-sensitive never routes:** `secret_sensitive` is blocked before routing (§13) so never reaches the agentic guard.

### The Carve-Out Guard (service.py ~line 390)

The router's `submit_task` gate (before any spend) rejects agentic when permission is denied:

```python
if submission.constraints.execution == "agentic":
    is_owned_local = cfg.type == "local" and not RoutingEngine._is_rented(cfg)
    rented_ok = (
        RoutingEngine._is_rented(cfg)
        and submission.constraints.allow_rented
        and RoutingEngine._rented_trust_ok(cfg)
    )
    if not (is_owned_local or rented_ok):
        # REJECTED — before _confirm_charge, so no billing occurs
        task_state = TaskState.REJECTED
```

The rejection explains which condition failed (cloud selected; un-opted rented node; untrusted rented node) for operator visibility.

### Dispatching the Loop Through Rented (Acquire Once, Bill Once)

When routing selects a rented node for agentic, the loop must dispatch every step **through the rented provider's local client**, not through the gateway. This ensures the rental is billed **exactly once** at spin-up and reused across all steps:

1. **Acquire once (outside the loop):** The rented node is provisioned via `NodePool.acquire(cfg)`, billed for the min rental window, and pinned in memory.
2. **Generate through the rented client (per step):** Each agentic loop iteration calls `client.generate(step_request)` directly against the rented node — not `gateway.call`, which mishandles rented.
3. **Bill once:** Rental cost is recorded **once** on `acquire`, when `not reused`. Reuse of a warm node is free. Token usage is summed across all steps for the evaluation record only — the single window bill is the only money charged.
4. **Release (in finally):** The node is released for the reaper when the loop completes or fails, allowing the idle reaper to terminate and save cost.

**Rationale for the carve-out:** A rented private node runs your weights ephemerally with no external logging — that is exactly the rented tier's purpose. Tool observations staying within such a node is acceptable because they never reach an untrusted third party.

---

## Capability Matching (Filter, Not Authorization)

Capabilities are a **ticket-to-worker matching filter only**, never a security boundary (tiers × privacy is the security boundary). A worker declaring insufficient capabilities simply is not offered a ticket; a worker over-declaring just isn't offered work it would fail at.

### Capability Requirements

`queue_add` accepts an optional `required_capabilities` list — a JSON array of capability names (e.g. `["patch_generation", "coding"]`) that a worker must declare to claim the ticket. The router's `enqueue_task(..., required_capabilities=None)` populates defaults from the task's `agent_type` (if not explicitly set by the caller):

```python
# RouterService.enqueue_task — required_capabilities is the method's own kwarg.
caps = (
    list(required_capabilities)
    if required_capabilities is not None
    else list(self._capabilities_for(submission.agent_type))
)
queue.add(..., required_capabilities=caps)
```

(An explicit caller list still wins — the test is `is not None`, so an
explicitly passed empty list `[]` is respected as a wildcard and is NOT
overridden by the `agent_type` default.)

### Claim Filtering (WorkQueue.claim_next ~line 253)

When a worker claims with `queue_next(worker_id, capabilities)`, the scan includes a subset filter:

```sql
WHERE …
  AND required_capabilities ⊆ worker_capabilities  -- JSON containment check
  AND …
```

A ticket with `required_capabilities=["coding"]` can only be claimed by a worker declaring `capabilities=["coding", …]`. An empty `required_capabilities` is a wildcard — claimable by any worker.

### Observability: Open Capability Requirements

`WorkQueue.open_capability_requirements()` returns distinct capability sets among open (unclaimed/unblocked) tickets with their counts:

```python
[
  {"required_capabilities": ["coding", "patch_generation"], "open_tickets": 3},
  {"required_capabilities": ["review"], "open_tickets": 1},
  {"required_capabilities": [], "open_tickets": 5},
]
```

This lets an operator spot "no capable worker can ever claim this" situations. The capability set is exposed in `list_tickets` and `stats` (see Operator View below).

---

## Operator View (Broker-Side, Read-Only)

The verification gate leaves untrusted results `in_review` awaiting a `local_only` reviewer. The operator (a human or a machine) needs visibility into queue state without seeing sensitive payloads — hence a **payload-free projection** that shows metadata, provenance, and (new in this phase) capability requirements.

### WorkQueue Methods (Payload-Free)

Three new `WorkQueue` methods expose operator-facing views:

**`list_tickets(status=None, privacy_class=None, limit=100) -> list[dict]`**  
Filtered ticket listing. Returns rows with metadata only: `ticket_id`, `status` (with derived `blocked` state), `privacy_class`, `allowed_tiers`, `required_capabilities`, `priority`, `attempts`, `max_attempts`, `result_status`, `assignee`, `park_reason`, `origin`, `payload_ref` (artifact id; content not included), `result_ref` (artifact id), `provenance` (audit trail), `created_at`, `updated_at`. **Never includes `task_id` (an internal handle) or any payload/result content.**

**`pending_review(limit=100) -> list[dict]`**  
Shorthand for `list_tickets(status="in_review")`. Returns the backlog of tickets awaiting a `local_only` reviewer's approval.

**`stats() -> dict`**  
Aggregate counts: `{"by_status": {"open": 3, "claimed": 1, "in_review": 2, "done": 10, …}, "by_privacy_class": {"public": 8, "repo_sensitive": 5, …}, "capability_requirements": [...]}`  (the result of `open_capability_requirements()`). Lets the operator spot workload composition at a glance.

### MCP Operator Tools (Read-Only, No Identity)

Three new read-only MCP tools (in `router/mcp_server.py`) mirror the `WorkQueue` methods. They require **no worker identity** (they leak no payloads) but honor token auth if configured:

**`queue_list(status=None, privacy_class=None, limit=100) -> str`** (JSON)  
Calls `WorkQueue.list_tickets(...)`.

**`queue_pending(limit=100) -> str`** (JSON)  
Calls `WorkQueue.pending_review(...)`.

**`queue_stats() -> str`** (JSON)  
Calls `WorkQueue.stats()`.

The existing `queue_approve(reviewer_id, ticket_id)` and `queue_reject(reviewer_id, ticket_id, reason)` remain the *actions* — they still require a `local_only` reviewer identity and re-check it independently (not delegated to the read tools).

### Queue Operator Web Host (Optional, Token-Guarded)

A reference read-only web host (`agentconnect.router.queue_web`, fastAPI-based, mirrors `approval_web.py`'s pattern) renders the backlog and provides approve/reject buttons:

**Module:** `router/queue_web.py`

**Creating the app:**
```python
from agentconnect.router.queue_web import create_queue_operator_app

app = create_queue_operator_app(
    wq=my_workqueue,
    reviewer_id="ops-alice",           # identity used for approve/reject
    reviewer_tier="local_only",        # tier re-checked by approve/reject
    token="optional-bearer-token",     # guards /api/* (no token = public /)
)
```

**Running a standalone server:**
```python
from agentconnect.router.queue_web import start_queue_operator

start_queue_operator(
    wq=my_workqueue,
    reviewer_id="ops-alice",
    reviewer_tier="local_only",
    host="127.0.0.1",     # bind loopback by default (change for tunneling)
    port=8771,
    token="s3cret-token", # guards /api/*, optional
)
# Runs in a daemon thread; access http://127.0.0.1:8771/
```

**Features:**
- `GET /` — HTML page listing open and in-review tickets (metadata + provenance; no payload content).
- `GET /api/list`, `/api/pending`, `/api/stats` — JSON endpoints (same data).
- `POST /api/tickets/{tid}/approve` — calls `WorkQueue.approve(reviewer_id, reviewer_tier, tid)`.
- `POST /api/tickets/{tid}/reject` — calls `WorkQueue.reject(reviewer_id, reviewer_tier, tid, reason)`.
- Token auth: all `/api/*` endpoints require `Authorization: Bearer <token>` if token is set. `/` (the page) is public on loopback; set `token=` to enable token auth on the page as well.

**Security:** the app controls no authorization itself — it is a *surface*. Approve/reject calls independently re-check the reviewer's tier via `WorkQueue.approve`/`.reject` (which consult the live `RoutingConfig`). Token auth protects the surface; tier re-check is the gate.

**Installation & Extra:** included in the existing `agentconnect-router[web]` extra (which includes `fastapi>=0.110`, `uvicorn>=0.29`). It is **never mounted by default** — an operator starts it explicitly via `start_queue_operator(...)` or manually creates and serves the app.

---

## MCP Tools

Eleven MCP tools expose the queue (in `router/mcp_server.py`):

### Adding Work

**`queue_add(task, agent_type=None, privacy_class=None, required_capabilities=None, priority="normal", dedup_key=None, depends_on=None, refs=None) -> {ticket_id, status, privacy_class, allowed_tiers, park_reason}`**

Enqueue a ticket. The MCP tool classifies + redacts `task` itself (like `submit_task`) and computes `allowed_tiers`; `refs` are file-path hints for classification, and `agent_type` records the enqueuing origin. A claimer only ever receives the redacted payload, never the raw text. Returns a compact summary; never inline payload. (The underlying core `WorkQueue.add(...)` accepts a wider set of parameters — `origin`, `payload`, `payload_ref`, `task_id`, `max_attempts`, `assignee`, `cloud_safe` — but those are not exposed on the MCP tool.)

### Claiming & Holding

**`queue_next(worker_id, capabilities=None, max=1) -> {tickets: [...]}`**

Atomically claim up to `max` authorized tickets for a worker. Returns worker-visible rows (ticket_id, payload_ref, lease_token, lease_expires_at, attempts, status) — never the internal `task_id`. Empty list when nothing is authorized.

**`queue_claim(worker_id, ticket_id) -> claimed | {error}`**

Targeted claim of one ticket. Same atomic semantics as `queue_next`.

**`queue_update(worker_id, ticket_id, lease_token, extend_seconds=120) -> {status, lease_expires_at}`**

Heartbeat/renew the lease. Returns updated state or `{error: "lease_lost"}`.

### Reporting Results

**`queue_report(worker_id, ticket_id, lease_token, status="completed", summary="", confidence=0.0, changed_artifacts=None, risks=None) -> {ticket_status, result_status, result_ref}`**

Report a result under the fencing token. Idempotent: a second report or stale token is refused. Trusted tiers auto-accept (`done`); untrusted tiers land `in_review` (pending approval).

### Dependencies

**`queue_link(ticket_id, depends_on) -> {ok} | {error: "privacy_downgrade"}`**

Add a dependency edge with privacy monotonicity check.

### Review Gate

**`queue_approve(reviewer_id, ticket_id) -> {ticket_status, result_status}`**

Approve an in-review result (reviewer must be `local_only`). Transitions to `done`, updates linked task to `COMPLETE`.

**`queue_reject(reviewer_id, ticket_id, reason="") -> {ticket_status, result_status}`**

Reject an in-review result (reviewer must be `local_only`). Requeues if attempts remain, otherwise fails the ticket.

### Status & Audit

**`queue_status(ticket_id=None, status=None, limit=50) -> [{ticket summary}]`**

Query ticket state. Never returns payload content — only metadata, refs, and derived state (`blocked`). Shows full audit trail via `provenance` JSON (enqueue, claim, report, review events with timestamps and identities).

### Operator View (Read-Only, No Identity)

**`queue_list(status=None, privacy_class=None, limit=100) -> [{ticket metadata}]`**

List tickets (open, claimed, in_review, done, parked, failed) with optional filters. Returns payload-free rows: ticket_id, status, privacy_class, allowed_tiers, required_capabilities, priority, attempts, result_status, assignee, park_reason, origin, payload_ref (id only), result_ref (id only), provenance, created_at, updated_at. **Never includes task_id or payload content.** Operator tool — no worker identity required (it is read-only and leaks no payloads).

**`queue_pending(limit=100) -> [{in_review tickets}]`**

Shorthand for `queue_list(status="in_review")`. Returns the backlog of tickets awaiting a `local_only` reviewer's approval. Same payload-free projection.

**`queue_stats() -> {by_status, by_privacy_class, capability_requirements}`**

Aggregate queue state: counts by status and privacy_class, plus the distinct capability sets among open tickets. Lets an operator spot workload composition and unmatchable capability requirements.
Operator tool — no worker identity required.

## HTTP Pull Endpoint (Optional)

An additive HTTP mount (`add_pull_routes` in `runtime/transport.py`) exposes the same claim/report flow over HTTPS with mutual TLS:

- Identity is the peer's TLS client certificate. Because identity IS authorization on this surface (it alone gates which privacy_class a caller may claim and whether its report auto-accepts), the anchor must be unspoofable: identity is read from the ASGI-TLS extension (the terminating server's view of the cert). The client-settable `X-Client-Cert-DN` / `X-SPIFFE-ID` header is trusted **only** when the operator passes `add_pull_routes(..., trust_proxy_headers=True)`, asserting a header-stripping mTLS-terminating reverse proxy is in front. By default (`trust_proxy_headers=False`) a header-only identity is treated as no identity → `403`, so a peer connecting directly over plain HTTP cannot spoof a trusted tier.
- Attested tier is resolved by a pluggable `tier_resolver: (identity) -> ProviderPrivacyTier | None`.
- `GET /queue/next?capabilities=...` — claim next ticket. Each claimed ticket carries its redacted task body **inline** as `payload`, so a remote worker (which has no access to the broker's artifact store) gets everything it needs to run in one round trip.
- `GET /queue/{ticket_id}/payload?lease_token=...` — re-fetch the redacted body (e.g. after a worker restart that kept the lease).
- `POST /queue/{ticket_id}/report` — body = `WorkerResult` + `lease_token`.
- `POST /queue/{ticket_id}/heartbeat` — renew lease.

**Authorized payload delivery.** The core `WorkQueue.payload_for(identity, ticket_id, lease_token, tier)` is the only seam that returns a task body, and it is doubly gated: the caller must hold the *live lease* (matching `lease_token`, not expired) **and** its attested tier must still `may_claim` the ticket's class (a belt-and-suspenders re-check so a post-claim tier downgrade is honored). The internal `task_id`/submission never cross — only the redacted `work_payload`, which was already made cloud-safe for that tier at enqueue.

If `queue=None`, routes are not mounted — the runtime worker app is unchanged. The pull surface is **trimmable** from the core without touching behavior.

Responses degrade to typed JSON (`{detail: "lease_lost"}`, `{error: "unknown_ticket"}`), never raw 500s.

## Worker-Side Pull Loop (`PullWorker`)

`runtime/pull_worker.py` is the client half of the federation — the program a compute contributor (a friend's box, a separate agent fleet) actually runs. The broker brokers tickets but **never executes**; the compute happens on the contributor's machine, with its own `AgentRuntime`.

```python
from agentconnect.runtime import PullWorker, LangGraphAgentRuntime

worker = PullWorker(
    LangGraphAgentRuntime(my_local_model_source),   # runs on MY box
    base_url="https://broker.example",
    tls=my_mutual_tls_config,                        # cert = identity = tier
    capabilities=["coding", "summarization"],
)
worker.run_forever()   # claim -> run locally -> report, backing off when idle
```

Each iteration: `claim` a ticket the worker's attested tier is authorized for (body delivered inline) → `execute` it through the local runtime → `report` the `WorkerResult` back under the lease. A trusted (`local_only`) worker's completed result auto-accepts to `done`; an untrusted contributor's result lands `in_review` until a `local_only` reviewer approves it. Identity is the mTLS client certificate the worker presents; `identity_headers` is only for the header-stripping-proxy / test seam.

**Heartbeating.** Crash safety is the broker's: a worker that dies mid-task loses its lease and the reaper requeues the ticket. To keep a *live* long task from being reaped out from under it, set `heartbeat_interval > 0` — a background thread renews the lease while the task runs (heartbeat failures are swallowed; `report` is the authoritative fence and returns `lease_lost` if the lease was already lost, refusing to corrupt a re-claim).

### Running one: `agentconnect-worker`

`runtime/worker_cli.py` (console script `agentconnect-worker`, or `python -m agentconnect.runtime.worker_cli`) makes a contributor launchable with no code. Install the `[worker]` extra for a real model, or use `--dry-run` (a built-in echo model) to smoke-test connectivity + auth + the claim/report flow first:

```bash
# Local dev / connectivity check (plain HTTP loopback, header identity):
agentconnect-worker --broker http://127.0.0.1:8000 --dry-run \
    --insecure-localhost --identity trusted-worker --once

# A friend's box contributing real compute over mutual TLS:
agentconnect-worker --broker https://broker:8443 \
    --capabilities coding,summarization \
    --ca ca.pem --cert worker.pem --key worker.key \
    --heartbeat-interval 30
```

In production, identity is the mTLS client certificate (`--ca/--cert/--key`) — no header is sent and the broker reads the cert from the TLS layer. `--identity` sends the `X-Client-Cert-DN` header instead, for the header-stripping-proxy topology (broker started with `trust_proxy_headers=True`) or local `--insecure-localhost` dev where there is no cert to carry identity. The model is an OpenAI-compatible server via the model-manager's `backend_from_env` (`MODEL_BACKEND_URL`, …). `--once` processes a single ticket; otherwise it loops, backing off `--poll-interval` seconds when idle.

## Data Model

All state in `SharedMemory._conn`:

### `work_queue` Table

| Column | Type | Notes |
|--------|------|-------|
| `ticket_id` | TEXT PK | `_new_id("wq")` |
| `dedup_key` | TEXT | Idempotency; partial-unique index, NULLs allowed |
| `origin` | TEXT | Enqueuing identity (audit) |
| `task_id` | TEXT | FK by convention to `tasks`; internal, never handed out |
| `payload_ref` | TEXT | Artifact id of redacted worker-visible payload |
| `privacy_class` | TEXT NOT NULL | Public, low_sensitive, repo_sensitive, secret_sensitive, restricted |
| `allowed_tiers` | TEXT | JSON list, denormalized cache (for UX/indexing; **never trusted for claim decision**) |
| `required_capabilities` | TEXT | JSON list (filter, not a security gate) |
| `priority` | TEXT | urgent, normal, low |
| `status` | TEXT NOT NULL | open, claimed, in_review, done, parked, failed; blocked is derived |
| `assignee` | TEXT | Advisory suggested worker/tier from router (never enforced) |
| `lease_holder` | TEXT | Attested identity currently holding |
| `lease_tier` | TEXT | Attested tier captured at claim (audit) |
| `lease_token` | TEXT | Fresh UUID per claim (fencing token) |
| `lease_expires_at` | REAL | Unix timestamp |
| `attempts` | INTEGER | Incremented at claim |
| `max_attempts` | INTEGER | Default 3 |
| `result_ref` | TEXT | Artifact id of stored `WorkerResult` JSON |
| `result_status` | TEXT | pending, approved, rejected |
| `park_reason` | TEXT | Why the ticket is parked (secret_sensitive_not_pullable, max_attempts_exhausted, etc.) |
| `provenance` | TEXT | JSON audit trail: [{event, identity?, tier?, ts}, ...] |
| `created_at` | REAL | Enqueue time |
| `updated_at` | REAL | Last transition |
| `claimed_at` | REAL | First claim time |
| `completed_at` | REAL | Approval/completion time |

### `work_queue_deps` Table (Edges)

| Column | Type | Notes |
|--------|------|-------|
| `ticket_id` | TEXT | Child (depends on parent) |
| `depends_on` | TEXT | Parent (must be done before child is claimable) |
| **PK** | (ticket_id, depends_on) | |

Indexes: `idx_wq_status_priority` (claim scan), `idx_wq_lease` (reaper), `idx_wqdeps_ticket` (blocked-by query).

## Router Tie-In (S2)

The router's `enqueue_task(submission)` method (in `service.py`):

1. Classifies the task via `privacy_mod.classify(submission)` → `PrivacyClass`
2. Redacts the payload via `privacy_mod.redact(submission)` → artifact id
3. Calls `WorkQueue.add(...)` with the class, redacted payload, and routing decision's `selected_provider` tier as the advisory `assignee`
4. Records a `RoutingDecision` in memory

This is a **thin tie-in**: pull workers self-select within their authorization; `assignee` is a hint, never enforced. `queue_add` and `submit_task` remain independent.

## Examples

### Scenario 1: Public Work, Any Tier Claims

```python
# Router enqueues a public task:
queue.add(
    task="summarize article",
    privacy_class=PrivacyClass.public,
    payload_ref=summary_artifact_id,
)

# Any worker tier can claim:
for tier in [local_only, private_rented, external, external_paid]:
    ticket = queue.claim_next(identity, tier)
    assert ticket is not None
```

### Scenario 2: Repo-Sensitive Work, Only Local Claims

```python
# Router enqueues a repo-sensitive task:
queue.add(
    task="code review private repo",
    privacy_class=PrivacyClass.repo_sensitive,
    payload_ref=redacted_diff_artifact,
)

# Only local_only claims:
ticket = queue.claim_next("local-gpu", local_only)
assert ticket is not None

# External worker is denied:
tickets = queue.claim_next("friend-compute", external)
assert tickets == []
```

### Scenario 3: Untrusted Result Requires Approval

```python
# External worker claims and completes a task:
ticket = queue.claim_next("cloud-worker", external)
result = WorkerResult(status="completed", confidence=0.95, output="result text")
queue.report("cloud-worker", ticket["ticket_id"], ticket["lease_token"], result)

# Ticket is in_review, task is REVIEW_READY (not COMPLETE):
status = queue.status(ticket["ticket_id"])
assert status["status"] == "in_review"
assert status["result_status"] == "pending"

# Local reviewer must approve:
queue.approve("local-reviewer", ticket["ticket_id"])

# Now status is done:
status = queue.status(ticket["ticket_id"])
assert status["status"] == "done"
```

### Scenario 4: Dependency + Privacy Monotonicity

```python
# Parent: repo_sensitive
parent = queue.add(
    task="code audit",
    privacy_class=PrivacyClass.repo_sensitive,
)

# Child: public (less restrictive)
child = queue.add(
    task="post summary",
    privacy_class=PrivacyClass.public,
)

# Linking fails: child is less restrictive than parent
result = queue.link(child["ticket_id"], parent["ticket_id"])
assert result["error"] == "privacy_downgrade"

# Reverse link is allowed (child more restrictive):
result = queue.link(parent["ticket_id"], child["ticket_id"])
assert result["ok"] is True
```

### Scenario 5: Lease Fencing After Reaper

```python
# Worker claims and lets lease expire:
ticket = queue.claim_next("slow-worker", local_only)
token1 = ticket["lease_token"]

# Reaper runs:
queue.reap_expired(now + 200)  # lease_expires_at + 80s

# Ticket is requeued:
status = queue.status(ticket["ticket_id"])
assert status["status"] == "open"

# New worker claims and gets fresh token:
ticket2 = queue.claim_next("new-worker", local_only)
token2 = ticket2["lease_token"]
assert token1 != token2

# Old worker's report is refused:
result = queue.report("slow-worker", ticket["ticket_id"], token1, result)
assert result["error"] == "lease_lost"

# New worker's report succeeds:
result = queue.report("new-worker", ticket["ticket_id"], token2, result)
assert result["ticket_status"] == "done"
```

## Security & Testing

Every invariant is tested offline:

- Authorization denial (external worker denied repo_sensitive, secret_sensitive denied all).
- Tier cannot self-elevate.
- Concurrent claim race — exactly one winner.
- Lease fencing after reaper requeue.
- Idempotency (dedup_key, already-reported errors).
- Privacy monotonicity (child ⊆ parent).
- Untrusted report lands in_review.

See `tests/test_workqueue.py`, `tests/test_workqueue_verify.py`, and `tests/test_workqueue_mcp.py` for the full test suite (all offline, in-memory SQLite, no network).

## Implementation Notes

- **No new server, no Postgres/Redis.** Everything is in the existing SQLite store with one transactional connection (`SharedMemory._conn`, `check_same_thread=False`).
- **Framework-free core.** `WorkQueue` is pure Python + pydantic (in `agentconnect-core/common/workqueue.py`); no FastAPI/MCP SDK needed for the core logic.
- **Router is a thin wrapper.** `RouterService.enqueue_task` and `RouterService.reap_work_queue` (mirroring `reap_idle_nodes`) are the only tie-ins.
- **HTTP pull is additive & trimmable.** `add_pull_routes(app, queue, tier_resolver)` in `runtime/transport.py` mounts routes if `queue is not None`. No queue → no routes.
- **Deterministic offline testing.** All tests use in-memory `SharedMemory`, scripted tier resolvers, and `TestClient` for HTTP (no real network/TLS).

---

**See Also:**
- `packages/agentconnect-core/src/agentconnect/common/workqueue.py` — implementation & module docstring (authorization invariants).
- `packages/agentconnect-router/src/agentconnect/router/mcp_server.py` — MCP tool definitions.
- `packages/agentconnect-runtime/src/agentconnect/runtime/transport.py` — HTTP routes.
- `config/routing.yaml` `privacy.classes` — the authorization source-of-truth.
- `tests/test_workqueue.py` — core authorization & concurrency tests.
- `tests/test_workqueue_verify.py` — verification gate tests.
- `tests/test_workqueue_mcp.py` — MCP tool round-trips.
