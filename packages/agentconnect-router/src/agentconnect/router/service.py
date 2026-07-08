"""Agent Router service — the global control plane (handoff §4.1, §11).

Implements the deterministic task-routing flow and backs the MCP tools. Given a
task submission it: classifies, runs privacy + redaction, computes eligible
providers, scores + selects, reserves quota / admits locally, dispatches through
the gateway, writes full output to shared memory, and returns a COMPACT summary
plus artifact references (context virtualization, §9).

Everything large lives in shared memory and is read back in bounded chunks; the
manager agent receives only summaries + refs.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from . import guard_hook
from ..common import privacy as privacy_mod
from ..common.authorization import ChargeRequest, DenyingSpendAuthorizer, SpendAuthorizer
from ..common.budget import BudgetManager
from ..common.evaluation import Evaluator
from ..common.config import (
    ProfilesConfig,
    RemoteWorkerConfig,
    RoutingConfig,
    load_all,
    load_remote_workers,
)
from ..common.memory import SharedMemory
from ..common.privacy import ClassificationHints
from ..common.providers import ProviderRegistry
from ..common.quota import QuotaLedger
from ..common.schemas import (
    GenerateRequest,
    ManagerStatus,
    Priority,
    PrivacyClass,
    RoutingDecision,
    TaskConstraints,
    TaskState,
    TaskSubmission,
    TaskSummary,
    WorkerResult,
)
from ..common.state import TERMINAL_STATES, assert_transition
from ..common.tokens import estimate_io_tokens
from ..common.workqueue import WorkQueue
from .gateway import GatewayResult, ProviderGateway
from .local_client import LocalClient
from .routing import RoutingContext, RoutingEngine

if TYPE_CHECKING:
    from .provisioning import NodePool, NodeProvisioner


@dataclass
class RouterService:
    memory: SharedMemory
    registry: ProviderRegistry
    profiles: ProfilesConfig
    routing_cfg: RoutingConfig
    quota: QuotaLedger
    engine: RoutingEngine
    gateway: ProviderGateway
    local_client: Optional[LocalClient] = None
    # Rented-node lifecycle (Goal 4). Defaults keep everything offline/testable.
    provisioner: Optional["NodeProvisioner"] = None
    rented_client_factory: Optional[Callable[[Any, Any], LocalClient]] = None
    # Evaluation & learning (Phase 6).
    evaluator: Optional[Evaluator] = None
    # Warm-node reuse for rented GPUs (Goal 4 amortization).
    node_pool: Optional["NodePool"] = None
    # Global spend budget (mandatory, no silent default).
    budget: Optional[BudgetManager] = None
    # Direct-to-user spend authorization (deterministic human gate). Default denies,
    # so paid/rented spend is disabled until a real user-facing authorizer is wired.
    authorizer: Optional[SpendAuthorizer] = None
    # Federated pull work-queue (S1 core), sharing this same memory._conn. Optional
    # only for callers that construct a bare RouterService by hand in tests.
    workqueue: Optional[WorkQueue] = None
    # Router-driven remote-worker dispatch: agentic tasks may be PUSHED whole to a
    # registered remote worker over mTLS instead of running the loop in-process.
    # Empty registry -> feature off (every agentic task runs in-process). The
    # factory maps a RemoteWorkerConfig to an AgentRuntime (default builds
    # HttpAgentRuntime); it is also the injection seam tests use to supply a
    # TestClient-backed runtime with no real network.
    remote_workers: list[RemoteWorkerConfig] = field(default_factory=list)
    remote_runtime_factory: Optional[Callable[[RemoteWorkerConfig], Any]] = None
    # In-process runtime injection: swap the local agentic runtime for your own
    # AgentRuntime (e.g. a wrapper around an existing LangGraph/CrewAI graph) without
    # editing the router. Signature (ModelSource, RuntimeConfig) -> AgentRuntime.
    # None -> lazily build the built-in LangGraphAgentRuntime (so one-shot-only
    # deployments never import the runtime / its langgraph dependency). See
    # _make_local_runtime and docs/AGENT_RUNTIME.md ("Bring your own runtime").
    local_runtime_factory: Optional[Callable[[Any, Any], Any]] = None
    # Hierarchical delegation (Track 4): default OFF, so an agentic run stays a single
    # worker unless explicitly enabled. When on, an agentic worker may emit sub-tasks
    # via the `delegate` action; the router runs each as a child agentic sub-run at the
    # next depth (privacy_class clamped child ⊆ parent — never a downgrade), then folds
    # the child summaries back into one parent summary (recursive context
    # virtualization). Bounded by depth + per-node fan-out so recursion can't run away.
    enable_delegation: bool = False
    max_delegation_depth: int = 2
    max_subtasks: int = 8

    # ------------------------------------------------------------- factory
    @classmethod
    def create(
        cls,
        memory: Optional[SharedMemory] = None,
        local_client: Optional[LocalClient] = None,
        gateway: Optional[ProviderGateway] = None,
        provisioner: Optional["NodeProvisioner"] = None,
        rented_client_factory: Optional[Callable[[Any, Any], LocalClient]] = None,
        authorizer: Optional[SpendAuthorizer] = None,
        local_runtime_factory: Optional[Callable[[Any, Any], Any]] = None,
    ) -> "RouterService":
        providers_cfg, profiles, routing_cfg = load_all()
        mem = memory or SharedMemory()
        registry = ProviderRegistry.from_config(providers_cfg)
        quota = QuotaLedger(memory=mem)
        engine = RoutingEngine(registry, profiles, routing_cfg, quota)
        gw = gateway or ProviderGateway(local_client=local_client)
        if local_client is not None:
            gw.bind_local(local_client)
        from .provisioning import NodePool, StubProvisioner

        min_samples = int(routing_cfg.scoring.get("learned_min_samples", 5))

        def _default_remote_runtime(w: RemoteWorkerConfig):
            # Lazy import behind the [remote] extra: only reached when a remote
            # worker is actually registered and selected.
            from agentconnect.runtime import HttpAgentRuntime

            return HttpAgentRuntime(w.endpoint, tls=w.tls)

        return cls(
            memory=mem, registry=registry, profiles=profiles, routing_cfg=routing_cfg,
            quota=quota, engine=engine, gateway=gw, local_client=local_client,
            provisioner=provisioner or StubProvisioner(),
            rented_client_factory=rented_client_factory,
            evaluator=Evaluator(mem, min_samples=min_samples),
            workqueue=WorkQueue(mem, routing_cfg),
            node_pool=NodePool(),
            budget=BudgetManager(mem, routing_cfg),
            authorizer=authorizer or DenyingSpendAuthorizer(),
            remote_workers=load_remote_workers(),
            remote_runtime_factory=_default_remote_runtime,
            # None stays None here on purpose: _make_local_runtime lazily builds the
            # built-in runtime so a one-shot-only deployment need not install it.
            local_runtime_factory=local_runtime_factory,
        )

    def _make_local_runtime(self, source, config):
        """Build the in-process agentic runtime. Uses an injected
        ``local_runtime_factory`` when set (bring-your-own AgentRuntime), else lazily
        constructs the built-in ``LangGraphAgentRuntime`` — the lazy import keeps
        one-shot-only deployments free of the runtime/langgraph dependency."""
        if self.local_runtime_factory is not None:
            return self.local_runtime_factory(source, config)
        from agentconnect.runtime import LangGraphAgentRuntime

        return LangGraphAgentRuntime(source, config)

    # ----------------------------------------------------------- evaluation
    def _record_eval(self, cfg, model, task_id, agent_type, status, latency_ms,
                     in_tok, out_tok, cost, confidence=None) -> None:
        # ``cfg`` is a ProviderConfig for routed dispatch, or a plain provider-label
        # string for router-driven remote dispatch (no provider object exists — the
        # remote worker ran its own model).
        provider = cfg if isinstance(cfg, str) else cfg.provider_id
        self.memory.record_evaluation(
            {
                "provider": provider, "model": model, "task_id": task_id,
                "agent_type": agent_type, "status": status, "latency_ms": latency_ms,
                "input_tokens": in_tok, "output_tokens": out_tok, "cost_usd": cost,
                "confidence": confidence, "retries": 0,
            }
        )

    def get_provider_scorecards(self) -> list[dict[str, Any]]:
        """Per-provider learned scorecards + current learned-quality signal (Phase 6)."""
        if self.evaluator is None:
            return []
        cards = self.evaluator.scorecards()
        learned = self.evaluator.learned_quality()
        out = []
        for pid, sc in cards.items():
            out.append(
                {
                    "provider": pid, "samples": sc.samples,
                    "success_rate": round(sc.success_rate, 4),
                    "avg_latency_ms": round(sc.avg_latency_ms, 2),
                    "avg_cost_usd": round(sc.avg_cost_usd, 6),
                    "avg_confidence": sc.avg_confidence,
                    "learned_quality_signal": round(learned.get(pid, 0.0), 4),
                }
            )
        return sorted(out, key=lambda r: r["provider"])

    # ------------------------------------------------------------- dispatch
    def _dispatch(self, cfg, gen_req: GenerateRequest) -> GatewayResult:
        """Run a generation. Owned-local/cloud go through the gateway; a rented
        node is provisioned on demand, used over mTLS, billed for its rental
        window, then torn down (handoff Goal 4)."""
        if not RoutingEngine._is_rented(cfg):
            return self.gateway.call(cfg, gen_req)

        from .provisioning import NodePool, spec_from_provider

        pool = self.node_pool or NodePool()
        spec = spec_from_provider(cfg, model_id=gen_req.model_id)
        handle, reused = pool.acquire(cfg, self.provisioner, spec)
        factory = self.rented_client_factory or self._default_rented_client
        client = factory(cfg, handle)
        resp = client.generate(gen_req)
        # Bill the rental window only on first spin-up; reuse within the warm
        # window is free (amortization). A production reaper trues up on teardown.
        if not reused:
            window = cfg.rental.min_rental_seconds if cfg.rental else 0
            self.quota.record_rental_window(cfg, gen_req.task_id, seconds=window)
        pool.release(cfg)
        self.memory.append_log(
            gen_req.task_id,
            f"rented_node={handle.node_id} endpoint={handle.manager_endpoint} reused={reused}",
        )
        return GatewayResult(
            output_text=resp.output_text,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            provider=cfg.provider_id,
            model=resp.model_id,
        )

    def reap_idle_nodes(self, now: float) -> list[str]:
        """Terminate rented nodes idle past their window. Call periodically."""
        if self.node_pool is None:
            return []
        cfgs = {p.provider_id: p for p in self.registry.all()}
        return self.node_pool.reap_idle(self.provisioner, cfgs, now)

    def reap_work_queue(self, now: float) -> dict[str, list[str]]:
        """Requeue expired leases / park exhausted ones. Mirrors
        ``reap_idle_nodes``: call periodically from an explicit loop, never a
        background thread (keeps the offline gate deterministic)."""
        if self.workqueue is None:
            return {"requeued": [], "parked": []}
        return self.workqueue.reap_expired(now)

    # ------------------------------------------------- router-as-assigner tie-in
    def enqueue_task(
        self,
        submission: TaskSubmission,
        *,
        dedup_key: Optional[str] = None,
        required_capabilities: Optional[list[str]] = None,
        priority: str = "normal",
        depends_on: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Classify/redact a submission and place it on the pull work-queue,
        THIN tie-in between the router and the federated queue.

        This mirrors ``submit_task``'s classify+redact steps only — it never
        dispatches. A routing decision is computed and recorded purely to leave
        an ADVISORY ``assignee`` hint on the ticket (a suggested provider/tier);
        pull workers self-select within their own authorization regardless, so a
        wrong or absent hint never grants or denies a claim. ``submit_task`` and
        ``queue_add`` remain independent surfaces onto the same store.
        """
        if self.workqueue is None:
            raise RuntimeError("RouterService has no workqueue configured")

        sub_dict = submission.model_dump(mode="json")
        task_id = self.memory.create_task(sub_dict, agent_type=submission.agent_type)
        self.memory.append_log(task_id, f"enqueue_task agent_type={submission.agent_type}")

        hints = ClassificationHints(
            file_paths=tuple(submission.refs), declared=submission.constraints.privacy_class
        )
        privacy_class = privacy_mod.classify(submission.task, hints)
        redaction, redacted_text = privacy_mod.redact(submission.task, privacy_class)
        payload_ref = self.memory.put_artifact(task_id, "sanitized_payload", redacted_text)
        redaction.payload_ref = payload_ref
        self.memory.update_task(task_id, privacy_class=privacy_class.value)

        # Advisory only: a best-effort routing pass to suggest an assignee. Never
        # blocks enqueueing — if no provider is eligible/live, the ticket is still
        # queued with assignee=None and waits for a pull worker to self-select.
        assignee: Optional[str] = None
        try:
            ctx = RoutingContext(
                task_id=task_id,
                privacy_class=privacy_class,
                needed_capabilities=self._capabilities_for(submission.agent_type),
                profile=submission.constraints.profile,
                allow_external=submission.constraints.allow_external,
                allow_paid=submission.constraints.allow_paid,
                allow_rented=submission.constraints.allow_rented,
                priority=submission.constraints.priority,
                quality="high" if submission.constraints.quality == "high" else "standard",
                cloud_safe=redaction.cloud_safe,
            )
            status = self._local_status()
            decision = self.engine.route(ctx, status)
            self.memory.record_routing_decision(task_id, decision.model_dump(mode="json"))
            assignee = decision.selected_provider
        except Exception as exc:  # advisory hint only; never fails the enqueue
            self.memory.append_log(task_id, f"enqueue_task advisory routing failed: {exc}", level="warn")

        # Capabilities are a MATCHING FILTER only, never authorization: an
        # explicit caller-supplied list wins; otherwise default from the
        # agent_type mapping so claim_next's existing subset filter actually
        # binds (an empty list matches everything and was effectively
        # vacuous). A worker under-declaring simply isn't offered the ticket;
        # a worker over-declaring gets work it may fail, which the
        # verification gate (in_review) catches. The tier x privacy boundary
        # above is the only authorization gate and is unchanged by this.
        caps = (
            list(required_capabilities)
            if required_capabilities is not None
            else list(self._capabilities_for(submission.agent_type))
        )

        return self.workqueue.add(
            task=submission.task,
            origin=f"router:{submission.agent_type or 'unknown'}",
            privacy_class=privacy_class,
            payload_ref=payload_ref,
            task_id=task_id,
            required_capabilities=caps,
            priority=priority,
            dedup_key=dedup_key,
            depends_on=depends_on,
            assignee=assignee,
            cloud_safe=redaction.cloud_safe,
        )

    def _default_rented_client(self, cfg, handle) -> LocalClient:
        from .local_client import HttpLocalClient

        if not handle.manager_endpoint:
            raise RuntimeError(f"Rented node {handle.node_id} has no endpoint after provisioning.")
        return HttpLocalClient(handle.manager_endpoint, tls=cfg.tls)

    # --------------------------------------------------------------- helpers
    def _local_status(self) -> Optional[ManagerStatus]:
        if self.local_client is None:
            return None
        try:
            return self.local_client.status()
        except Exception:
            self.registry.set_health("local_r9700", "down")
            return None

    def _transition(self, task_id: str, current: TaskState, dst: TaskState) -> TaskState:
        assert_transition(current, dst)
        self.memory.update_task(task_id, state=dst.value)
        return dst

    def _clamp(self, text: str) -> str:
        hard = int(self.routing_cfg.mcp_output_policy.get("hard_max_chars", 12000))
        return text if len(text) <= hard else text[:hard]

    # --------------------------------------------------------- MCP: submit_task
    def submit_task(self, submission: TaskSubmission) -> TaskSummary:
        # 1-2. Receive + assign id.
        sub_dict = submission.model_dump(mode="json")
        task_id = self.memory.create_task(sub_dict, agent_type=submission.agent_type)
        state = TaskState.CREATED
        self.memory.append_log(task_id, f"submitted agent_type={submission.agent_type}")

        # 3. Classify task type -> resolve a capability profile.
        profile = submission.constraints.profile or self.profiles.agent_defaults.get(
            submission.agent_type or "", "resident_ok"
        )
        state = self._transition(task_id, state, TaskState.CLASSIFIED)

        # 4. Privacy class + redaction pass (§13, §14).
        hints = ClassificationHints(
            file_paths=tuple(submission.refs),
            declared=submission.constraints.privacy_class,
        )
        privacy_class = privacy_mod.classify(submission.task, hints)
        redaction, redacted_text = privacy_mod.redact(submission.task, privacy_class)
        payload_ref = self.memory.put_artifact(task_id, "sanitized_payload", redacted_text)
        redaction.payload_ref = payload_ref
        self.memory.append_log(
            task_id,
            f"privacy_class={privacy_class.value} cloud_safe={redaction.cloud_safe} "
            f"redactions={len(redaction.redactions)} lossiness={redaction.lossiness}",
        )
        self.memory.update_task(task_id, privacy_class=privacy_class.value)
        state = self._transition(task_id, state, TaskState.PRIVACY_CHECKED)

        # secret_sensitive is blocked from any LLM (§13). Fail closed.
        if privacy_class == PrivacyClass.secret_sensitive:
            self._transition(task_id, state, TaskState.REJECTED)
            self.memory.update_task(
                task_id,
                state=TaskState.REJECTED.value,
                summary="Blocked: task contains secret-sensitive content and may not be sent to any model.",
                recommended_next_action="Remove/redact secrets or handle out-of-band; do not route to an LLM.",
            )
            return self._summary(task_id)

        # 4b. Optional fascia-guard pass — adds prompt-injection detection and an
        # independent secret/PII scan on top of privacy_mod (defense in depth).
        # Dormant unless FASCIA_GUARD[_ENFORCE] is set; only ENFORCE can reject.
        guard_verdict = guard_hook.scan_task(submission.task, task_id)
        if guard_verdict is not None:
            self.memory.append_log(task_id, guard_hook.describe(guard_verdict, "task"))
            if guard_hook.enforcing() and guard_hook.is_block(guard_verdict):
                self._transition(task_id, state, TaskState.REJECTED)
                self.memory.update_task(
                    task_id,
                    state=TaskState.REJECTED.value,
                    summary="Blocked by fascia-guard: task contains secret/credential material.",
                    recommended_next_action="Remove/redact the flagged content; do not route to an LLM.",
                )
                return self._summary(task_id)

        # Router-driven remote-worker dispatch: an agentic task may be PUSHED whole
        # to a registered remote worker whose ATTESTED tier is trusted for this
        # privacy class (the same fail-closed WorkQueue.may_claim predicate the pull
        # federation uses) and that reports capacity. Placed BEFORE provider routing
        # because the worker runs its OWN model — the wire carries no model pin, so no
        # provider/spend/quota decision applies. No eligible/available worker -> fall
        # through to in-process routing (today's path). secret_sensitive is already
        # blocked above, and may_claim([], ...) would deny anyway.
        if submission.constraints.execution == "agentic":
            selected = self._select_remote_worker(privacy_class)
            if selected is not None:
                worker_cfg, runtime = selected
                return self._run_agentic_remote(
                    worker_cfg, runtime, submission, task_id, state, privacy_class
                )

        # 5-6. Quality requirement + token estimate.
        max_out = submission.constraints.max_output_tokens or int(
            self.routing_cfg.local_inference_defaults.get("default_max_output_tokens", 1200)
        )
        in_tok, out_tok = estimate_io_tokens(submission.task, max_out)

        # 7. Fetch local status. 8-12. Compute eligibility + score + select.
        status = self._local_status()
        ctx = RoutingContext(
            task_id=task_id,
            privacy_class=privacy_class,
            needed_capabilities=self._capabilities_for(submission.agent_type),
            profile=profile,
            require_exact_model=submission.constraints.require_exact_model,
            est_input_tokens=in_tok,
            est_output_tokens=out_tok,
            allow_external=submission.constraints.allow_external,
            allow_paid=submission.constraints.allow_paid,
            allow_rented=submission.constraints.allow_rented,
            priority=submission.constraints.priority,
            quality="high" if submission.constraints.quality == "high" else "standard",
            cloud_safe=redaction.cloud_safe,
        )
        state = self._transition(task_id, state, TaskState.ELIGIBLE_PROVIDERS_COMPUTED)
        # Refresh the learned-quality prior from observed outcomes (Phase 6).
        if self.evaluator is not None:
            self.engine.set_learned_quality(self.evaluator.learned_quality())
        decision = self._route_with_budget_prompt(ctx, status)
        self.memory.record_routing_decision(task_id, decision.model_dump(mode="json"))
        self.memory.append_log(
            task_id, f"routing_decision={decision.decision} provider={decision.selected_provider} "
            f"model={decision.selected_model}"
        )

        if decision.selected_provider is None:
            self._transition(task_id, state, TaskState.REJECTED)
            reasons = {r.reason for r in decision.rejected_options}
            if "budget_not_configured" in reasons:
                # The user was prompted directly (via the authorizer) and no budget was
                # set — deterministic, not left to the stochastic agent.
                summary = "No eligible provider: paid/rented need a spend budget, and none was set."
                next_action = (
                    "The user declined / no budget is set. Set one via set_budget, or run the "
                    "task on free/local providers."
                )
            else:
                summary = f"No eligible provider ({decision.decision})."
                next_action = "Relax constraints, sanitize the payload, or wait for quota/capacity."
            self.memory.update_task(
                task_id, state=TaskState.REJECTED.value, summary=summary,
                recommended_next_action=next_action,
            )
            return self._summary(task_id)

        cfg = self.registry.get(decision.selected_provider)
        assert cfg is not None

        # Agentic execution runs the worker runtime's tool loop in-process and
        # feeds each tool observation back to the model. Those tool observations
        # must never reach an untrusted external model, so agentic is permitted
        # only on (a) an OWNED local resident model, or (b) a TRUSTED, opted-in
        # rented PRIVATE node — a box running YOUR weights ephemerally with no
        # external logging (the whole point of the rented tier). Cloud
        # (external/external_paid) is always rejected; secret_sensitive never
        # reaches here (blocked pre-routing). This guard is belt-and-suspenders:
        # even if routing selected rented for a public task without opt-in/trust,
        # we fail closed. It is a legal ELIGIBLE_PROVIDERS_COMPUTED -> REJECTED,
        # kept BEFORE any spend confirmation so a rejected route never bills.
        if submission.constraints.execution == "agentic":
            is_owned_local = cfg.type == "local" and not RoutingEngine._is_rented(cfg)
            rented_ok = (
                RoutingEngine._is_rented(cfg)
                and submission.constraints.allow_rented
                and RoutingEngine._rented_trust_ok(cfg)
            )
            if not (is_owned_local or rented_ok):
                self._transition(task_id, state, TaskState.REJECTED)
                if RoutingEngine._is_rented(cfg):
                    summary = (
                        f"Agentic execution on a rented node ({cfg.provider_id}) needs "
                        "allow_rented=True and a node whose trust satisfies the repo_sensitive "
                        "policy (ephemeral/encrypted_volume/own_image/no_external_logging)."
                    )
                    next_action = (
                        "Set allow_rented=True and route to a trusted private rented node, or "
                        "resubmit as one-shot."
                    )
                else:
                    summary = (
                        "Agentic execution needs a local or trusted rented node; routing "
                        f"selected {cfg.provider_id} ({cfg.type})."
                    )
                    next_action = (
                        "Resubmit as one-shot, or constrain the task so a local (or trusted "
                        "rented) model is eligible."
                    )
                self.memory.update_task(
                    task_id, state=TaskState.REJECTED.value,
                    summary=summary, recommended_next_action=next_action,
                )
                return self._summary(task_id)

        # Deterministic per-charge user confirmation for real-money routes, BEFORE
        # queueing (so a decline is a legal ELIGIBLE_PROVIDERS_COMPUTED -> REJECTED).
        if cfg.privacy == "external_paid" or RoutingEngine._is_rented(cfg):
            if not self._confirm_charge(cfg, ctx, submission.task, task_id):
                self._transition(task_id, state, TaskState.REJECTED)
                self.memory.update_task(
                    task_id, state=TaskState.REJECTED.value,
                    summary="Spend not confirmed by the user.",
                    recommended_next_action="Approve the charge when prompted, adjust the budget, "
                    "or resubmit constrained to free/local providers.",
                )
                return self._summary(task_id)

        # 13. Reserve cloud quota (pre-queue so a denial is a legal REJECTED).
        reservation = None
        if cfg.type == "cloud":
            reservation = self.quota.reserve(cfg, task_id, in_tok, out_tok)
            if not reservation.granted:
                self._transition(task_id, state, TaskState.REJECTED)
                self.memory.update_task(
                    task_id, state=TaskState.REJECTED.value,
                    summary=f"Quota reservation denied for {cfg.provider_id}: {reservation.reason}.",
                    recommended_next_action="Wait for quota reset or route to a local model.",
                )
                return self._summary(task_id)

        state = self._transition(task_id, state, TaskState.QUEUED)
        state = self._transition(task_id, state, TaskState.DISPATCHED)
        state = self._transition(task_id, state, TaskState.RUNNING)

        if submission.constraints.execution == "agentic":
            return self._run_agentic(
                cfg, submission, task_id, state, max_out, reservation, privacy_class
            )

        gen_req = GenerateRequest(
            request_id=f"req_{uuid.uuid4().hex[:10]}",
            task_id=task_id,
            model_id=decision.selected_model or self.profiles.default_resident_model,
            messages=[{"role": "user", "content": redacted_text if cfg.type == "cloud" else submission.task}],
            max_output_tokens=max_out,
            temperature=0.2,
            priority=submission.constraints.priority,
        )
        started = time.perf_counter()
        try:
            result = self._dispatch(cfg, gen_req)
        except Exception as exc:  # dispatch failure -> FAILED, reconcile as failure.
            latency_ms = (time.perf_counter() - started) * 1000.0
            if reservation is not None:
                self.quota.reconcile(reservation, cfg, 0, 0, status="failed", failure_reason=str(exc))
            self._record_eval(
                cfg, gen_req.model_id, task_id, submission.agent_type, "failed", latency_ms, 0, 0, 0.0
            )
            self.memory.append_log(task_id, f"dispatch_failed: {exc}", level="error")
            self._transition(task_id, state, TaskState.FAILED)
            self.memory.update_task(
                task_id, state=TaskState.FAILED.value,
                summary=f"Dispatch to {cfg.provider_id} failed.",
                recommended_next_action="Inspect logs via get_log_slice; retry or reroute.",
            )
            return self._summary(task_id)
        latency_ms = (time.perf_counter() - started) * 1000.0

        if reservation is not None:
            self.quota.reconcile(reservation, cfg, result.input_tokens, result.output_tokens)
        cost = self.quota.estimate_cost_usd(cfg, result.input_tokens, result.output_tokens)
        self._record_eval(
            cfg, result.model, task_id, submission.agent_type, "completed",
            latency_ms, result.input_tokens, result.output_tokens, cost,
        )

        # 15. Store full result in shared memory (never returned inline). Optional
        # fascia-guard output scan: when enforcing, persist the redacted form so a
        # secret leaked in worker output never lands in the artifact/summary.
        stored_output = result.output_text
        guard_out = guard_hook.scan_output(result.output_text, task_id)
        if guard_out is not None:
            self.memory.append_log(task_id, guard_hook.describe(guard_out, "output"))
            if guard_hook.enforcing() and guard_hook.should_redact_output(guard_out):
                stored_output = guard_out.redacted_text
        output_ref = self.memory.put_artifact(task_id, "output", self._clamp(stored_output))
        state = self._transition(task_id, state, TaskState.ARTIFACTS_WRITTEN)
        state = self._transition(task_id, state, TaskState.CHECKS_RUN)
        state = self._transition(task_id, state, TaskState.REVIEW_READY)
        state = self._transition(task_id, state, TaskState.APPROVED)
        self._transition(task_id, state, TaskState.COMPLETE)

        summary = self._first_line(stored_output)
        self.memory.update_task(
            task_id, state=TaskState.COMPLETE.value, summary=summary,
            recommended_next_action="Read the output artifact chunk if details are needed.",
        )
        self.memory.append_log(
            task_id, f"completed provider={cfg.provider_id} model={result.model} "
            f"in={result.input_tokens} out={result.output_tokens} output_ref={output_ref}"
        )
        # 16. Return compact summary + refs. 17. Decision already logged above.
        return self._summary(task_id)

    # --------------------------------------------------------- agentic dispatch
    def _select_remote_worker(self, privacy_class):
        """Pick a registered remote worker trusted for ``privacy_class`` and
        reporting capacity, returning ``(worker_cfg, runtime)`` or ``None``.

        Trust reuses ``WorkQueue.may_claim`` — the pull federation's live,
        fail-closed tier x privacy predicate — so a worker registered in
        ``remote_workers.yaml`` can only be picked for a class its attested tier
        already admits under ``routing.yaml``. Availability is a ``can_accept``
        probe; any predicate miss, unreachable worker, or malformed response is
        skipped, so a fully-unavailable fleet cleanly falls back to in-process.
        """
        if not self.remote_workers or self.workqueue is None or self.remote_runtime_factory is None:
            return None
        for w in self.remote_workers:
            if not self.workqueue.may_claim(w.tier, privacy_class):
                continue  # tier not trusted for this class -> skip
            try:
                runtime = self.remote_runtime_factory(w)
                if runtime.can_accept().can_accept:
                    return w, runtime
            except Exception:
                continue  # unreachable / malformed -> try the next worker
        return None

    def _run_agentic_remote(
        self, worker_cfg: RemoteWorkerConfig, runtime, submission: TaskSubmission,
        task_id: str, state: TaskState, privacy_class,
    ) -> TaskSummary:
        """Dispatch an agentic task WHOLE to a trusted remote worker over mTLS and
        fold its ``WorkerResult`` into the state machine, mirroring
        ``_run_agentic``'s reconcile/eval/state tail.

        The worker runs its OWN model, so there is no provider ``cfg``, no quota
        reservation, and no router-side spend — the worker self-reports token usage
        in ``WorkerResult.usage``, which we record for observability. Availability
        fallback already happened at selection (``can_accept``); a ``run()`` failure
        AFTER acceptance is a genuine FAILED (the worker may already have performed
        side effects) — never a silent in-process re-run.
        """
        provider_label = f"remote:{worker_cfg.worker_id}"
        # Routing/quota/spend were skipped; walk the linear FSM to RUNNING.
        state = self._transition(task_id, state, TaskState.ELIGIBLE_PROVIDERS_COMPUTED)
        self.memory.append_log(
            task_id,
            f"remote_dispatch worker={worker_cfg.worker_id} tier={worker_cfg.tier} "
            f"privacy_class={privacy_class.value} endpoint={worker_cfg.endpoint}",
        )
        state = self._transition(task_id, state, TaskState.QUEUED)
        state = self._transition(task_id, state, TaskState.DISPATCHED)
        state = self._transition(task_id, state, TaskState.RUNNING)

        started = time.perf_counter()
        try:
            worker = runtime.run(submission, task_id=task_id)
        except Exception as exc:  # accepted then dropped -> genuine FAILED, no re-run.
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._record_eval(
                provider_label, "", task_id, submission.agent_type, "failed",
                latency_ms, 0, 0, 0.0,
            )
            self.memory.append_log(
                task_id, f"remote_dispatch_failed worker={worker_cfg.worker_id}: {exc}", level="error"
            )
            self._transition(task_id, state, TaskState.FAILED)
            self.memory.update_task(
                task_id, state=TaskState.FAILED.value,
                summary=f"Remote agentic dispatch to {worker_cfg.worker_id} failed.",
                recommended_next_action="Inspect logs via get_log_slice; retry or run in-process.",
            )
            return self._summary(task_id)
        latency_ms = (time.perf_counter() - started) * 1000.0

        # Worker-reported usage (None on an older worker -> record zeros + a marker).
        usage = worker.usage
        if usage is None:
            in_tok = out_tok = 0
            model_id = ""
            self.memory.append_log(task_id, f"usage_unreported worker={worker_cfg.worker_id}")
        else:
            in_tok, out_tok, model_id = usage.input_tokens, usage.output_tokens, usage.model_id or ""
        eval_status = "completed" if worker.status == "completed" else "failed"
        # cost=0.0: remote compute is the worker's own; we record usage for
        # observability, not billing against a router-side budget.
        self._record_eval(
            provider_label, model_id, task_id, submission.agent_type, eval_status,
            latency_ms, in_tok, out_tok, 0.0, confidence=worker.confidence,
        )

        output_ref = self.memory.put_artifact(
            task_id, "output", self._clamp(worker.model_dump_json(indent=2))
        )
        self.memory.append_log(
            task_id, f"agentic_remote worker={worker_cfg.worker_id} model={model_id} "
            f"status={worker.status} confidence={worker.confidence} in={in_tok} out={out_tok} "
            f"changed={len(worker.changed_artifacts)} output_ref={output_ref}"
        )

        if worker.status == "completed":
            state = self._transition(task_id, state, TaskState.ARTIFACTS_WRITTEN)
            state = self._transition(task_id, state, TaskState.CHECKS_RUN)
            state = self._transition(task_id, state, TaskState.REVIEW_READY)
            state = self._transition(task_id, state, TaskState.APPROVED)
            self._transition(task_id, state, TaskState.COMPLETE)
            self.memory.update_task(
                task_id, state=TaskState.COMPLETE.value,
                summary=self._first_line(worker.summary) or "Remote agentic task completed.",
                recommended_next_action=worker.recommended_next_action
                or "Read the output artifact chunk if details are needed.",
            )
        else:
            # Loop stopped without a finish (e.g. step cap): not an exception, but
            # not a success — surface as FAILED with the worker's own next-action.
            self._transition(task_id, state, TaskState.FAILED)
            self.memory.update_task(
                task_id, state=TaskState.FAILED.value,
                summary=self._first_line(worker.summary) or "Remote agentic task did not complete.",
                recommended_next_action=worker.recommended_next_action
                or "Raise the step limit or narrow the task, then retry.",
            )
        return self._summary(task_id)

    def _run_agentic(
        self, cfg, submission: TaskSubmission, task_id: str, state: TaskState,
        max_out: int, reservation, privacy_class=None,
    ) -> TaskSummary:
        """Execute the task through the worker runtime's act/tool loop instead of
        a single generation. The model is reached through the gateway (same
        provider/secrets/mTLS as one-shot); token usage is summed across steps
        (and across any delegated sub-runs) and reconciled once. Reached only for
        owned-local or trusted-rented providers (guarded above)."""
        # A rented private node cannot be served through the gateway (it neither
        # provisions nor bills), so agentic on a trusted rented tier takes a
        # dedicated path that acquires the node ONCE, reuses it across every
        # step, bills the rental window once, and releases/reaps after.
        if RoutingEngine._is_rented(cfg):
            return self._run_agentic_rented(cfg, submission, task_id, state, max_out)

        model_id = submission.constraints.require_exact_model or self.profiles.default_resident_model
        # One meter for the whole delegation tree: the parent worker plus every child
        # sub-run and the synthesis call bill against ONE reservation on ONE provider,
        # so usage is summed here and reconciled once (like a flat agentic run).
        meter = {"in": 0, "out": 0, "calls": 0, "children": 0}

        started = time.perf_counter()
        try:
            worker = self._agentic_tree(cfg, submission, task_id, privacy_class, 0, max_out, meter)
        except Exception as exc:  # runtime failure -> FAILED, reconcile as failure.
            latency_ms = (time.perf_counter() - started) * 1000.0
            if reservation is not None:
                self.quota.reconcile(reservation, cfg, meter["in"], meter["out"],
                                     status="failed", failure_reason=str(exc))
            self._record_eval(
                cfg, model_id, task_id, submission.agent_type, "failed", latency_ms,
                meter["in"], meter["out"], 0.0
            )
            self.memory.append_log(task_id, f"agentic_run_failed: {exc}", level="error")
            self._transition(task_id, state, TaskState.FAILED)
            self.memory.update_task(
                task_id, state=TaskState.FAILED.value,
                summary=f"Agentic run on {cfg.provider_id} failed.",
                recommended_next_action="Inspect logs via get_log_slice; retry or reroute.",
            )
            return self._summary(task_id)
        latency_ms = (time.perf_counter() - started) * 1000.0

        in_tok, out_tok = meter["in"], meter["out"]
        if reservation is not None:
            self.quota.reconcile(reservation, cfg, in_tok, out_tok)
        cost = self.quota.estimate_cost_usd(cfg, in_tok, out_tok)
        eval_status = "completed" if worker.status == "completed" else "failed"
        self._record_eval(
            cfg, model_id, task_id, submission.agent_type, eval_status,
            latency_ms, in_tok, out_tok, cost, confidence=worker.confidence,
        )

        # Full structured result to shared memory; the manager sees only a summary.
        output_ref = self.memory.put_artifact(
            task_id, "output", self._clamp(worker.model_dump_json(indent=2))
        )
        self.memory.append_log(
            task_id, f"agentic provider={cfg.provider_id} model={model_id} steps={meter['calls']} "
            f"status={worker.status} confidence={worker.confidence} in={in_tok} out={out_tok} "
            f"changed={len(worker.changed_artifacts)} delegated={meter['children']} "
            f"output_ref={output_ref}"
        )

        if worker.status == "completed":
            state = self._transition(task_id, state, TaskState.ARTIFACTS_WRITTEN)
            state = self._transition(task_id, state, TaskState.CHECKS_RUN)
            state = self._transition(task_id, state, TaskState.REVIEW_READY)
            state = self._transition(task_id, state, TaskState.APPROVED)
            self._transition(task_id, state, TaskState.COMPLETE)
            self.memory.update_task(
                task_id, state=TaskState.COMPLETE.value,
                summary=self._first_line(worker.summary) or "Agentic task completed.",
                recommended_next_action=worker.recommended_next_action
                or "Read the output artifact chunk if details are needed.",
            )
        else:
            # The loop stopped without a finish (e.g. step cap). Not an exception,
            # but not a success either — surface it as FAILED with the runtime's
            # own risks/next-action so the manager can decide.
            self._transition(task_id, state, TaskState.FAILED)
            self.memory.update_task(
                task_id, state=TaskState.FAILED.value,
                summary=self._first_line(worker.summary) or "Agentic task did not complete.",
                recommended_next_action=worker.recommended_next_action
                or "Raise the step limit or narrow the task, then retry.",
            )
        return self._summary(task_id)

    # ------------------------------------------------- hierarchical delegation
    def _agentic_tree(self, cfg, submission, task_id, privacy_class, depth, max_out, meter):
        """Run ONE agentic worker and, if it delegates, its whole sub-tree — returning
        the final (possibly synthesis-folded) WorkerResult. Recursion is bounded: a
        node may delegate only while ``depth < max_delegation_depth`` and the runtime
        advertises the ``delegate`` action only below that limit, so leaves never
        decompose. Every node's token usage accumulates into ``meter`` (in/out/calls),
        so the caller reconciles the whole tree against one reservation. ``meter`` is
        updated as we go, so a mid-tree exception still leaves partial usage to bill."""
        # Lazy import: one-shot-only deployments need not install the runtime (and its
        # langgraph dependency). RuntimeConfig is data; the impl is resolved through
        # _make_local_runtime (injectable).
        from agentconnect.runtime import RuntimeConfig
        from .runtime_dispatch import GatewayModelSource

        model_id = submission.constraints.require_exact_model or self.profiles.default_resident_model
        can_delegate = self.enable_delegation and depth < self.max_delegation_depth
        source = GatewayModelSource(self.gateway, cfg, model_id)
        runtime = self._make_local_runtime(
            source,
            RuntimeConfig(
                model_id=model_id, max_output_tokens=max_out,
                allow_delegation=can_delegate, delegation_depth=depth,
                max_delegation_depth=self.max_delegation_depth, max_subtasks=self.max_subtasks,
            ),
        )
        try:
            worker = runtime.run(submission, task_id=task_id)
        finally:
            meter["in"] += source.total_input_tokens
            meter["out"] += source.total_output_tokens
            meter["calls"] += source.calls

        subtasks = list(getattr(worker, "subtasks", []) or [])
        # No delegation requested (leaf, disabled, or the worker didn't decompose):
        # return the worker verbatim.
        if not (can_delegate and worker.status == "completed" and subtasks):
            return worker

        # Decompose -> execute each child as its own agentic sub-run at depth+1, with a
        # privacy_class clamped to child ⊆ parent (never a downgrade — same monotonicity
        # the WorkQueue enforces on dependency edges). Then synthesize.
        child_records = []
        for i, st in enumerate(subtasks):
            child_pc = self._child_privacy_class(privacy_class, st.privacy_class)
            child_id = f"{task_id}/d{depth + 1}.{i + 1}"
            child_pc_val = child_pc.value if hasattr(child_pc, "value") else child_pc
            if child_pc_val == PrivacyClass.secret_sensitive.value:
                # Same rule as the top-level dispatch guard: secret_sensitive content
                # must never reach an LLM. A child that clamps to it is refused here —
                # it is NOT run on this (or any) model — and folded in as a failure.
                child_worker = WorkerResult(
                    status="failed",
                    summary="Sub-task refused: secret_sensitive content must not reach an LLM.",
                    confidence=0.0,
                    risks=["secret_sensitive_child_refused"],
                )
            else:
                child_sub = TaskSubmission(
                    task=st.task,
                    agent_type=st.agent_type or submission.agent_type,
                    constraints=TaskConstraints(
                        privacy_class=child_pc,
                        execution="agentic",
                        max_output_tokens=submission.constraints.max_output_tokens,
                    ),
                )
                child_worker = self._agentic_tree(
                    cfg, child_sub, child_id, child_pc, depth + 1, max_out, meter
                )
            meter["children"] += 1
            # Child output lands under the PARENT task's artifacts (it's a sub-run, not
            # a first-class queued task) so the whole tree is inspectable from the root.
            self.memory.put_artifact(
                task_id, "child_output", self._clamp(child_worker.model_dump_json(indent=2))
            )
            pc_label = child_pc.value if hasattr(child_pc, "value") else (child_pc or "inherit")
            self.memory.append_log(
                task_id,
                f"delegate_child depth={depth + 1} idx={i + 1} child={child_id} "
                f"privacy={pc_label} status={child_worker.status} "
                f"confidence={child_worker.confidence}",
            )
            child_records.append((st.task, child_worker))

        return self._synthesize_children(
            cfg, model_id, task_id, submission, worker, child_records, max_out, meter
        )

    def _child_privacy_class(self, parent_pc, proposed):
        """Resolve a delegated child's privacy_class, enforcing child ⊆ parent. A
        proposal is accepted only when it is equal-or-more-restrictive than the parent
        (its admissible-tier set is a subset of the parent's — the same test the
        WorkQueue applies to dependency edges); otherwise it is clamped to the parent's
        class so sensitive work can never be laundered down to a looser tier. Fail-closed:
        an unknown/empty proposal set is clamped, not widened."""
        if proposed is None or parent_pc is None:
            # No proposal -> inherit the parent. No parent constraint -> take the
            # proposal (nothing stricter to enforce against).
            return proposed if parent_pc is None else parent_pc
        from agentconnect.common.privacy import allowed_tiers

        prop_val = proposed.value if hasattr(proposed, "value") else proposed
        parent_val = parent_pc.value if hasattr(parent_pc, "value") else parent_pc
        # A KNOWN class may map to an empty tier set (secret_sensitive: un-routable, the
        # strictest of all) — that is still a valid, stricter proposal, so gate on map
        # membership, not on non-emptiness. An UNKNOWN class (absent from the map) is
        # fail-closed: clamped to the parent, never honored.
        classes = self.routing_cfg.privacy.get("classes", {}) or {}
        child_tiers = set(allowed_tiers(self.routing_cfg, prop_val))
        parent_tiers = set(allowed_tiers(self.routing_cfg, parent_val))
        if prop_val in classes and child_tiers.issubset(parent_tiers):
            return proposed
        return parent_pc

    def _synthesize_children(self, cfg, model_id, task_id, submission, parent_worker,
                             child_records, max_out, meter):
        """Fold the parent worker's own findings plus every child summary into ONE
        consolidated parent summary (a single gateway generation on the same provider),
        so the manager sees one summary for the whole sub-tree — recursive context
        virtualization. Confidence collapses to the weakest link; risks and changed
        artifacts union across the tree."""
        lines = [
            f"Parent task: {submission.task}",
            f"Your own findings: {parent_worker.summary or '(none)'}",
            "",
            "Results of the sub-tasks you delegated:",
        ]
        for st_task, cw in child_records:
            lines.append(f"- [{cw.status}] {st_task}: {cw.summary or '(no summary)'}")
        lines += [
            "",
            "Write a SINGLE consolidated summary of the overall task for the manager, "
            "integrating the sub-task results. Be concise; do not repeat this prompt.",
        ]
        from .runtime_dispatch import GatewayModelSource

        source = GatewayModelSource(self.gateway, cfg, model_id)
        req = GenerateRequest(
            request_id=f"req_{task_id}_synth",
            task_id=task_id,
            model_id=model_id,
            messages=[{"role": "user", "content": "\n".join(lines)}],
            max_output_tokens=max_out,
            temperature=0.2,
        )
        try:
            resp = source.generate(req)
            synth_summary = (resp.output_text or "").strip()
        finally:
            meter["in"] += source.total_input_tokens
            meter["out"] += source.total_output_tokens
            meter["calls"] += source.calls

        risks = list(parent_worker.risks)
        changed = list(parent_worker.changed_artifacts)
        confidences = [parent_worker.confidence]
        for _st_task, cw in child_records:
            risks += list(cw.risks)
            changed += [a for a in cw.changed_artifacts if a not in changed]
            confidences.append(cw.confidence)
        return parent_worker.model_copy(update={
            "summary": synth_summary or parent_worker.summary,
            "confidence": min(confidences),
            "risks": risks,
            "changed_artifacts": changed,
        })

    def _run_agentic_rented(
        self, cfg, submission: TaskSubmission, task_id: str, state: TaskState, max_out: int,
    ) -> TaskSummary:
        """Agentic tool loop on a TRUSTED, opted-in rented private node.

        Mirrors ``_dispatch``'s rented branch but acquires the node ONCE, OUTSIDE
        the loop, and reuses it across every step. Privacy: each per-step
        ``generate()`` reaches the acquired node's own mTLS ``LocalClient`` (your
        weights, ephemeral, no external logging — trust-gated by the caller's
        guard), never ``gateway._call_cloud`` — so a repo-sensitive tool trace
        never leaves the rented tier. Billing: the rental window is recorded
        EXACTLY ONCE, at spin-up (only when not reused), outside the loop; the
        per-step generates never bill. Rented is ``cfg.type=="local"`` so
        ``submit_task`` reserved no cloud quota (there is no reconcile / second
        money path). Token totals feed only the evaluation record, as one-shot
        rented does. The node is always released in ``finally`` for the reaper."""
        from agentconnect.runtime import RuntimeConfig
        from .runtime_dispatch import RentedModelSource
        from .provisioning import NodePool, spec_from_provider

        model_id = submission.constraints.require_exact_model or self.profiles.default_resident_model
        pool = self.node_pool or NodePool()
        spec = spec_from_provider(cfg, model_id=model_id)
        factory = self.rented_client_factory or self._default_rented_client

        # Provisioning, client wiring, billing AND the loop all run inside one
        # try so an operational spin-up failure (provisioner.wait_ready raising
        # because the node never boots) degrades to a FAILED TaskSummary instead
        # of escaping raw to the MCP caller — matching the one-shot rented path's
        # contract (_dispatch runs under submit_task's try). The finally still
        # releases even when a *post-acquire* step (factory / billing) raises, so
        # a just-provisioned node is never left un-released. release() is a safe
        # no-op when acquire never populated the pool. ``source`` may be None if
        # we failed before constructing it, so the except guards token totals.
        source = None
        handle = None
        started = time.perf_counter()
        try:
            handle, reused = pool.acquire(cfg, self.provisioner, spec)
            client = factory(cfg, handle)
            # Bill the rental window once, at spin-up (mirrors _dispatch); warm reuse is free.
            if not reused:
                window = cfg.rental.min_rental_seconds if cfg.rental else 0
                self.quota.record_rental_window(cfg, task_id, seconds=window)
            source = RentedModelSource(client, model_id)
            runtime = self._make_local_runtime(
                source, RuntimeConfig(model_id=model_id, max_output_tokens=max_out)
            )
            worker = runtime.run(submission, task_id=task_id)
        except Exception as exc:  # setup/runtime failure -> FAILED (no cloud quota to reconcile).
            latency_ms = (time.perf_counter() - started) * 1000.0
            in_tok = source.total_input_tokens if source is not None else 0
            out_tok = source.total_output_tokens if source is not None else 0
            self._record_eval(
                cfg, model_id, task_id, submission.agent_type, "failed", latency_ms,
                in_tok, out_tok, 0.0,
            )
            self.memory.append_log(task_id, f"agentic_run_failed: {exc}", level="error")
            self._transition(task_id, state, TaskState.FAILED)
            self.memory.update_task(
                task_id, state=TaskState.FAILED.value,
                summary=f"Agentic run on rented node {cfg.provider_id} failed.",
                recommended_next_action="Inspect logs via get_log_slice; retry or reroute.",
            )
            return self._summary(task_id)
        finally:
            # Free the node for the idle reaper even if setup or the loop raised.
            pool.release(cfg)
        latency_ms = (time.perf_counter() - started) * 1000.0

        in_tok, out_tok = source.total_input_tokens, source.total_output_tokens
        cost = self.quota.estimate_cost_usd(cfg, in_tok, out_tok)
        eval_status = "completed" if worker.status == "completed" else "failed"
        self._record_eval(
            cfg, model_id, task_id, submission.agent_type, eval_status,
            latency_ms, in_tok, out_tok, cost, confidence=worker.confidence,
        )

        output_ref = self.memory.put_artifact(
            task_id, "output", self._clamp(worker.model_dump_json(indent=2))
        )
        self.memory.append_log(
            task_id, f"agentic provider={cfg.provider_id} model={model_id} "
            f"rented_node={handle.node_id} steps={source.calls} status={worker.status} "
            f"confidence={worker.confidence} in={in_tok} out={out_tok} "
            f"changed={len(worker.changed_artifacts)} output_ref={output_ref}"
        )

        if worker.status == "completed":
            state = self._transition(task_id, state, TaskState.ARTIFACTS_WRITTEN)
            state = self._transition(task_id, state, TaskState.CHECKS_RUN)
            state = self._transition(task_id, state, TaskState.REVIEW_READY)
            state = self._transition(task_id, state, TaskState.APPROVED)
            self._transition(task_id, state, TaskState.COMPLETE)
            self.memory.update_task(
                task_id, state=TaskState.COMPLETE.value,
                summary=self._first_line(worker.summary) or "Agentic task completed.",
                recommended_next_action=worker.recommended_next_action
                or "Read the output artifact chunk if details are needed.",
            )
        else:
            self._transition(task_id, state, TaskState.FAILED)
            self.memory.update_task(
                task_id, state=TaskState.FAILED.value,
                summary=self._first_line(worker.summary) or "Agentic task did not complete.",
                recommended_next_action=worker.recommended_next_action
                or "Raise the step limit or narrow the task, then retry.",
            )
        return self._summary(task_id)

    # ------------------------------------------------------ other MCP tools
    def get_task_status(self, task_id: str) -> Optional[TaskSummary]:
        if self.memory.get_task(task_id) is None:
            return None
        return self._summary(task_id)

    def get_task_artifacts(self, task_id: str) -> dict[str, str]:
        return self.memory.task_artifacts(task_id)

    def read_artifact_chunk(self, artifact_id: str, offset: int = 0, max_chars: Optional[int] = None) -> dict[str, Any]:
        default = int(self.routing_cfg.mcp_output_policy.get("default_max_chars", 4000))
        hard = int(self.routing_cfg.mcp_output_policy.get("hard_max_chars", 12000))
        mc = min(max_chars or default, hard)
        chunk = self.memory.read_artifact_chunk(artifact_id, offset, mc)
        if chunk is None:
            return {"error": "artifact_not_found", "artifact_id": artifact_id}
        return {
            "artifact_id": chunk.artifact_id,
            "offset": chunk.offset,
            "content": chunk.content,
            "total_size": chunk.total_size,
            "next_offset": chunk.next_offset,
        }

    def get_log_slice(self, task_id: str, level: Optional[str] = None, query: Optional[str] = None, max_lines: int = 100) -> list[dict[str, Any]]:
        return self.memory.get_log_slice(task_id, level=level, query=query, max_lines=max_lines)

    def search_memory(self, query: str, scope: str = "all", limit: int = 20) -> list[dict[str, Any]]:
        return self.memory.search_memory(query, scope=scope, limit=limit)

    def get_router_status(self) -> dict[str, Any]:
        status = self._local_status()
        budget_brief = None
        if self.budget is not None:
            configured = self.budget.is_configured()
            budget_brief = {
                "configured": configured,
                "action_required": None if configured or not self.budget.require_explicit else "set_budget",
            }
        return {
            "policy_version": self.registry.policy_version,
            "providers": [p.provider_id for p in self.registry.all()],
            "local_manager": status.model_dump(mode="json") if status else None,
            "output_policy": self.routing_cfg.mcp_output_policy,
            "budget": budget_brief,
        }

    # ------------------------------------------------------------- budget
    def set_budget(self, amount_usd: float, period: str = "monthly") -> dict[str, Any]:
        """Set the global spend budget (amount + daily/weekly/monthly period). This
        is the ONLY way to set it — there is no default. Persisted across restarts."""
        if self.budget is None:
            return {"error": "budget_manager_unavailable"}
        try:
            self.budget.set(amount_usd, period)
        except ValueError as exc:
            return {"error": str(exc)}
        return self.budget.status(time.time())

    def get_budget_status(self) -> dict[str, Any]:
        if self.budget is None:
            return {"configured": False, "error": "budget_manager_unavailable"}
        return self.budget.status(time.time())

    # ---------------------------------------------------- routing + spend gate
    def _refresh_budget_state(self) -> None:
        if self.budget is not None:
            now = time.time()
            self.engine.set_budget_state(
                self.budget.is_configured(), self.budget.remaining(now),
                self.budget.pressure(now), self.budget.require_explicit,
            )

    def _route_with_budget_prompt(self, ctx: RoutingContext, status) -> RoutingDecision:
        """Route; if the only blocker is a missing budget, prompt the USER directly
        (via the authorizer) to set one and route once more — never delegating the
        money decision to the stochastic agent."""
        decision = None
        for attempt in range(2):
            self._refresh_budget_state()
            decision = self.engine.route(ctx, status)
            if decision.selected_provider is not None:
                return decision
            reasons = {r.reason for r in decision.rejected_options}
            if (
                attempt == 0
                and "budget_not_configured" in reasons
                and self.budget is not None
                and self.authorizer is not None
            ):
                got = self.authorizer.request_budget(self.budget.suggested_period)
                if got and got.get("amount_usd", 0) > 0:
                    try:
                        self.budget.set(got["amount_usd"], got.get("period", self.budget.suggested_period))
                        continue  # re-route now that a budget exists
                    except ValueError:
                        pass
            break
        return decision

    def _confirm_charge(self, cfg, ctx: RoutingContext, task_text: str, task_id: str) -> bool:
        """Ask the user (directly, via the authorizer) to approve THIS charge."""
        now = time.time()
        if RoutingEngine._is_rented(cfg):
            r = cfg.rental
            cost = (r.max_hourly_usd * (r.min_rental_seconds / 3600.0)) if r else 0.0
            kind = "rented_gpu"
        else:
            cost = self.quota.estimate_cost_usd(cfg, ctx.est_input_tokens, ctx.est_output_tokens)
            kind = "paid_cloud"
        configured = self.budget is not None and self.budget.is_configured()
        req = ChargeRequest(
            provider=cfg.provider_id, kind=kind, estimated_cost_usd=cost,
            task_summary=task_text[:120],
            period=self.budget.config().period if configured else None,
            budget_amount_usd=self.budget.config().amount_usd if configured else None,
            remaining_usd=self.budget.remaining(now) if configured else None,
        )
        approved = self.authorizer.confirm_charge(req) if self.authorizer is not None else False
        self.memory.append_log(
            task_id, f"spend_confirmation provider={cfg.provider_id} est_cost={cost:.4f} approved={approved}"
        )
        return approved

    def get_provider_status(self) -> list[dict[str, Any]]:
        out = []
        for cfg in self.registry.all():
            rem = self.quota.remaining(cfg)
            out.append(
                {
                    "provider": cfg.provider_id,
                    "type": cfg.type,
                    "privacy": cfg.privacy,
                    "health": self.registry.health(cfg.provider_id),
                    "capabilities": list(cfg.capabilities),
                    "quota_remaining": rem,
                }
            )
        return out

    def promote_task(self, task_id: str) -> dict[str, Any]:
        task = self.memory.get_task(task_id)
        if task is None:
            return {"error": "task_not_found"}
        self.memory.append_log(task_id, "promoted priority=urgent")
        return {"task_id": task_id, "priority": "urgent", "note": "Priority raised; requeue on next dispatch."}

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        task = self.memory.get_task(task_id)
        if task is None:
            return {"error": "task_not_found"}
        current = TaskState(task["state"])
        if current in TERMINAL_STATES:
            return {"task_id": task_id, "state": current.value, "note": "already terminal"}
        self._transition(task_id, current, TaskState.CANCELLED)
        self.memory.update_task(task_id, summary="Cancelled by manager.")
        return {"task_id": task_id, "state": TaskState.CANCELLED.value}

    # ----------------------------------------------------------- internals
    def _capabilities_for(self, agent_type: Optional[str]) -> tuple[str, ...]:
        mapping = {
            "repo_scout": ("summarization", "coding"),
            "patch_worker": ("patch_generation", "coding"),
            "patch_reviewer": ("review", "coding"),
            "log_summarizer": ("summarization",),
            "test_worker": ("coding",),
            "memory_indexer": ("summarization",),
            "cloud_safe_transformer": ("summarization",),
            "critic_worker": ("review",),
        }
        return mapping.get(agent_type or "", ("coding",))

    @staticmethod
    def _first_line(text: str, limit: int = 240) -> str:
        line = text.strip().splitlines()[0] if text.strip() else ""
        return line[:limit]

    def _summary(self, task_id: str) -> TaskSummary:
        task = self.memory.get_task(task_id)
        assert task is not None
        return TaskSummary(
            task_id=task_id,
            status=TaskState(task["state"]),
            summary=task.get("summary"),
            artifacts=self.memory.task_artifacts(task_id),
            recommended_next_action=task.get("recommended_next_action"),
            risks=task.get("risks", []),
        )
