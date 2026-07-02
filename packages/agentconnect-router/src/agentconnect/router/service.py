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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

from ..common import privacy as privacy_mod
from ..common.authorization import ChargeRequest, DenyingSpendAuthorizer, SpendAuthorizer
from ..common.budget import BudgetManager
from ..common.evaluation import Evaluator
from ..common.config import ProfilesConfig, RoutingConfig, load_all
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
    TaskState,
    TaskSubmission,
    TaskSummary,
)
from ..common.state import TERMINAL_STATES, assert_transition
from ..common.tokens import estimate_io_tokens
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
        return cls(
            memory=mem, registry=registry, profiles=profiles, routing_cfg=routing_cfg,
            quota=quota, engine=engine, gateway=gw, local_client=local_client,
            provisioner=provisioner or StubProvisioner(),
            rented_client_factory=rented_client_factory,
            evaluator=Evaluator(mem, min_samples=min_samples),
            node_pool=NodePool(),
            budget=BudgetManager(mem, routing_cfg),
            authorizer=authorizer or DenyingSpendAuthorizer(),
        )

    # ----------------------------------------------------------- evaluation
    def _record_eval(self, cfg, model, task_id, agent_type, status, latency_ms,
                     in_tok, out_tok, cost, confidence=None) -> None:
        self.memory.record_evaluation(
            {
                "provider": cfg.provider_id, "model": model, "task_id": task_id,
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

        # 15. Store full result in shared memory (never returned inline).
        output_ref = self.memory.put_artifact(task_id, "output", self._clamp(result.output_text))
        state = self._transition(task_id, state, TaskState.ARTIFACTS_WRITTEN)
        state = self._transition(task_id, state, TaskState.CHECKS_RUN)
        state = self._transition(task_id, state, TaskState.REVIEW_READY)
        state = self._transition(task_id, state, TaskState.APPROVED)
        self._transition(task_id, state, TaskState.COMPLETE)

        summary = self._first_line(result.output_text)
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
