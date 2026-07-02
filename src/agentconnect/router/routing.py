"""Deterministic routing engine (handoff §10, §11, §12, §16, §17).

Given a classified task, this engine computes eligible providers (applying HARD
privacy/quota/context constraints), resolves capability profiles against live
local residency, scores the survivors, and returns an explainable
:class:`RoutingDecision`.

Determinism rule (§10): randomness may live inside model generation, never inside
infrastructure policy. Everything here is a pure function of (task, config, live
status snapshots) — no randomness, no hidden clocks beyond quota's day boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..common.config import ProfilesConfig, ProviderConfig, RoutingConfig
from ..common.providers import ProviderRegistry
from ..common.quota import QuotaLedger
from ..common.schemas import (
    ManagerStatus,
    Priority,
    PrivacyClass,
    RejectedOption,
    RoutingDecision,
    ScoreBreakdown,
)


@dataclass
class RoutingContext:
    task_id: str
    privacy_class: PrivacyClass
    needed_capabilities: tuple[str, ...] = ()
    profile: Optional[str] = None
    require_exact_model: Optional[str] = None
    est_input_tokens: int = 0
    est_output_tokens: int = 0
    allow_external: bool = True
    allow_paid: bool = False
    priority: Priority = Priority.normal
    quality: str = "standard"  # standard | high | best_effort
    cloud_safe: bool = True  # result of the redaction pass, if run
    # How many same-model tasks are queued behind this one (feeds switch policy §16).
    pending_same_model_batch: int = 0


@dataclass
class ModelChoice:
    model_id: Optional[str]
    switch_required: bool
    reason: str


class RoutingEngine:
    def __init__(
        self,
        registry: ProviderRegistry,
        profiles: ProfilesConfig,
        routing: RoutingConfig,
        quota: QuotaLedger,
    ):
        self.registry = registry
        self.profiles = profiles
        self.routing = routing
        self.quota = quota
        self._w = routing.scoring.get("weights", {})
        self._scarcity_threshold = routing.scoring.get("quota_scarcity_threshold_pct", 0.2)

    # ------------------------------------------------------------- profile
    def resolve_local_model(self, ctx: RoutingContext, status: ManagerStatus) -> ModelChoice:
        """Resolve a capability profile to a concrete local model, honoring the
        residency-preference policy (§16, §17)."""
        loaded = status.loaded_model.model_id if status.loaded_model else None
        available = {m.model_id for m in status.available_models}

        if ctx.require_exact_model:
            m = ctx.require_exact_model
            return ModelChoice(
                model_id=m if m in available else None,
                switch_required=(m != loaded),
                reason="exact_model_requested" if m in available else "exact_model_unavailable",
            )

        prof = self.profiles.profiles.get(ctx.profile or "resident_ok", {})
        preferred = prof.get("preferred")
        acceptable = list(prof.get("acceptable", []))

        # Resolve sentinels.
        def deref(name: Optional[str]) -> Optional[str]:
            if name == "current_resident":
                return loaded
            if name == "any":
                return loaded or (next(iter(available)) if available else None)
            return name

        pref_model = deref(preferred)
        acceptable_models = [deref(a) for a in acceptable]
        acceptable_models = [a for a in acceptable_models if a in available]

        prefer_resident = self.routing.model_switching.get("prefer_resident_model", True)

        # If we prefer the resident model and it satisfies the profile, use it.
        resident_ok = loaded is not None and (
            loaded == pref_model or loaded in acceptable_models or preferred in ("current_resident", "any")
        )
        if prefer_resident and resident_ok:
            return ModelChoice(model_id=loaded, switch_required=False, reason="resident_satisfies_profile")

        # Otherwise prefer the profile's preferred model if installed.
        if pref_model in available:
            return ModelChoice(
                model_id=pref_model,
                switch_required=(pref_model != loaded),
                reason="preferred_model",
            )
        if acceptable_models:
            chosen = acceptable_models[0]
            return ModelChoice(model_id=chosen, switch_required=(chosen != loaded), reason="acceptable_model")
        # Fall back to whatever is resident.
        if loaded:
            return ModelChoice(model_id=loaded, switch_required=False, reason="fallback_resident")
        return ModelChoice(model_id=None, switch_required=False, reason="no_local_model_available")

    def _switch_allowed(self, ctx: RoutingContext, status: ManagerStatus) -> bool:
        ms = self.routing.model_switching
        if ctx.priority == Priority.urgent and ms.get("urgent_switch_allowed", True):
            return True
        if ms.get("avoid_switch_if_current_queue_nonempty", True) and status.queue.local_waiting > 0:
            return False
        return ctx.pending_same_model_batch >= ms.get("min_batch_size_for_switch", 4)

    # ------------------------------------------------------- eligibility
    def _allowed_privacy_tiers(self, privacy_class: PrivacyClass) -> list[str]:
        classes = self.routing.privacy.get("classes", {})
        return list(classes.get(privacy_class.value, []))

    def eligibility(
        self, ctx: RoutingContext, cfg: ProviderConfig, status: Optional[ManagerStatus]
    ) -> tuple[bool, str]:
        """Apply HARD constraints (§12). Returns (eligible, reason_if_not)."""
        allowed_tiers = self._allowed_privacy_tiers(ctx.privacy_class)
        if not allowed_tiers:
            return False, "privacy_class_blocks_all_llm_routing"
        if cfg.privacy not in allowed_tiers:
            return False, "privacy_policy_blocks_provider_tier"

        if not self.registry.is_available(cfg.provider_id):
            return False, "provider_unhealthy"

        if cfg.type == "cloud":
            if not ctx.allow_external:
                return False, "external_routing_not_allowed_for_task"
            # low_sensitive requires a successful redaction pass.
            if ctx.privacy_class in (PrivacyClass.low_sensitive,) and not ctx.cloud_safe:
                return False, "redaction_failed_not_cloud_safe"
            if cfg.privacy == "external_paid":
                if not ctx.allow_paid:
                    return False, "paid_routing_not_allowed_for_task"
            ok, reason = self.quota.can_reserve(cfg, ctx.est_input_tokens, ctx.est_output_tokens)
            if not ok:
                return False, reason
        elif cfg.type == "local":
            if status is None:
                return False, "local_manager_status_unavailable"
            choice = self.resolve_local_model(ctx, status)
            if choice.model_id is None:
                return False, "no_eligible_local_model"
            ca = status_can_accept(status, choice, ctx)
            if not ca[0]:
                return False, ca[1]
            if choice.switch_required and not self._switch_allowed(ctx, status):
                return False, "not_loaded_and_switch_threshold_not_met"
        return True, ""

    # ------------------------------------------------------------- scoring
    def score(
        self, ctx: RoutingContext, cfg: ProviderConfig, status: Optional[ManagerStatus]
    ) -> ScoreBreakdown:
        w = self._w
        terms: dict[str, float] = {}
        model_id: Optional[str] = None

        cap = self.registry.capability_overlap(cfg, ctx.needed_capabilities)
        terms["capability_fit"] = cap * w.get("capability_fit", 3.0)

        # expected_quality: paid > local-large > free-cloud, scaled by requirement.
        quality_base = {"external_paid": 1.0, "local_only": 0.85, "external": 0.6}.get(cfg.privacy, 0.5)
        if ctx.quality == "high":
            quality_base *= 1.15
        terms["expected_quality"] = quality_base * w.get("expected_quality", 2.0)

        terms["privacy_fit"] = (1.0 if cfg.privacy == "local_only" else 0.5) * w.get("privacy_fit", 1.5)
        terms["availability"] = (1.0 if self.registry.is_available(cfg.provider_id) else 0.0) * w.get("availability", 1.0)

        if cfg.type == "local" and status is not None:
            choice = self.resolve_local_model(ctx, status)
            model_id = choice.model_id
            terms["latency_fit"] = (1.0 if not choice.switch_required else 0.3) * w.get("latency_fit", 1.0)
            terms["residency_bonus"] = (w.get("residency_bonus", 2.5) if not choice.switch_required else 0.0)
            terms["model_switch_penalty"] = -(w.get("model_switch_penalty", 3.0) if choice.switch_required else 0.0)
            queue_wait = status.queue.oldest_wait_seconds + 6 * status.queue.local_waiting
            terms["queue_delay_penalty"] = -min(1.0, queue_wait / 60.0) * w.get("queue_delay_penalty", 1.5)
            terms["cost_penalty"] = 0.0
            terms["opportunity_cost"] = 0.0
        else:
            terms["latency_fit"] = 0.6 * w.get("latency_fit", 1.0)
            terms["residency_bonus"] = 0.0
            terms["model_switch_penalty"] = 0.0
            terms["queue_delay_penalty"] = 0.0
            # cost penalty for paid providers.
            if cfg.privacy == "external_paid":
                cost = self.quota.estimate_cost_usd(cfg, ctx.est_input_tokens, ctx.est_output_tokens)
                terms["cost_penalty"] = -min(1.0, cost / 0.05) * w.get("cost_penalty", 2.0)
            else:
                terms["cost_penalty"] = 0.0
            # scarcity + opportunity cost for consuming a limited free tier.
            rem = self.quota.remaining(cfg)
            scarcity = self._scarcity_fraction(rem)
            if scarcity is not None and scarcity < self._scarcity_threshold:
                terms["quota_scarcity_penalty"] = -(1.0 - scarcity) * w.get("quota_scarcity_penalty", 2.0)
            else:
                terms["quota_scarcity_penalty"] = 0.0
            # Opportunity cost: using cloud when the task is local-eligible wastes
            # scarce quota. Public tasks incur it more than sensitive ones.
            local_capable = ctx.privacy_class in (
                PrivacyClass.public, PrivacyClass.low_sensitive, PrivacyClass.repo_sensitive, PrivacyClass.restricted
            )
            terms["opportunity_cost"] = -(0.5 if local_capable else 0.0) * w.get("opportunity_cost", 1.5)

        total = round(sum(terms.values()), 4)
        return ScoreBreakdown(provider=cfg.provider_id, model=model_id, total=total, terms=terms)

    @staticmethod
    def _scarcity_fraction(rem: dict[str, float]) -> Optional[float]:
        fracs = [v for k, v in rem.items() if k.endswith("_frac")]
        return min(fracs) if fracs else None

    # -------------------------------------------------------------- route
    def route(self, ctx: RoutingContext, status: Optional[ManagerStatus]) -> RoutingDecision:
        """Full deterministic routing flow (§11 steps 8-12, 17)."""
        rejected: list[RejectedOption] = []
        scores: list[ScoreBreakdown] = []

        for cfg in self.registry.all():
            eligible, reason = self.eligibility(ctx, cfg, status)
            if not eligible:
                rejected.append(RejectedOption(provider=cfg.provider_id, reason=reason))
                continue
            scores.append(self.score(ctx, cfg, status))

        scores.sort(key=lambda s: s.total, reverse=True)

        if not scores:
            decision = (
                "blocked_secret_sensitive"
                if not self._allowed_privacy_tiers(ctx.privacy_class)
                else "no_eligible_provider"
            )
            return RoutingDecision(
                task_id=ctx.task_id, decision=decision, selected_provider=None, selected_model=None,
                rejected_options=rejected, scores=scores, policy_version=self.registry.policy_version,
            )

        best = scores[0]
        cfg = self.registry.get(best.provider)
        assert cfg is not None
        if cfg.type == "local":
            choice = self.resolve_local_model(ctx, status) if status else ModelChoice(None, False, "")
            decision = (
                "route_to_local_resident_model" if not choice.switch_required else "route_to_local_after_switch"
            )
        else:
            decision = "route_to_cloud_provider"

        return RoutingDecision(
            task_id=ctx.task_id,
            decision=decision,
            selected_provider=best.provider,
            selected_model=best.model,
            rejected_options=rejected,
            scores=scores,
            policy_version=self.registry.policy_version,
        )


def status_can_accept(status: ManagerStatus, choice: "ModelChoice", ctx: RoutingContext) -> tuple[bool, str]:
    """Context-cap admission check derived from the published status (no call)."""
    model_id = choice.model_id
    meta = next((m for m in status.available_models if m.model_id == model_id), None)
    if meta is None:
        return False, "model_not_available"
    total_ctx = ctx.est_input_tokens + ctx.est_output_tokens
    if total_ctx > meta.max_model_len:
        return False, f"context_exceeds_max_model_len({total_ctx}>{meta.max_model_len})"
    return True, "capacity_available"
