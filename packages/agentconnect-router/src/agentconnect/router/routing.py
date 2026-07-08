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
    # Opt-in for running a repo_sensitive task on a rented GPU node (Goal 4).
    allow_rented: bool = False


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
        # provider_id -> bounded [-1,1] learned-quality signal (Phase 6). Refreshed
        # by the service from observed outcomes before each routing pass.
        self._learned: dict[str, float] = {}
        # Global spend-budget snapshot, refreshed by the service each pass. Default
        # is "not configured" -> paid/rented fail closed (mandatory-prompt policy).
        self._budget: dict = {
            "configured": False, "remaining_usd": 0.0, "pressure": 0.0, "require_explicit": True,
        }
        # provider_id -> currently circuit-tripped. Refreshed by the service
        # (from CircuitBreakerRegistry.is_open) right before each route() call,
        # same pattern as _budget above.
        self._circuit_open: set[str] = set()

    def set_learned_quality(self, learned: dict[str, float]) -> None:
        self._learned = dict(learned or {})

    def set_circuit_state(self, open_provider_ids: set[str]) -> None:
        self._circuit_open = set(open_provider_ids or ())

    def set_budget_state(
        self, configured: bool, remaining_usd: float, pressure: float, require_explicit: bool
    ) -> None:
        self._budget = {
            "configured": bool(configured),
            "remaining_usd": float(remaining_usd),
            "pressure": float(pressure),
            "require_explicit": bool(require_explicit),
        }

    # --------------------------------------------------------------- budget
    def _call_cost(self, cfg: ProviderConfig, ctx: RoutingContext) -> float:
        if self._is_rented(cfg):
            r = cfg.rental
            return (r.max_hourly_usd * (r.min_rental_seconds / 3600.0)) if r else 0.0
        return self.quota.estimate_cost_usd(cfg, ctx.est_input_tokens, ctx.est_output_tokens)

    def _budget_gate(self, cfg: ProviderConfig, ctx: RoutingContext) -> Optional[str]:
        """Global spend-budget hard gate for paid/rented providers. Returns a
        rejection reason or None. Layered on top of per-provider caps."""
        b = self._budget
        if b["require_explicit"] and not b["configured"]:
            return "budget_not_configured"
        if b["configured"] and self._call_cost(cfg, ctx) > b["remaining_usd"] + 1e-9:
            return "period_budget_exhausted"
        return None

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
    def _allowed_privacy_tiers(self, ctx: RoutingContext) -> list[str]:
        classes = self.routing.privacy.get("classes", {})
        tiers = list(classes.get(ctx.privacy_class.value, []))
        # A repo_sensitive task may use a rented node ONLY with explicit opt-in
        # (handoff Goal 4). secret_sensitive/restricted are never widened.
        if (
            ctx.allow_rented
            and ctx.privacy_class == PrivacyClass.repo_sensitive
            and "private_rented" not in tiers
        ):
            tiers.append("private_rented")
        return tiers

    @staticmethod
    def _is_rented(cfg: ProviderConfig) -> bool:
        return cfg.type == "local" and (cfg.node_class == "rented" or cfg.privacy == "private_rented")

    @staticmethod
    def _rented_trust_ok(cfg: ProviderConfig) -> bool:
        t = (cfg.rental.trust if cfg.rental else {}) or {}
        return all(t.get(k) for k in ("ephemeral", "encrypted_volume", "own_image", "no_external_logging"))

    def eligibility(
        self, ctx: RoutingContext, cfg: ProviderConfig, status: Optional[ManagerStatus]
    ) -> tuple[bool, str]:
        """Apply HARD constraints (§12). Returns (eligible, reason_if_not)."""
        allowed_tiers = self._allowed_privacy_tiers(ctx)
        if not allowed_tiers:
            return False, "privacy_class_blocks_all_llm_routing"
        if cfg.privacy not in allowed_tiers:
            return False, "privacy_policy_blocks_provider_tier"

        if not self.registry.is_available(cfg.provider_id):
            return False, "provider_unhealthy"
        if cfg.provider_id in self._circuit_open:
            return False, "circuit_open"

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
            if cfg.privacy == "external_paid":
                gate = self._budget_gate(cfg, ctx)
                if gate:
                    return False, gate
        elif self._is_rented(cfg):
            # Rented node: provisioned on demand, so no live ManagerStatus is
            # required at routing time. Enforce trust policy + rental budget.
            if ctx.privacy_class == PrivacyClass.repo_sensitive and not self._rented_trust_ok(cfg):
                return False, "rented_node_trust_policy_unmet"
            ok, reason = self.quota.can_reserve_rental(cfg)
            if not ok:
                return False, reason
            gate = self._budget_gate(cfg, ctx)
            if gate:
                return False, gate
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

        # expected_quality: paid ~ rented-large > local-large > free-cloud.
        quality_base = {
            "external_paid": 1.0, "private_rented": 0.95, "local_only": 0.85, "external": 0.6,
        }.get(cfg.privacy, 0.5)
        if ctx.quality == "high":
            quality_base *= 1.15
        terms["expected_quality"] = quality_base * w.get("expected_quality", 2.0)

        # privacy_fit: owned local best; rented (your weights, others' hardware) mid.
        privacy_fit = {"local_only": 1.0, "private_rented": 0.7}.get(cfg.privacy, 0.5)
        terms["privacy_fit"] = privacy_fit * w.get("privacy_fit", 1.5)
        terms["availability"] = (1.0 if self.registry.is_available(cfg.provider_id) else 0.0) * w.get("availability", 1.0)

        if self._is_rented(cfg):
            # Rented GPU node: pay a spin-up/min-window setup penalty and an
            # hourly-cost penalty vs. the daily rental budget. Same "is the
            # expensive setup worth it?" logic as model switching, one level up.
            model_id = ctx.require_exact_model or self._profile_preferred(ctx.profile) or self.profiles.default_resident_model
            terms["latency_fit"] = 0.4 * w.get("latency_fit", 1.0)
            terms["residency_bonus"] = 0.0
            terms["model_switch_penalty"] = 0.0
            terms["queue_delay_penalty"] = 0.0
            terms["cost_penalty"] = 0.0
            terms["opportunity_cost"] = 0.0
            terms["quota_scarcity_penalty"] = 0.0
            terms["rental_setup_penalty"] = -w.get("rental_setup_penalty", 2.0)
            hourly = cfg.rental.max_hourly_usd if cfg.rental else 0.0
            window_cost = hourly * ((cfg.rental.min_rental_seconds if cfg.rental else 0) / 3600.0)
            rem = self.quota.rental_remaining_usd(cfg)
            frac = 0.0 if rem == float("inf") else min(1.0, window_cost / max(rem, 1e-9))
            terms["rental_cost_penalty"] = -frac * w.get("rental_cost_penalty", 2.0)
        elif cfg.type == "local" and status is not None:
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

        # Learned-quality prior from observed outcomes (Phase 6). Bounded [-1,1],
        # so it tilts close calls without overriding hard constraints or dominating.
        terms["learned_quality"] = self._learned.get(cfg.provider_id, 0.0) * w.get("learned_quality", 1.0)

        # Budget pace pressure: as spend runs ahead of pace / nears the cap, steer
        # paid & rented down so the router prefers free/local (soft even-burn).
        if self._budget.get("configured") and (cfg.privacy == "external_paid" or self._is_rented(cfg)):
            terms["budget_pressure_penalty"] = -self._budget["pressure"] * w.get("budget_pressure_penalty", 3.0)
        else:
            terms["budget_pressure_penalty"] = 0.0

        total = round(sum(terms.values()), 4)
        return ScoreBreakdown(provider=cfg.provider_id, model=model_id, total=total, terms=terms)

    @staticmethod
    def _scarcity_fraction(rem: dict[str, float]) -> Optional[float]:
        fracs = [v for k, v in rem.items() if k.endswith("_frac")]
        return min(fracs) if fracs else None

    def _profile_preferred(self, profile: Optional[str]) -> Optional[str]:
        prof = self.profiles.profiles.get(profile or "", {})
        pref = prof.get("preferred")
        return pref if pref not in ("current_resident", "any", None) else None

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
                if not self._allowed_privacy_tiers(ctx)
                else "no_eligible_provider"
            )
            return RoutingDecision(
                task_id=ctx.task_id, decision=decision, selected_provider=None, selected_model=None,
                rejected_options=rejected, scores=scores, policy_version=self.registry.policy_version,
            )

        best = scores[0]
        cfg = self.registry.get(best.provider)
        assert cfg is not None
        if self._is_rented(cfg):
            decision = "route_to_rented_node"
        elif cfg.type == "local":
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
