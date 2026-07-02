"""Quota reservation & reconciliation (handoff §15).

Prevents concurrent agents from over-consuming a shared free-tier quota by
reserving capacity *before* the provider call and reconciling actual usage
*after*. Reservations are held in-process (fast, deterministic) and committed
usage is persisted to shared memory so limits survive across tasks and daily
resets.

Flow (§15):
    1. estimate -> 2. check -> 3. reserve -> 4. call -> 5. reconcile -> 6. release
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .config import ProviderConfig
from .memory import SharedMemory
from .schemas import QuotaReservation


def _day_start(now: float) -> float:
    """Epoch seconds for the start of the UTC day containing `now`."""
    return now - (now % 86400)


@dataclass
class _LiveReservation:
    reservation_id: str
    provider: str
    task_id: str
    requests: int
    tokens: int
    est_cost_usd: float
    expires_at: float


@dataclass
class QuotaLedger:
    """Tracks reservations + committed usage against per-provider quota rules."""

    memory: SharedMemory
    _reservations: dict[str, _LiveReservation] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # --------------------------------------------------------------- helpers
    def _expire(self, now: float) -> None:
        dead = [rid for rid, r in self._reservations.items() if r.expires_at <= now]
        for rid in dead:
            del self._reservations[rid]

    def _reserved_totals(self, provider: str) -> tuple[int, int, float]:
        req = sum(r.requests for r in self._reservations.values() if r.provider == provider)
        tok = sum(r.tokens for r in self._reservations.values() if r.provider == provider)
        cost = sum(r.est_cost_usd for r in self._reservations.values() if r.provider == provider)
        return req, tok, cost

    @staticmethod
    def estimate_cost_usd(cfg: ProviderConfig, in_tokens: int, out_tokens: int) -> float:
        q = cfg.quota
        pin = q.get("price_per_1k_input_usd", 0.0)
        pout = q.get("price_per_1k_output_usd", 0.0)
        return (in_tokens / 1000.0) * pin + (out_tokens / 1000.0) * pout

    # ------------------------------------------------------------ public API
    def remaining(self, cfg: ProviderConfig, now: Optional[float] = None) -> dict[str, float]:
        """Remaining quota headroom for a provider, accounting for committed
        usage today AND outstanding reservations. Returns fractions in [0,1] plus
        absolute remainders. Local GPU providers are considered unlimited here
        (their admission is decided by the Local Model Manager)."""
        now = time.time() if now is None else now
        with self._lock:
            self._expire(now)
            res_req, res_tok, res_cost = self._reserved_totals(cfg.provider_id)
        q = cfg.quota
        kind = q.get("kind")
        if kind in (None, "local_gpu"):
            return {"kind": kind or "unknown", "unlimited": 1.0}

        used = self.memory.quota_usage_since(cfg.provider_id, _day_start(now))
        out: dict[str, float] = {}
        if "max_daily_requests" in q:
            cap = q["max_daily_requests"]
            out["requests_remaining"] = max(0, cap - used["requests"] - res_req)
            out["requests_frac"] = out["requests_remaining"] / cap if cap else 0.0
        if "max_daily_tokens" in q:
            cap = q["max_daily_tokens"]
            out["tokens_remaining"] = max(0, cap - used["tokens"] - res_tok)
            out["tokens_frac"] = out["tokens_remaining"] / cap if cap else 0.0
        if "max_daily_spend_usd" in q:
            cap = q["max_daily_spend_usd"]
            out["spend_remaining_usd"] = max(0.0, cap - used["cost"] - res_cost)
            out["spend_frac"] = out["spend_remaining_usd"] / cap if cap else 0.0
        return out

    def can_reserve(self, cfg: ProviderConfig, in_tokens: int, out_tokens: int) -> tuple[bool, str]:
        rem = self.remaining(cfg)
        if rem.get("unlimited"):
            return True, "local_gpu_admission_delegated"
        tokens = in_tokens + out_tokens
        if "requests_remaining" in rem and rem["requests_remaining"] < 1:
            return False, "daily_request_quota_exhausted"
        if "tokens_remaining" in rem and rem["tokens_remaining"] < tokens:
            return False, "daily_token_quota_insufficient"
        if "spend_remaining_usd" in rem:
            cost = self.estimate_cost_usd(cfg, in_tokens, out_tokens)
            if rem["spend_remaining_usd"] < cost:
                return False, "daily_spend_budget_exhausted"
        return True, "capacity_available"

    def reserve(
        self, cfg: ProviderConfig, task_id: str, in_tokens: int, out_tokens: int, ttl: int = 120
    ) -> QuotaReservation:
        ok, reason = self.can_reserve(cfg, in_tokens, out_tokens)
        tokens = in_tokens + out_tokens
        cost = self.estimate_cost_usd(cfg, in_tokens, out_tokens)
        rid = f"resv_{uuid.uuid4().hex[:10]}"
        if not ok:
            return QuotaReservation(
                reservation_id=rid, provider=cfg.provider_id, task_id=task_id,
                estimated_input_tokens=in_tokens, estimated_output_tokens=out_tokens,
                requests=1, tokens=tokens, expires_in_seconds=ttl, granted=False, reason=reason,
            )
        now = time.time()
        with self._lock:
            self._reservations[rid] = _LiveReservation(
                reservation_id=rid, provider=cfg.provider_id, task_id=task_id,
                requests=1, tokens=tokens, est_cost_usd=cost, expires_at=now + ttl,
            )
        return QuotaReservation(
            reservation_id=rid, provider=cfg.provider_id, task_id=task_id,
            estimated_input_tokens=in_tokens, estimated_output_tokens=out_tokens,
            requests=1, tokens=tokens, expires_in_seconds=ttl, granted=True, reason=reason,
        )

    def reconcile(
        self,
        reservation: QuotaReservation,
        cfg: ProviderConfig,
        act_input: int,
        act_output: int,
        status: str = "completed",
        failure_reason: Optional[str] = None,
    ) -> None:
        """Commit actual usage to shared memory and release the reservation (§15)."""
        with self._lock:
            self._reservations.pop(reservation.reservation_id, None)
        self.memory.record_quota_usage(
            {
                "provider": cfg.provider_id,
                "task_id": reservation.task_id,
                "est_input": reservation.estimated_input_tokens,
                "est_output": reservation.estimated_output_tokens,
                "act_input": act_input,
                "act_output": act_output,
                "requests": 1 if status == "completed" else 0,
                "est_cost_usd": self.estimate_cost_usd(
                    cfg, reservation.estimated_input_tokens, reservation.estimated_output_tokens
                ),
                "act_cost_usd": self.estimate_cost_usd(cfg, act_input, act_output),
                "status": status,
                "failure_reason": failure_reason,
            }
        )

    def release(self, reservation: QuotaReservation) -> None:
        """Drop a reservation without committing usage (e.g. task cancelled pre-call)."""
        with self._lock:
            self._reservations.pop(reservation.reservation_id, None)
