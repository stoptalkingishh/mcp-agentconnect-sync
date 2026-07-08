"""Per-provider circuit breaker (native reimplementation of the OmniRoute
resilience concept — no dependency on OmniRoute itself).

Tracks consecutive outbound-call failures per provider and trips a breaker
open so the routing engine stops sending traffic to a provider that is
actively failing, instead of discovering the failure fresh on every task.
After a cooldown, one probe call is allowed through (half-open); a successful
probe closes the breaker, a failed one re-opens it with a fresh cooldown.

Deterministic and in-process, same philosophy as :class:`QuotaLedger` and
:class:`ProviderRegistry` — no background thread, no randomness. State is
advanced only when explicitly queried (`is_open`) or reported (`record_success`
/ `record_failure`), both called from the router service on the request path.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


@dataclass
class _BreakerState:
    state: str = CLOSED
    consecutive_failures: int = 0
    opened_at: Optional[float] = None
    last_failure_reason: Optional[str] = None


@dataclass
class CircuitBreakerRegistry:
    """In-process breaker state, one per provider id.

    ``failure_threshold``/``cooldown_seconds`` are the defaults; ``overrides``
    is a ``provider_id -> {failure_threshold, cooldown_seconds}`` map (from
    ``config/routing.yaml`` ``resilience.circuit_breaker.overrides``) applied
    on top of the defaults for that one provider.
    """

    failure_threshold: int = 5
    cooldown_seconds: float = 60.0
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    _states: dict[str, _BreakerState] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _threshold_for(self, provider_id: str) -> int:
        return int(self.overrides.get(provider_id, {}).get("failure_threshold", self.failure_threshold))

    def _cooldown_for(self, provider_id: str) -> float:
        return float(self.overrides.get(provider_id, {}).get("cooldown_seconds", self.cooldown_seconds))

    def _state_for(self, provider_id: str) -> _BreakerState:
        return self._states.setdefault(provider_id, _BreakerState())

    def record_success(self, provider_id: str) -> None:
        """A real call to ``provider_id`` succeeded — reset the failure count
        and close the breaker (a successful half-open probe closes it too)."""
        with self._lock:
            st = self._state_for(provider_id)
            st.state = CLOSED
            st.consecutive_failures = 0
            st.opened_at = None
            st.last_failure_reason = None

    def record_failure(self, provider_id: str, reason: Optional[str] = None) -> None:
        """A real call to ``provider_id`` failed. Trips the breaker open once
        ``consecutive_failures`` reaches the threshold (or immediately
        re-opens it if a half-open probe was the one that just failed)."""
        with self._lock:
            st = self._state_for(provider_id)
            st.consecutive_failures += 1
            st.last_failure_reason = reason
            if st.state == HALF_OPEN or st.consecutive_failures >= self._threshold_for(provider_id):
                st.state = OPEN
                st.opened_at = time.time()

    def is_open(self, provider_id: str, now: Optional[float] = None) -> bool:
        """Whether the breaker currently blocks calls to ``provider_id``.

        Advances ``open -> half_open`` once the cooldown has elapsed, letting
        exactly one probe call through (returns ``False`` for that call) —
        the caller is expected to report its outcome via
        ``record_success``/``record_failure`` immediately after."""
        now = time.time() if now is None else now
        with self._lock:
            st = self._states.get(provider_id)
            if st is None or st.state == CLOSED:
                return False
            if st.state == HALF_OPEN:
                return False
            # state == OPEN
            opened_at = st.opened_at or now
            if now - opened_at >= self._cooldown_for(provider_id):
                st.state = HALF_OPEN
                return False
            return True

    def status(self, provider_id: str) -> dict[str, Any]:
        st = self._states.get(provider_id) or _BreakerState()
        return {
            "state": st.state,
            "consecutive_failures": st.consecutive_failures,
            "opened_at": st.opened_at,
            "last_failure_reason": st.last_failure_reason,
        }

    def status_all(self) -> dict[str, dict[str, Any]]:
        return {pid: self.status(pid) for pid in self._states}

    @classmethod
    def from_config(cls, resilience: dict[str, Any]) -> "CircuitBreakerRegistry":
        """Build from ``routing.yaml``'s ``resilience`` section. Returns a
        registry with the breaker effectively disabled (never trips — an
        unreachable threshold) if ``circuit_breaker.enabled`` is false."""
        cb = (resilience or {}).get("circuit_breaker", {})
        enabled = cb.get("enabled", True)
        threshold = int(cb.get("failure_threshold", 5)) if enabled else 2**31
        return cls(
            failure_threshold=threshold,
            cooldown_seconds=float(cb.get("cooldown_seconds", 60.0)),
            overrides=dict(cb.get("overrides", {}) or {}),
        )
