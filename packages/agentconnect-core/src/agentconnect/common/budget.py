"""Global spend budget with period pacing (mandatory, no silent default).

A single user-set budget — one amount + a daily/weekly/monthly period — that the
router paces and enforces across all real-money spend (paid cloud + rented GPU;
free-tier and owned-local are always $0).

Fail-closed rule: there is **no default amount**. Until the user explicitly sets a
budget (via ``RouterService.set_budget`` -> ``memory.set_setting("budget", ...)``),
:meth:`is_configured` is False and the router treats paid/rented providers as
ineligible, surfacing a "budget required" signal so the manager agent prompts the
user rather than spending against an assumed default.

Once configured, the meter is the sum of committed ``act_cost_usd`` in
``quota_records`` over the current period window (``memory.total_spend_since``).
All methods take an explicit ``now`` (epoch seconds) so period math is deterministic
and testable.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .config import RoutingConfig
from .memory import SharedMemory

_SETTING_KEY = "budget"
_EPS = 1e-9
VALID_PERIODS = ("daily", "weekly", "monthly")


@dataclass(frozen=True)
class BudgetConfig:
    amount_usd: float
    period: str  # daily | weekly | monthly


def _period_window(now: float, period: str) -> tuple[float, float]:
    """[start, end) epoch seconds for the period containing `now` (UTC)."""
    dt = datetime.fromtimestamp(now, tz=timezone.utc)
    if period == "daily":
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    elif period == "weekly":
        monday = dt.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=dt.weekday())
        start = monday
        end = start + timedelta(days=7)
    elif period == "monthly":
        start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        days_in_month = calendar.monthrange(dt.year, dt.month)[1]
        end = start + timedelta(days=days_in_month)
    else:
        raise ValueError(f"Unknown budget period {period!r}; expected one of {VALID_PERIODS}")
    return start.timestamp(), end.timestamp()


class BudgetManager:
    def __init__(self, memory: SharedMemory, routing: Optional[RoutingConfig] = None):
        self.memory = memory
        raw = routing.raw.get("budget", {}) if routing else {}
        # Policy only — never an amount.
        self.require_explicit: bool = bool(raw.get("require_explicit", True))
        self.suggested_period: str = raw.get("suggested_period", "monthly")

    # --------------------------------------------------------------- config
    def config(self) -> Optional[BudgetConfig]:
        s = self.memory.get_setting(_SETTING_KEY)
        if not s:
            return None
        amount = float(s.get("amount_usd", 0.0) or 0.0)
        if amount <= 0:
            return None
        period = s.get("period", self.suggested_period)
        if period not in VALID_PERIODS:
            period = self.suggested_period
        return BudgetConfig(amount_usd=amount, period=period)

    def is_configured(self) -> bool:
        return self.config() is not None

    def set(self, amount_usd: float, period: str = "monthly") -> None:
        if amount_usd <= 0:
            raise ValueError("budget amount_usd must be positive")
        if period not in VALID_PERIODS:
            raise ValueError(f"period must be one of {VALID_PERIODS}")
        self.memory.set_setting(_SETTING_KEY, {"amount_usd": float(amount_usd), "period": period})

    # ------------------------------------------------------------- metering
    def window(self, now: float) -> tuple[float, float]:
        cfg = self.config()
        period = cfg.period if cfg else self.suggested_period
        return _period_window(now, period)

    def spent(self, now: float) -> float:
        start, _ = self.window(now)
        return self.memory.total_spend_since(start)

    def remaining(self, now: float) -> float:
        cfg = self.config()
        if cfg is None:
            return 0.0
        return max(0.0, cfg.amount_usd - self.spent(now))

    def elapsed_fraction(self, now: float) -> float:
        start, end = self.window(now)
        span = end - start
        if span <= _EPS:
            return 1.0
        return max(0.0, min(1.0, (now - start) / span))

    def paced_allowance(self, now: float) -> float:
        cfg = self.config()
        return cfg.amount_usd * self.elapsed_fraction(now) if cfg else 0.0

    def over_pace_usd(self, now: float) -> float:
        return self.spent(now) - self.paced_allowance(now)

    def pressure(self, now: float) -> float:
        """[0,1] steer signal: max of cap-proximity and ahead-of-pace."""
        cfg = self.config()
        if cfg is None or cfg.amount_usd <= _EPS:
            return 0.0
        cap = self.spent(now) / cfg.amount_usd
        pace = self.over_pace_usd(now) / max(0.15 * cfg.amount_usd, _EPS)
        return max(0.0, min(1.0, max(cap, pace)))

    def can_afford(self, cost_usd: float, now: float) -> tuple[bool, str]:
        if not self.is_configured():
            return (not self.require_explicit), (
                "budget_ok" if not self.require_explicit else "budget_not_configured"
            )
        if self.remaining(now) + _EPS < cost_usd:
            return False, "period_budget_exhausted"
        return True, "budget_ok"

    # --------------------------------------------------------------- status
    def status(self, now: float) -> dict[str, Any]:
        cfg = self.config()
        if cfg is None:
            return {
                "configured": False,
                "require_explicit": self.require_explicit,
                "suggested_period": self.suggested_period,
                "action_required": "set_budget" if self.require_explicit else None,
                "message": (
                    "No spend budget set. Paid cloud and rented GPU are disabled until the "
                    "user sets one via set_budget(amount_usd, period)."
                    if self.require_explicit else
                    "No budget set; policy does not require one."
                ),
            }
        start, end = self.window(now)
        spent = self.spent(now)
        elapsed = self.elapsed_fraction(now)
        projected = spent / max(elapsed, _EPS)
        return {
            "configured": True,
            "period": cfg.period,
            "amount_usd": round(cfg.amount_usd, 6),
            "window_start_epoch": start,
            "window_end_epoch": end,
            "reset_at_epoch": end,
            "spent_usd": round(spent, 6),
            "remaining_usd": round(max(0.0, cfg.amount_usd - spent), 6),
            "paced_allowance_usd": round(self.paced_allowance(now), 6),
            "over_pace_usd": round(self.over_pace_usd(now), 6),
            "pressure": round(self.pressure(now), 4),
            "projected_period_spend_usd": round(projected, 6),
            "on_track": projected <= cfg.amount_usd + _EPS,
        }
