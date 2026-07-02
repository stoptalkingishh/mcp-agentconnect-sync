"""Direct user authorization for spend (deterministic human-in-the-loop).

Money decisions must NOT depend on the stochastic manager agent. Instead the router
calls a :class:`SpendAuthorizer` directly, out of band from the agent, to:

  * ``request_budget`` — prompt the user to enter/save a spend budget when none is set;
  * ``confirm_charge`` — confirm each individual paid-cloud / rented-GPU charge before
    it happens.

This bounds the "stochastic blast zone": the agent can propose work, but every
real-money charge passes through a deterministic gate the *user* controls.

Implementations:
  * :class:`DenyingSpendAuthorizer` — the safe DEFAULT for headless/agent contexts.
    Requests nothing and approves nothing, so paid/rented spend is disabled until a
    real user-facing authorizer is wired.
  * :class:`ConsoleSpendAuthorizer` — prompts on the controlling terminal (CLI use).
  * :class:`CallbackSpendAuthorizer` — delegates to caller-supplied functions so a
    deployment can wire any native UI (desktop notification, web modal, push).
  * :class:`AutoApproveSpendAuthorizer` — approves everything; for trusted automation
    and tests ONLY (documented risk).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ChargeRequest:
    """A specific paid/rented charge awaiting the user's confirmation."""

    provider: str
    kind: str  # "paid_cloud" | "rented_gpu"
    estimated_cost_usd: float
    task_summary: str
    period: Optional[str] = None
    budget_amount_usd: Optional[float] = None
    remaining_usd: Optional[float] = None

    def describe(self) -> str:
        rem = f", {self.remaining_usd:.2f} left this {self.period}" if self.remaining_usd is not None else ""
        return (
            f"Approve ~${self.estimated_cost_usd:.4f} to {self.provider} ({self.kind}){rem} "
            f"for: {self.task_summary!r}"
        )


class SpendAuthorizer(abc.ABC):
    @abc.abstractmethod
    def request_budget(self, suggested_period: str = "monthly") -> Optional[dict]:
        """Prompt the user to set a budget. Return {'amount_usd', 'period'} or None."""

    @abc.abstractmethod
    def confirm_charge(self, request: ChargeRequest) -> bool:
        """Prompt the user to approve a specific charge. Return True to allow."""


class DenyingSpendAuthorizer(SpendAuthorizer):
    """Fail-closed default: no direct user channel wired, so no real-money spend."""

    def request_budget(self, suggested_period: str = "monthly") -> Optional[dict]:
        return None

    def confirm_charge(self, request: ChargeRequest) -> bool:
        return False


class AutoApproveSpendAuthorizer(SpendAuthorizer):
    """Approves everything. Trusted automation / tests ONLY — bypasses the human gate."""

    def __init__(self, amount_usd: float = 100.0, period: str = "monthly"):
        self._amount, self._period = amount_usd, period

    def request_budget(self, suggested_period: str = "monthly") -> Optional[dict]:
        return {"amount_usd": self._amount, "period": suggested_period or self._period}

    def confirm_charge(self, request: ChargeRequest) -> bool:
        return True


class CallbackSpendAuthorizer(SpendAuthorizer):
    """Wire a deployment's own user-facing prompt (native UI, push, webhook, …)."""

    def __init__(
        self,
        confirm_fn: Callable[[ChargeRequest], bool],
        request_budget_fn: Optional[Callable[[str], Optional[dict]]] = None,
    ):
        self._confirm = confirm_fn
        self._request = request_budget_fn

    def request_budget(self, suggested_period: str = "monthly") -> Optional[dict]:
        return self._request(suggested_period) if self._request else None

    def confirm_charge(self, request: ChargeRequest) -> bool:
        return bool(self._confirm(request))


class ConsoleSpendAuthorizer(SpendAuthorizer):
    """Prompts on stdin/stdout. Only use when attached to an interactive terminal
    (NOT for an MCP stdio server, whose stdin carries the protocol)."""

    def __init__(self, input_fn=input, output_fn=print):
        self._input = input_fn
        self._output = output_fn

    def request_budget(self, suggested_period: str = "monthly") -> Optional[dict]:
        self._output("A spend budget is required before using paid cloud or rented GPUs.")
        raw = self._input(f"Enter budget amount in USD (blank to decline): ").strip()
        if not raw:
            return None
        try:
            amount = float(raw)
        except ValueError:
            self._output("Not a number; declining.")
            return None
        if amount <= 0:
            return None
        period = self._input(f"Period [daily/weekly/monthly] (default {suggested_period}): ").strip()
        return {"amount_usd": amount, "period": period or suggested_period}

    def confirm_charge(self, request: ChargeRequest) -> bool:
        ans = self._input(f"{request.describe()} [y/N]: ").strip().lower()
        return ans in ("y", "yes")
