"""Blocking approval queue + a web-backed spend authorizer (reference host).

This is the deployment-side of the deterministic money gate. A
:class:`WebApprovalAuthorizer` implements :class:`SpendAuthorizer` by putting each
pending decision on a thread-safe :class:`ApprovalQueue` and BLOCKING until a user
resolves it out of band (a browser / phone hitting the approval endpoint — see
``router/approval_web.py``) or a timeout elapses. On timeout it fails closed
(deny / no budget).

Deliberately stdlib-only (threading + urllib) so the core stays free of web
frameworks — any transport can drive the same queue; the shipped one is HTTP.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .authorization import ChargeRequest, SpendAuthorizer

_log = logging.getLogger(__name__)


@dataclass
class PendingApproval:
    id: str
    kind: str  # "charge" | "budget"
    text: str  # human-readable description
    payload: dict[str, Any]  # charge fields, or {"suggested_period": ...}
    created_at: float
    _event: threading.Event = field(default_factory=threading.Event, repr=False)
    _result: Any = field(default=None, repr=False)

    def to_public(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "text": self.text, **self.payload}


class ApprovalQueue:
    """Thread-safe registry of pending approvals. The authorizer waits on an item's
    event; the transport (web endpoint) resolves it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, PendingApproval] = {}

    def submit(self, kind: str, text: str, payload: dict[str, Any], now: float) -> PendingApproval:
        item = PendingApproval(
            id=f"appr_{uuid.uuid4().hex[:12]}", kind=kind, text=text, payload=payload, created_at=now
        )
        with self._lock:
            self._items[item.id] = item
        return item

    def get(self, approval_id: str) -> Optional[PendingApproval]:
        with self._lock:
            return self._items.get(approval_id)

    def pending(self) -> list[dict[str, Any]]:
        with self._lock:
            return [i.to_public() for i in self._items.values() if not i._event.is_set()]

    def resolve(self, approval_id: str, result: Any) -> bool:
        """Set an item's result and wake its waiter. Returns False if unknown/done."""
        with self._lock:
            item = self._items.get(approval_id)
            if item is None or item._event.is_set():
                return False
            item._result = result
            item._event.set()
            return True

    def _wait(self, item: PendingApproval, timeout: float) -> tuple[bool, Any]:
        done = item._event.wait(timeout=timeout)
        with self._lock:
            self._items.pop(item.id, None)  # one-shot
        return done, (item._result if done else None)


class WebApprovalAuthorizer(SpendAuthorizer):
    def __init__(
        self,
        queue: ApprovalQueue,
        base_url: str = "http://127.0.0.1:8770",
        webhook_url: Optional[str] = None,
        timeout_seconds: float = 300.0,
        clock=None,
    ):
        self._q = queue
        self._base = base_url.rstrip("/")
        self._webhook = webhook_url
        self._timeout = timeout_seconds
        # Injectable clock keeps created_at deterministic in tests if desired.
        import time as _time

        self._now = clock or _time.time

    # ----------------------------------------------------------- notify
    def _notify(self, item: PendingApproval) -> None:
        approve_url = f"{self._base}/#/{item.id}"
        _log.warning("SPEND APPROVAL NEEDED [%s] %s — approve at %s", item.kind, item.text, approve_url)
        if not self._webhook:
            return

        def _post() -> None:
            body = json.dumps(
                {
                    "id": item.id, "kind": item.kind, "text": item.text,
                    "dashboard_url": f"{self._base}/",
                    "approve_url": f"{self._base}/api/charges/{item.id}/approve",
                    "deny_url": f"{self._base}/api/charges/{item.id}/deny",
                }
            ).encode()
            try:
                req = urllib.request.Request(
                    self._webhook, data=body, headers={"Content-Type": "application/json"}
                )
                urllib.request.urlopen(req, timeout=10)  # best effort
            except Exception as exc:  # noqa: BLE001 — notification must never break routing
                _log.info("approval webhook POST failed: %s", exc)

        threading.Thread(target=_post, daemon=True).start()

    # ----------------------------------------------------- SpendAuthorizer
    def confirm_charge(self, request: ChargeRequest) -> bool:
        payload = {
            "provider": request.provider, "kind_of_spend": request.kind,
            "estimated_cost_usd": request.estimated_cost_usd, "period": request.period,
            "budget_amount_usd": request.budget_amount_usd, "remaining_usd": request.remaining_usd,
        }
        item = self._q.submit("charge", request.describe(), payload, self._now())
        self._notify(item)
        done, result = self._q._wait(item, self._timeout)
        if not done:
            _log.warning("spend approval timed out for %s — denying (fail-closed).", request.provider)
            return False
        return bool(result)

    def request_budget(self, suggested_period: str = "monthly") -> Optional[dict]:
        text = f"Set a spend budget (suggested period: {suggested_period}) to enable paid/rented."
        item = self._q.submit("budget", text, {"suggested_period": suggested_period}, self._now())
        self._notify(item)
        done, result = self._q._wait(item, self._timeout)
        if not done or not result:
            return None
        # result is {"amount_usd", "period"} from the endpoint.
        try:
            amount = float(result["amount_usd"])
        except (KeyError, TypeError, ValueError):
            return None
        if amount <= 0:
            return None
        return {"amount_usd": amount, "period": result.get("period", suggested_period)}
