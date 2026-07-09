"""SubtaskWorkflow — durable execution of one bounded worker subtask.

**Determinism rules, enforced by construction:**

* Activities are referenced by *string name*, never by importing the activity
  module. That keeps `agentconnect.core` (and pydantic, and sqlite) out of the
  workflow sandbox entirely.
* Nothing here touches the clock, the filesystem, the network, a model, or
  Linear. All of that is in activities.
* Every value crossing the boundary is a plain dict or str.

The workflow is the *plan*; the ledger is the *truth*. If this workflow is
replayed after a crash, the activities it re-runs are idempotent (a routed
subtask re-routes to the same answer; a succeeded subtask is not re-run).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Optional

from temporalio import workflow
from temporalio.common import RetryPolicy

SHORT = timedelta(seconds=30)
WORKER_TIMEOUT = timedelta(minutes=30)

#: A worker may have side effects, so retry it sparingly and only on transport-ish
#: failure; `run_worker` itself refuses to re-run a subtask that already finished.
WORKER_RETRY = RetryPolicy(maximum_attempts=3)
#: Pure decisions are cheap and safe to retry hard.
DECISION_RETRY = RetryPolicy(maximum_attempts=5)


@workflow.defn(name="SubtaskWorkflow")
class SubtaskWorkflow:
    def __init__(self) -> None:
        self._status = "starting"
        self._route: dict[str, Any] = {}
        self._wait_reason: Optional[str] = None
        self._approval: Optional[str] = None  # granted | denied
        self._approved_by: Optional[str] = None
        self._max_cost_usd: Optional[float] = None
        self._deny_reason: str = ""
        self._cancelled = False
        self._notes: list[str] = []

    # ------------------------------------------------------------- queries
    @workflow.query
    def status(self) -> str:
        return self._status

    @workflow.query
    def selected_route(self) -> dict[str, Any]:
        return self._route

    @workflow.query
    def current_wait_reason(self) -> Optional[str]:
        return self._wait_reason

    @workflow.query
    def progress(self) -> dict[str, Any]:
        return {
            "status": self._status,
            "wait_reason": self._wait_reason,
            "selected_worker": self._route.get("selected_worker"),
            "notes": len(self._notes),
        }

    # ------------------------------------------------------------- signals
    @workflow.signal
    def approval_granted(self, payload: dict[str, Any]) -> None:
        self._approval = "granted"
        self._approved_by = payload.get("approved_by")
        self._max_cost_usd = payload.get("max_cost_usd")

    @workflow.signal
    def approval_denied(self, payload: dict[str, Any]) -> None:
        self._approval = "denied"
        self._deny_reason = str(payload.get("reason", ""))

    @workflow.signal
    def cancel_requested(self) -> None:
        self._cancelled = True

    @workflow.signal
    def manager_note_added(self, note: str) -> None:
        self._notes.append(note)

    # ----------------------------------------------------------------- run
    @workflow.run
    async def run(self, subtask_id: str) -> dict[str, Any]:
        self._status = "routing"
        self._route = await workflow.execute_activity(
            "route_subtask", subtask_id,
            start_to_close_timeout=SHORT, retry_policy=DECISION_RETRY,
        )

        if self._route.get("needs_approval"):
            self._status = "needs_approval"
            self._wait_reason = (
                f"awaiting human approval for {self._route.get('approval_candidate')} "
                f"(~${self._route.get('estimated_cost_usd', 0):.4f})"
            )
            await workflow.execute_activity(
                "request_approval", subtask_id,
                start_to_close_timeout=SHORT, retry_policy=DECISION_RETRY,
            )
            await workflow.wait_condition(
                lambda: self._approval is not None or self._cancelled
            )
            self._wait_reason = None

            if self._cancelled:
                return await self._cancel(subtask_id)
            if self._approval == "denied":
                self._status = "denied"
                return {"subtask_id": subtask_id, "status": "failed",
                        "reason": self._deny_reason or "approval denied"}

            # The grant is already durable in the ledger (the API recorded it
            # before signalling). Re-route: the registry may have changed, and the
            # approval may carry a different cost ceiling than the one requested.
            self._status = "routing"
            self._route = await workflow.execute_activity(
                "route_subtask", subtask_id,
                start_to_close_timeout=SHORT, retry_policy=DECISION_RETRY,
            )
            if not self._route.get("selected_worker"):
                self._status = "failed"
                return {"subtask_id": subtask_id, "status": "failed",
                        "reason": "no eligible worker after approval"}

        if self._cancelled:
            return await self._cancel(subtask_id)

        if not self._route.get("selected_worker"):
            self._status = "failed"
            return {"subtask_id": subtask_id, "status": "failed",
                    "reason": "no eligible worker"}

        self._status = "running"
        result = await workflow.execute_activity(
            "run_worker", subtask_id,
            start_to_close_timeout=WORKER_TIMEOUT, retry_policy=WORKER_RETRY,
        )
        self._status = result.get("status", "unknown")

        # Mirroring to Linear is best effort: the ledger already has the truth,
        # and a tracker outage must not fail the subtask.
        try:
            await workflow.execute_activity(
                "update_linear", args=[result.get("task_id", ""), "subtask", subtask_id],
                start_to_close_timeout=SHORT, retry_policy=RetryPolicy(maximum_attempts=2),
            )
        except Exception:  # noqa: BLE001 - deliberate: mirror failure is not task failure
            workflow.logger.warning("Linear mirror failed for subtask %s", subtask_id)

        return result

    async def _cancel(self, subtask_id: str) -> dict[str, Any]:
        self._status = "cancelled"
        await workflow.execute_activity(
            "cancel_subtask", subtask_id,
            start_to_close_timeout=SHORT, retry_policy=DECISION_RETRY,
        )
        return {"subtask_id": subtask_id, "status": "cancelled"}
