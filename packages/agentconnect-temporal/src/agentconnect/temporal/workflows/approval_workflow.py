"""ApprovalWorkflow — a durable wait on a human.

Standalone counterpart to the approval branch inside `SubtaskWorkflow`, for
approvals that outlive (or exist without) a specific subtask run. Expiry is a
recorded decision, not a silent drop.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Optional

from temporalio import workflow
from temporalio.common import RetryPolicy

SHORT = timedelta(seconds=30)
DEFAULT_TIMEOUT_SECONDS = 3 * 24 * 3600
RETRY = RetryPolicy(maximum_attempts=5)


@workflow.defn(name="ApprovalWorkflow")
class ApprovalWorkflow:
    def __init__(self) -> None:
        self._decision: Optional[str] = None  # granted | denied | expired
        self._decided_by: Optional[str] = None
        self._max_cost_usd: Optional[float] = None
        self._reason = ""

    @workflow.query
    def status(self) -> str:
        return self._decision or "pending"

    @workflow.signal
    def approval_granted(self, payload: dict[str, Any]) -> None:
        self._decision = "granted"
        self._decided_by = payload.get("approved_by")
        self._max_cost_usd = payload.get("max_cost_usd")

    @workflow.signal
    def approval_denied(self, payload: dict[str, Any]) -> None:
        self._decision = "denied"
        self._decided_by = payload.get("denied_by")
        self._reason = str(payload.get("reason", ""))

    @workflow.signal
    def approval_expired(self) -> None:
        self._decision = "expired"

    @workflow.run
    async def run(self, approval_id: str, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> dict:
        approval = await workflow.execute_activity(
            "load_approval", approval_id, start_to_close_timeout=SHORT, retry_policy=RETRY,
        )
        await workflow.execute_activity(
            "update_linear",
            args=[approval.get("task_id", ""), "approval_request", approval.get("subtask_id", "")],
            start_to_close_timeout=SHORT, retry_policy=RetryPolicy(maximum_attempts=2),
        )
        try:
            await workflow.wait_condition(
                lambda: self._decision is not None,
                timeout=timedelta(seconds=timeout_seconds),
            )
        except TimeoutError:
            self._decision = "expired"

        await workflow.execute_activity(
            "record_approval_decision",
            args=[approval_id, self._decision, self._decided_by, self._max_cost_usd, self._reason],
            start_to_close_timeout=SHORT, retry_policy=RETRY,
        )
        return {"approval_id": approval_id, "status": self._decision}
