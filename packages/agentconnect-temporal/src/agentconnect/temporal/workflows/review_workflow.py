"""ReviewWorkflow — a durable wait on another manager.

A review is mostly *waiting*, which is exactly what a workflow is good at and
what an in-process backend cannot survive. The timeout does not fail the review;
it marks it as needing attention and notifies the requester. A human deciding to
abandon a review is a decision, not a timer.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Optional

from temporalio import workflow
from temporalio.common import RetryPolicy

SHORT = timedelta(seconds=30)
DEFAULT_TIMEOUT_SECONDS = 7 * 24 * 3600
RETRY = RetryPolicy(maximum_attempts=5)


@workflow.defn(name="ReviewWorkflow")
class ReviewWorkflow:
    def __init__(self) -> None:
        self._status = "open"
        self._assigned_to: Optional[str] = None
        self._artifact_refs: list[str] = []
        self._result_artifact_id: Optional[str] = None
        self._completed = False
        self._cancelled = False

    @workflow.query
    def status(self) -> str:
        return self._status

    @workflow.query
    def assigned_to(self) -> Optional[str]:
        return self._assigned_to

    @workflow.query
    def artifact_refs(self) -> list[str]:
        return self._artifact_refs

    @workflow.signal
    def review_claimed(self, payload: dict[str, Any]) -> None:
        self._status = "claimed"
        self._assigned_to = payload.get("manager_id") or self._assigned_to

    @workflow.signal
    def review_completed(self, payload: dict[str, Any]) -> None:
        self._completed = True
        self._status = str(payload.get("status", "completed"))
        self._result_artifact_id = payload.get("result_artifact_id")

    @workflow.signal
    def cancel_requested(self) -> None:
        self._cancelled = True

    @workflow.run
    async def run(self, review_id: str, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> dict:
        review = await workflow.execute_activity(
            "load_review", review_id, start_to_close_timeout=SHORT, retry_policy=RETRY,
        )
        self._assigned_to = review.get("assigned_to")
        self._artifact_refs = list(review.get("artifact_refs", []))
        self._status = review.get("status", "open")

        await workflow.execute_activity(
            "update_linear", args=[review.get("task_id", ""), "review_request", review_id],
            start_to_close_timeout=SHORT, retry_policy=RetryPolicy(maximum_attempts=2),
        )

        timed_out = False
        try:
            await workflow.wait_condition(
                lambda: self._completed or self._cancelled,
                timeout=timedelta(seconds=timeout_seconds),
            )
        except TimeoutError:
            timed_out = True

        if timed_out:
            self._status = "needs_attention"
            await workflow.execute_activity(
                "notify_manager",
                args=[review.get("requested_by", ""),
                      f"Review {review_id} has gone stale (no result within the window)."],
                start_to_close_timeout=SHORT, retry_policy=RetryPolicy(maximum_attempts=2),
            )
            return {"review_id": review_id, "status": "needs_attention"}

        if self._cancelled:
            self._status = "cancelled"
            return {"review_id": review_id, "status": "cancelled"}

        # The result was written to the ledger by whoever completed it; the
        # workflow only mirrors it outward.
        await workflow.execute_activity(
            "update_linear", args=[review.get("task_id", ""), "review_result", review_id],
            start_to_close_timeout=SHORT, retry_policy=RetryPolicy(maximum_attempts=2),
        )
        return {
            "review_id": review_id, "status": self._status,
            "result_artifact_id": self._result_artifact_id,
        }
