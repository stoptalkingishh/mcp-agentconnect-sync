"""Execution backends (Temporal spec, "Core execution abstraction").

The backplane decides *what* work exists; a backend decides *how it runs*. Two
implementations:

* :class:`DirectExecutionBackend` — runs everything inline, in-process. Tests and
  local smoke flows only. It has no durability: kill the process mid-worker and
  the subtask is stranded in ``running``.
* ``TemporalExecutionBackend`` (in ``agentconnect-temporal``) — starts a
  workflow and returns. Retries, approval waits, timers, and crash recovery are
  Temporal's problem.

The service never imports Temporal. It calls this interface, and the *adapters*
choose which backend to bind. That is what keeps `agentconnect-core` installable
without a workflow server (§8: zero-infra default).
"""

from __future__ import annotations

import abc
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field

from .models import now

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from .service import AgentConnectService


class ExecutionState(str, Enum):
    running = "running"
    waiting_approval = "waiting_approval"
    waiting_review = "waiting_review"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"
    unknown = "unknown"


class ExecutionHandle(BaseModel):
    """A durable pointer to an execution. For Temporal, ``handle_id`` *is* the
    workflow id — so `/workflows/{workflow_id}` resolves without a second table
    lookup, and a human can find the run in the Temporal UI from the ledger."""

    handle_id: str
    backend: str
    entity_type: str  # subtask | review | approval
    entity_id: str
    workflow_id: Optional[str] = None
    run_id: Optional[str] = None
    state: ExecutionState = ExecutionState.running
    created_at: float = Field(default_factory=now)
    updated_at: float = Field(default_factory=now)
    detail: str = ""


class ExecutionStatus(BaseModel):
    handle_id: str
    state: ExecutionState
    detail: str = ""
    wait_reason: Optional[str] = None
    result: Optional[dict[str, Any]] = None


class ExecutionBackend(abc.ABC):
    """`signal` is the only way an outside event (a Linear approval, an API call)
    reaches running work. Adapters never poke the workflow directly."""

    name = "abstract"

    @abc.abstractmethod
    def start_subtask(self, subtask_id: str) -> ExecutionHandle: ...

    @abc.abstractmethod
    def start_review(self, review_id: str) -> ExecutionHandle: ...

    @abc.abstractmethod
    def start_approval(self, approval_id: str) -> ExecutionHandle: ...

    @abc.abstractmethod
    def get_status(self, handle_id: str) -> ExecutionStatus: ...

    @abc.abstractmethod
    def cancel(self, handle_id: str) -> None: ...

    @abc.abstractmethod
    def signal(self, handle_id: str, name: str, payload: dict[str, Any]) -> None: ...


class DirectExecutionBackend(ExecutionBackend):
    """Inline execution. Not durable — for tests and single-shot local runs.

    Every method does synchronously what a workflow would do asynchronously, so
    the same service code path is exercised by both backends. A subtask that
    needs approval simply *stops*; the approval signal resumes it.
    """

    name = "direct"

    def __init__(self, service: "AgentConnectService") -> None:
        self.service = service

    def _handle(self, entity_type: str, entity_id: str, state: ExecutionState,
                detail: str = "") -> ExecutionHandle:
        handle = ExecutionHandle(
            handle_id=f"direct-{entity_type}-{entity_id}", backend=self.name,
            entity_type=entity_type, entity_id=entity_id, workflow_id=None,
            state=state, detail=detail, created_at=self.service._now(),
            updated_at=self.service._now(),
        )
        return self.service.put_execution(handle)

    # ---------------------------------------------------------------- start
    def start_subtask(self, subtask_id: str) -> ExecutionHandle:
        explanation = self.service.route_subtask(subtask_id)
        if explanation.needs_approval:
            return self._handle(
                "subtask", subtask_id, ExecutionState.waiting_approval,
                f"awaiting approval for {explanation.approval_candidate}",
            )
        if explanation.selected_worker is None:
            return self._handle("subtask", subtask_id, ExecutionState.failed,
                                "no eligible worker")
        subtask = self.service.run_subtask(subtask_id)
        state = (
            ExecutionState.completed if subtask.status.value == "succeeded"
            else ExecutionState.failed
        )
        return self._handle("subtask", subtask_id, state)

    def start_review(self, review_id: str) -> ExecutionHandle:
        # A review is inherently a wait on another manager; there is nothing to
        # run inline. The handle exists so the ledger can show it as pending.
        return self._handle("review", review_id, ExecutionState.waiting_review,
                            "awaiting reviewer")

    def start_approval(self, approval_id: str) -> ExecutionHandle:
        return self._handle("approval", approval_id, ExecutionState.waiting_approval,
                            "awaiting human decision")

    # --------------------------------------------------------------- inspect
    def get_status(self, handle_id: str) -> ExecutionStatus:
        handle = self.service.get_execution(handle_id)
        if handle is None:
            return ExecutionStatus(handle_id=handle_id, state=ExecutionState.unknown)
        return ExecutionStatus(
            handle_id=handle_id, state=handle.state, detail=handle.detail,
            wait_reason=handle.detail if handle.state.value.startswith("waiting") else None,
        )

    def cancel(self, handle_id: str) -> None:
        handle = self.service.get_execution(handle_id)
        if handle is None:
            return
        if handle.entity_type == "subtask":
            subtask = self.service.storage.get_subtask(handle.entity_id)
            # `cancel_subtask` calls back here after marking the ledger terminal;
            # re-entering it would recurse. Terminal means there is nothing to do.
            if subtask is not None and subtask.status.value not in (
                "succeeded", "failed", "cancelled"
            ):
                self.service.cancel_subtask(handle.entity_id)
                return
        self.service.update_execution(handle_id, state=ExecutionState.cancelled)

    # --------------------------------------------------------------- signal
    def signal(self, handle_id: str, name: str, payload: dict[str, Any]) -> None:
        handle = self.service.get_execution(handle_id)
        if handle is None or handle.entity_type != "subtask":
            return
        subtask_id = handle.entity_id
        if name == "approval_granted":
            # The grant is already recorded; resume exactly where the workflow
            # would: re-route (the registry may have changed), then run.
            explanation = self.service.route_subtask(subtask_id)
            if explanation.selected_worker is None:
                self.service.update_execution(
                    handle_id, state=ExecutionState.failed,
                    detail="no eligible worker after approval",
                )
                return
            subtask = self.service.run_subtask(subtask_id)
            self.service.update_execution(
                handle_id,
                state=(ExecutionState.completed if subtask.status.value == "succeeded"
                       else ExecutionState.failed),
            )
        elif name == "approval_denied":
            self.service.update_execution(handle_id, state=ExecutionState.failed,
                                          detail="approval denied")
        elif name == "cancel_requested":
            self.cancel(handle_id)
