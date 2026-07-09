"""Temporal client, worker factory, and `TemporalExecutionBackend`.

The service is synchronous and the Temporal SDK is not, so the backend owns a
private event loop on a background thread and submits coroutines to it. That is
deliberate: making `AgentConnectService` async would push async all the way up
into the CLI and the MCP tool bodies for no benefit — the ledger is SQLite.

Workflow IDs are derived from the entity id (`subtask-<id>`), so:

* the handle id *is* the workflow id — `/workflows/{workflow_id}` resolves with
  no extra table, and a human can paste it into the Temporal UI;
* starting the same subtask twice is a no-op rather than two runs — we reuse the
  running workflow instead of racing it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from agentconnect.core.execution import (
    ExecutionBackend,
    ExecutionHandle,
    ExecutionState,
    ExecutionStatus,
)
from agentconnect.core.service import AgentConnectService

from .activities import BackplaneActivities
from .workflows import ALL_WORKFLOWS, ApprovalWorkflow, ReviewWorkflow, SubtaskWorkflow

_log = logging.getLogger(__name__)

DEFAULT_TASK_QUEUE = "agentconnect"
DEFAULT_ADDRESS = "localhost:7233"
DEFAULT_NAMESPACE = "default"

#: Temporal's own terminal statuses, mapped onto ours.
_STATUS_MAP = {
    1: ExecutionState.running,     # RUNNING
    2: ExecutionState.completed,   # COMPLETED
    3: ExecutionState.failed,      # FAILED
    4: ExecutionState.cancelled,   # CANCELED
    5: ExecutionState.failed,      # TERMINATED
    6: ExecutionState.running,     # CONTINUED_AS_NEW
    7: ExecutionState.failed,      # TIMED_OUT
}


def task_queue_from_env() -> str:
    return os.environ.get("AGENTCONNECT_TEMPORAL_TASK_QUEUE", DEFAULT_TASK_QUEUE)


async def connect(
    address: Optional[str] = None, namespace: Optional[str] = None
):
    from temporalio.client import Client

    return await Client.connect(
        address or os.environ.get("TEMPORAL_ADDRESS", DEFAULT_ADDRESS),
        namespace=namespace or os.environ.get("TEMPORAL_NAMESPACE", DEFAULT_NAMESPACE),
    )


def build_worker(
    client: Any,
    service: AgentConnectService,
    linear_sync: Optional[Any] = None,
    task_queue: Optional[str] = None,
    max_concurrent_activities: int = 8,
):
    """A Temporal worker serving the backplane's workflows and activities.

    Activities are synchronous under the hood (SQLite, worker harnesses), so they
    run in a thread pool; workflows stay on the event loop where they belong.
    """
    from temporalio.worker import Worker

    activities = BackplaneActivities(service, linear_sync)
    return Worker(
        client,
        task_queue=task_queue or task_queue_from_env(),
        workflows=ALL_WORKFLOWS,
        activities=activities.all(),
        activity_executor=ThreadPoolExecutor(max_workers=max_concurrent_activities),
    )


class _LoopThread:
    """A private asyncio loop so a sync caller can await Temporal coroutines."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, name="agentconnect-temporal", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro, timeout: float = 30.0):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)


class TemporalExecutionBackend(ExecutionBackend):
    name = "temporal"

    def __init__(
        self,
        client: Any,
        service: AgentConnectService,
        task_queue: Optional[str] = None,
        loop: Optional[_LoopThread] = None,
        timeout: float = 30.0,
    ) -> None:
        self.client = client
        self.service = service
        self.task_queue = task_queue or task_queue_from_env()
        self._loop = loop or _LoopThread()
        self._timeout = timeout

    # ---------------------------------------------------------------- utils
    def _run(self, coro):
        return self._loop.submit(coro, self._timeout)

    def _record(self, entity_type: str, entity_id: str, workflow_id: str, run_id: str,
                state: ExecutionState, detail: str = "") -> ExecutionHandle:
        now = self.service._now()
        return self.service.put_execution(ExecutionHandle(
            handle_id=workflow_id, backend=self.name, entity_type=entity_type,
            entity_id=entity_id, workflow_id=workflow_id, run_id=run_id, state=state,
            created_at=now, updated_at=now, detail=detail,
        ))

    def _start(self, workflow, arg: str, entity_type: str, entity_id: str,
               detail: str) -> ExecutionHandle:
        workflow_id = f"{entity_type}-{entity_id}"
        existing = self.service.get_execution(workflow_id)
        if existing is not None and existing.state in (
            ExecutionState.running, ExecutionState.waiting_approval,
            ExecutionState.waiting_review,
        ):
            return existing  # already in flight; do not race it
        handle = self._run(self.client.start_workflow(
            workflow, arg, id=workflow_id, task_queue=self.task_queue,
        ))
        return self._record(
            entity_type, entity_id, workflow_id, handle.result_run_id or "",
            ExecutionState.running, detail,
        )

    # ---------------------------------------------------------------- start
    def start_subtask(self, subtask_id: str) -> ExecutionHandle:
        return self._start(SubtaskWorkflow.run, subtask_id, "subtask", subtask_id,
                           "SubtaskWorkflow started")

    def start_review(self, review_id: str) -> ExecutionHandle:
        return self._start(ReviewWorkflow.run, review_id, "review", review_id,
                           "ReviewWorkflow started")

    def start_approval(self, approval_id: str) -> ExecutionHandle:
        return self._start(ApprovalWorkflow.run, approval_id, "approval", approval_id,
                           "ApprovalWorkflow started")

    # -------------------------------------------------------------- inspect
    def get_status(self, handle_id: str) -> ExecutionStatus:
        stored = self.service.get_execution(handle_id)
        workflow_id = (stored.workflow_id if stored else None) or handle_id
        try:
            handle = self.client.get_workflow_handle(workflow_id)
            described = self._run(handle.describe())
        except Exception as exc:
            # A workflow server outage must not take the API down; report what
            # the ledger last knew.
            _log.warning("temporal describe(%s) failed: %s", workflow_id, exc)
            return ExecutionStatus(
                handle_id=workflow_id,
                state=stored.state if stored else ExecutionState.unknown,
                detail=f"temporal unreachable: {exc}",
            )
        raw = getattr(described.status, "value", described.status)
        state = _STATUS_MAP.get(raw, ExecutionState.unknown)
        wait_reason = None
        if state is ExecutionState.running:
            try:
                wait_reason = self._run(handle.query(SubtaskWorkflow.current_wait_reason))
            except Exception:
                wait_reason = None
            if wait_reason:
                state = ExecutionState.waiting_approval
        if stored is not None and stored.state is not state:
            self.service.update_execution(workflow_id, state=state.value)
        return ExecutionStatus(
            handle_id=workflow_id, state=state, wait_reason=wait_reason,
        )

    def cancel(self, handle_id: str) -> None:
        stored = self.service.get_execution(handle_id)
        workflow_id = (stored.workflow_id if stored else None) or handle_id
        handle = self.client.get_workflow_handle(workflow_id)
        try:
            self._run(handle.signal("cancel_requested"))
        except Exception as exc:
            _log.warning("temporal cancel signal(%s) failed: %s", workflow_id, exc)

    def signal(self, handle_id: str, name: str, payload: dict[str, Any]) -> None:
        stored = self.service.get_execution(handle_id)
        workflow_id = (stored.workflow_id if stored else None) or handle_id
        handle = self.client.get_workflow_handle(workflow_id)
        args = [] if name == "cancel_requested" else [payload]
        try:
            self._run(handle.signal(name, *args))
        except Exception as exc:
            # The grant/denial is already durable in the ledger; a missed signal
            # means the workflow resumes on its next poll or is retried by hand.
            _log.warning("temporal signal %s(%s) failed: %s", name, workflow_id, exc)
