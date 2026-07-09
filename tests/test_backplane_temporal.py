"""Temporal workflows against an in-process time-skipping server.

Skipped (never failed) when temporalio is absent or the test server cannot start
— it downloads a binary on first use, and the offline gate must stay green.

What is actually proven here: a subtask runs through real activities; the route
explanation is persisted from inside the workflow; an approval wait blocks until
a signal arrives; a denial ends the run; a cancellation signal is honored; and a
flaky activity is retried rather than failing the workflow.
"""

import asyncio
import uuid

import pytest

pytest.importorskip("temporalio")

from temporalio.client import WorkflowFailureError  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402
from temporalio.worker import Worker  # noqa: E402

from agentconnect.core import (  # noqa: E402
    AgentConnectService,
    CreateTaskRequest,
    EchoWorker,
    PrivacyTier,
    RoutePolicy,
    SubtaskRequest,
    SubtaskStatus,
    WorkerLocation,
)
from agentconnect.core.workers import RawModelWorker  # noqa: E402
from agentconnect.temporal import BackplaneActivities  # noqa: E402
from agentconnect.temporal.workflows import ALL_WORKFLOWS, SubtaskWorkflow  # noqa: E402

TASK_QUEUE = "agentconnect-test"


@pytest.fixture(scope="module")
def env():
    async def _start():
        return await WorkflowEnvironment.start_time_skipping()

    try:
        environment = asyncio.get_event_loop().run_until_complete(_start())
    except Exception as exc:  # no bundled/downloadable test server in this sandbox
        pytest.skip(f"Temporal test server unavailable: {exc}")
    yield environment
    asyncio.get_event_loop().run_until_complete(environment.shutdown())


def make_service(tmp_path, workers=None, policy=None):
    return AgentConnectService.create(
        db_path=":memory:", artifact_dir=str(tmp_path / "artifacts"),
        workers=workers if workers is not None else [EchoWorker()], policy=policy,
    )


def cloud_worker():
    return RawModelWorker(
        "cheap_cloud", lambda p: "cloud output", model="gpt", location=WorkerLocation.cloud,
        privacy_tiers=[PrivacyTier.public], cost_per_1k_tokens_usd=0.5,
    )


def _worker(env, service, activities=None):
    acts = activities if activities is not None else BackplaneActivities(service).all()
    return Worker(
        env.client, task_queue=TASK_QUEUE, workflows=ALL_WORKFLOWS, activities=acts,
        activity_executor=__import__("concurrent.futures", fromlist=["ThreadPoolExecutor"])
        .ThreadPoolExecutor(max_workers=4),
    )


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_subtask_workflow_routes_runs_and_persists_the_explanation(env, tmp_path):
    svc = make_service(tmp_path)
    task = svc.create_task(CreateTaskRequest(title="t", goal="g"))
    # Create the subtask record without running it: the workflow owns execution.
    from agentconnect.core.execution import ExecutionBackend, ExecutionHandle, ExecutionState

    class NullBackend(ExecutionBackend):
        name = "null"

        def start_subtask(self, subtask_id):
            return ExecutionHandle(handle_id=f"null-{subtask_id}", backend="null",
                                   entity_type="subtask", entity_id=subtask_id,
                                   state=ExecutionState.running)

        def start_review(self, review_id): ...
        def start_approval(self, approval_id): ...
        def get_status(self, handle_id): ...
        def cancel(self, handle_id): ...
        def signal(self, handle_id, name, payload): ...

    svc.bind_execution(NullBackend())
    subtask = svc.submit_subtask(task.id, SubtaskRequest(
        title="find dupes", instructions="inspect auth files"))
    assert subtask.status is SubtaskStatus.queued  # nothing ran

    async def go():
        async with _worker(env, svc):
            return await env.client.execute_workflow(
                SubtaskWorkflow.run, subtask.id,
                id=f"subtask-{uuid.uuid4()}", task_queue=TASK_QUEUE,
            )

    result = _run(go())
    assert result["status"] == "succeeded"

    stored = svc.get_subtask(subtask.id).subtask
    assert stored.status is SubtaskStatus.succeeded
    assert stored.assigned_worker == "echo_worker"
    assert stored.result_artifact_id
    assert svc.explain_route(subtask.id).selected_worker == "echo_worker"
    body = svc.read_artifact_chunk(stored.result_artifact_id, 0, 8000).content
    assert "inspect auth files" in body


def test_approval_wait_blocks_until_the_signal_arrives(env, tmp_path):
    svc = make_service(tmp_path, workers=[cloud_worker()], policy=RoutePolicy(max_cost_usd=10.0))
    task = svc.create_task(CreateTaskRequest(title="t"))
    from agentconnect.core.models import Subtask
    from agentconnect.core import ids

    subtask = Subtask(id=ids.new_id(ids.SUBTASK), parent_task_id=task.id, title="t",
                      instructions="i", privacy_tier=PrivacyTier.public)
    svc.storage.insert_subtask(subtask)

    async def go():
        async with _worker(env, svc):
            handle = await env.client.start_workflow(
                SubtaskWorkflow.run, subtask.id,
                id=f"subtask-{uuid.uuid4()}", task_queue=TASK_QUEUE,
            )
            # Wait until the workflow has parked on the human.
            for _ in range(50):
                if await handle.query(SubtaskWorkflow.status) == "needs_approval":
                    break
                await asyncio.sleep(0.05)
            assert await handle.query(SubtaskWorkflow.status) == "needs_approval"
            assert "awaiting human approval" in await handle.query(
                SubtaskWorkflow.current_wait_reason)
            assert svc.get_subtask(subtask.id).subtask.status is SubtaskStatus.needs_approval

            # The human decides. The grant is recorded durably first, then signalled.
            svc.grant_approval(subtask.id, "matthew", 3.0)
            await handle.signal(SubtaskWorkflow.approval_granted,
                                {"approved_by": "matthew", "max_cost_usd": 3.0})
            return await handle.result()

    result = _run(go())
    assert result["status"] == "succeeded"
    assert svc.get_subtask(subtask.id).subtask.assigned_worker == "cheap_cloud"
    assert svc.list_approvals(task.id)[0].status.value == "granted"


def test_denied_approval_ends_the_workflow_without_running_a_worker(env, tmp_path):
    worker = cloud_worker()
    svc = make_service(tmp_path, workers=[worker], policy=RoutePolicy(max_cost_usd=10.0))
    task = svc.create_task(CreateTaskRequest(title="t"))
    from agentconnect.core import ids
    from agentconnect.core.models import Subtask

    subtask = Subtask(id=ids.new_id(ids.SUBTASK), parent_task_id=task.id, title="t",
                      instructions="i", privacy_tier=PrivacyTier.public)
    svc.storage.insert_subtask(subtask)

    async def go():
        async with _worker(env, svc):
            handle = await env.client.start_workflow(
                SubtaskWorkflow.run, subtask.id,
                id=f"subtask-{uuid.uuid4()}", task_queue=TASK_QUEUE,
            )
            for _ in range(50):
                if await handle.query(SubtaskWorkflow.status) == "needs_approval":
                    break
                await asyncio.sleep(0.05)
            await handle.signal(SubtaskWorkflow.approval_denied, {"reason": "too expensive"})
            return await handle.result()

    result = _run(go())
    assert result["status"] == "failed" and result["reason"] == "too expensive"
    assert svc.get_subtask(subtask.id).subtask.status is SubtaskStatus.needs_approval


def test_cancel_signal_stops_a_parked_subtask(env, tmp_path):
    svc = make_service(tmp_path, workers=[cloud_worker()], policy=RoutePolicy(max_cost_usd=10.0))
    task = svc.create_task(CreateTaskRequest(title="t"))
    from agentconnect.core import ids
    from agentconnect.core.models import Subtask

    subtask = Subtask(id=ids.new_id(ids.SUBTASK), parent_task_id=task.id, title="t",
                      instructions="i", privacy_tier=PrivacyTier.public)
    svc.storage.insert_subtask(subtask)

    async def go():
        async with _worker(env, svc):
            handle = await env.client.start_workflow(
                SubtaskWorkflow.run, subtask.id,
                id=f"subtask-{uuid.uuid4()}", task_queue=TASK_QUEUE,
            )
            for _ in range(50):
                if await handle.query(SubtaskWorkflow.status) == "needs_approval":
                    break
                await asyncio.sleep(0.05)
            await handle.signal(SubtaskWorkflow.cancel_requested)
            return await handle.result()

    result = _run(go())
    assert result["status"] == "cancelled"
    assert svc.get_subtask(subtask.id).subtask.status is SubtaskStatus.cancelled


def test_a_flaky_activity_is_retried_rather_than_failing_the_workflow(env, tmp_path):
    svc = make_service(tmp_path)
    task = svc.create_task(CreateTaskRequest(title="t"))
    from agentconnect.core import ids
    from agentconnect.core.models import Subtask

    subtask = Subtask(id=ids.new_id(ids.SUBTASK), parent_task_id=task.id, title="t",
                      instructions="i")
    svc.storage.insert_subtask(subtask)

    activities = BackplaneActivities(svc)
    calls = {"n": 0}
    real_route = activities.route_subtask

    from temporalio import activity

    @activity.defn(name="route_subtask")
    async def flaky_route(subtask_id: str):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient sqlite lock")
        return await real_route(subtask_id)

    acts = [a for a in activities.all() if getattr(a, "__name__", "") != "route_subtask"]
    acts.append(flaky_route)

    async def go():
        async with _worker(env, svc, activities=acts):
            return await env.client.execute_workflow(
                SubtaskWorkflow.run, subtask.id,
                id=f"subtask-{uuid.uuid4()}", task_queue=TASK_QUEUE,
            )

    result = _run(go())
    assert calls["n"] == 2  # retried once, then succeeded
    assert result["status"] == "succeeded"


def test_run_worker_activity_is_idempotent_on_retry(tmp_path):
    """A retried `run_worker` on an already-succeeded subtask must not re-run the
    worker — proven without Temporal, since it is a plain service invariant."""
    svc = make_service(tmp_path)
    task = svc.create_task(CreateTaskRequest(title="t"))
    subtask = svc.submit_subtask(task.id, SubtaskRequest(title="t", instructions="i"))
    assert subtask.status is SubtaskStatus.succeeded

    runs_before = len(svc.get_subtask(subtask.id).runs)
    again = svc.run_subtask(subtask.id)  # what the activity retry would do
    assert again.status is SubtaskStatus.succeeded
    assert len(svc.get_subtask(subtask.id).runs) == runs_before
