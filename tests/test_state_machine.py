import pytest

from agentconnect.common.schemas import TaskState
from agentconnect.common.state import (
    IllegalTransition,
    TaskFSM,
    assert_transition,
    can_transition,
)


def test_happy_path_is_legal():
    fsm = TaskFSM()
    path = [
        TaskState.CLASSIFIED,
        TaskState.PRIVACY_CHECKED,
        TaskState.ELIGIBLE_PROVIDERS_COMPUTED,
        TaskState.QUEUED,
        TaskState.DISPATCHED,
        TaskState.RUNNING,
        TaskState.ARTIFACTS_WRITTEN,
        TaskState.CHECKS_RUN,
        TaskState.REVIEW_READY,
        TaskState.APPROVED,
        TaskState.COMPLETE,
    ]
    for s in path:
        fsm.to(s)
    assert fsm.is_terminal
    assert fsm.state == TaskState.COMPLETE


def test_illegal_transition_raises():
    with pytest.raises(IllegalTransition):
        assert_transition(TaskState.CREATED, TaskState.COMPLETE)


def test_cancel_from_any_nonterminal():
    assert can_transition(TaskState.RUNNING, TaskState.CANCELLED)
    assert can_transition(TaskState.QUEUED, TaskState.CANCELLED)
    # terminal states cannot be cancelled
    assert not can_transition(TaskState.COMPLETE, TaskState.CANCELLED)


def test_retry_loops_back_to_queued():
    assert can_transition(TaskState.REVIEW_READY, TaskState.RETRY)
    assert can_transition(TaskState.RETRY, TaskState.QUEUED)
