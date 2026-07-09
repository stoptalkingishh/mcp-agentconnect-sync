"""Temporal execution engine for the AgentConnect backplane.

Temporal makes execution durable; AgentConnect makes work state durable. This
package never becomes a source of truth — workflows carry artifact *ids*, never
artifact bodies, and every durable fact is written through `AgentConnectService`.

NOT the same thing as the separate `agentconnect-temporal` repo, which wraps the
router-era `AgentRuntime` protocol in a Temporal substrate. Different layer.
"""

from .activities import BackplaneActivities
from .client import (
    DEFAULT_TASK_QUEUE,
    TemporalExecutionBackend,
    build_worker,
    connect,
    task_queue_from_env,
)
from .workflows import ALL_WORKFLOWS, ApprovalWorkflow, ReviewWorkflow, SubtaskWorkflow

__all__ = [
    "ALL_WORKFLOWS", "ApprovalWorkflow", "BackplaneActivities", "DEFAULT_TASK_QUEUE",
    "ReviewWorkflow", "SubtaskWorkflow", "TemporalExecutionBackend", "build_worker",
    "connect", "task_queue_from_env",
]
