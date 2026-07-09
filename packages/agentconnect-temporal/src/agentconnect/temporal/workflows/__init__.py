"""Deterministic workflow definitions.

`TaskWorkflow`, `ManagerHandoffWorkflow`, and `WorkerPipelineWorkflow` from the
spec are **not implemented yet** — the spec defers the pipeline until
SubtaskWorkflow and ReviewWorkflow are stable, and the other two have no behavior
that the ledger does not already provide synchronously.
"""

from .approval_workflow import ApprovalWorkflow
from .review_workflow import ReviewWorkflow
from .subtask_workflow import SubtaskWorkflow

ALL_WORKFLOWS = [SubtaskWorkflow, ReviewWorkflow, ApprovalWorkflow]

__all__ = ["ALL_WORKFLOWS", "ApprovalWorkflow", "ReviewWorkflow", "SubtaskWorkflow"]
