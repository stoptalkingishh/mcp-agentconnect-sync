"""Top-level runtime interfaces.

The eventual runtime will use LangChain + LangGraph under the hood, but the repo
currently only needs a stable contract for where that code will live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agentconnect.common.schemas import TaskSubmission, WorkerResult


class RuntimeNotImplementedError(NotImplementedError):
    """Raised by the skeleton runtime until the worker loop is implemented."""


@dataclass(frozen=True)
class RuntimeConfig:
    workspace_root: str = ""
    max_steps: int = 12
    allow_shell: bool = True
    allow_browser: bool = False
    agent_profile: str = "resident_ok"


class AgentRuntime(Protocol):
    def run(self, task: TaskSubmission) -> WorkerResult:
        """Execute a task and return the worker contract."""


class LangGraphAgentRuntime:
    """Placeholder runtime implementation.

    The real implementation will bind the LangChain/LangGraph loop here.
    """

    def __init__(self, config: RuntimeConfig | None = None):
        self.config = config or RuntimeConfig()

    def run(self, task: TaskSubmission) -> WorkerResult:
        raise RuntimeNotImplementedError(
            "LangChain/LangGraph runtime is not implemented yet."
        )
