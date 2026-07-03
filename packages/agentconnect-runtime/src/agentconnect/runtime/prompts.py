"""Prompt assembly helpers for the worker runtime."""

from __future__ import annotations

from agentconnect.common.schemas import TaskSubmission

from .agent import RuntimeConfig


def build_system_prompt(task: TaskSubmission, config: RuntimeConfig) -> str:
    """Build the system prompt for the task.

    The first implementation will likely expand this into role instructions,
    tool guidance, and output contracts.
    """
    return (
        "You are an agent runtime executing a task under router policy. "
        f"Profile={config.agent_profile}. "
        f"Task={task.task}"
    )
