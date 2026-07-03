"""LangGraph execution graph placeholder."""

from __future__ import annotations

from typing import Any

from .agent import RuntimeConfig, RuntimeNotImplementedError


def build_execution_graph(config: RuntimeConfig) -> Any:
    """Build the worker graph.

    This is intentionally a stub until the worker implementation is added.
    """
    raise RuntimeNotImplementedError(
        f"Execution graph not implemented yet for profile {config.agent_profile!r}."
    )
