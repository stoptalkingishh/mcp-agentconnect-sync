"""Agent runtime skeleton for AgentConnect.

This package is the worker layer that will execute tasks after the router has
selected a backend. It is intentionally minimal for now: the repo documents the
runtime boundary, but the actual LangChain/LangGraph implementation comes later.
"""

from .agent import AgentRuntime, RuntimeConfig, RuntimeNotImplementedError
from .results import RuntimeResult, build_worker_result
from .state import RuntimeState

__all__ = [
    "AgentRuntime",
    "RuntimeConfig",
    "RuntimeNotImplementedError",
    "RuntimeResult",
    "RuntimeState",
    "build_worker_result",
]
