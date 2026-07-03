"""Agent runtime for AgentConnect: the worker layer behind the router.

A LangGraph act/tool loop executes one task inside a confined workspace using
filesystem + shell tools, then returns the shared ``WorkerResult`` contract.
The model is reached through the ``ModelSource`` protocol, satisfied by the
model-manager backends and the router's local clients alike.
"""

from .actions import Action, parse_action
from .agent import AgentRuntime, LangGraphAgentRuntime, ModelSource, RuntimeConfig
from .results import worker_result_from_state
from .state import RuntimeState
from .transport import HttpAgentRuntime, RuntimeEndpoint, add_pull_routes, create_worker_app
from .workspace import Workspace, WorkspaceError

__all__ = [
    "Action",
    "AgentRuntime",
    "HttpAgentRuntime",
    "LangGraphAgentRuntime",
    "ModelSource",
    "RuntimeConfig",
    "RuntimeEndpoint",
    "RuntimeState",
    "Workspace",
    "WorkspaceError",
    "add_pull_routes",
    "create_worker_app",
    "parse_action",
    "worker_result_from_state",
]
