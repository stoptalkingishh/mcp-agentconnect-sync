"""Top-level runtime entrypoint: LangGraph worker loop behind the router.

The runtime depends on a *model source* — anything with
``generate(GenerateRequest) -> GenerateResponse``. The model-manager backends
(`StubBackend`, real vLLM/llama.cpp), `ResidencyManager`, and the router's
`LocalClient` implementations all satisfy it, so the same loop runs against an
in-process stub in tests and a real serving backend in production.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agentconnect.common.schemas import GenerateRequest, GenerateResponse, TaskSubmission, WorkerResult

from .results import worker_result_from_state
from .state import RuntimeState
from .workspace import Workspace


class ModelSource(Protocol):
    def generate(self, req: GenerateRequest) -> GenerateResponse: ...


@dataclass(frozen=True)
class RuntimeConfig:
    workspace_root: str = ""  # empty -> fresh temp dir per task
    model_id: str = "qwen3.6-35b-a3b"
    max_steps: int = 12
    max_output_tokens: int = 800
    temperature: float = 0.2
    allow_shell: bool = True
    allow_browser: bool = False
    shell_timeout_seconds: float = 60.0
    # Tool output shown to the model is truncated to this many chars.
    observation_max_chars: int = 4000
    agent_profile: str = "resident_ok"


class AgentRuntime(Protocol):
    def run(self, task: TaskSubmission) -> WorkerResult:
        """Execute a task and return the worker contract."""


class LangGraphAgentRuntime:
    """Executes one task at a time through the LangGraph act/tool loop."""

    def __init__(self, model_source: ModelSource, config: RuntimeConfig | None = None):
        self.model_source = model_source
        self.config = config or RuntimeConfig()

    def run(self, task: TaskSubmission, task_id: str = "task_local") -> WorkerResult:
        from .graph import build_execution_graph
        from .prompts import build_system_prompt

        workspace = Workspace.create(self.config.workspace_root or None, task_id=task_id)
        try:
            graph = build_execution_graph(self.config, self.model_source, workspace)
            initial: RuntimeState = {
                "task_id": task_id,
                "messages": [
                    {"role": "system", "content": build_system_prompt(task, self.config)},
                    {"role": "user", "content": task.task},
                ],
                "iteration": 0,
                "changed_artifacts": [],
                "evidence_refs": [],
                "risks": [],
            }
            # Each step is an act node plus at most one tool node; headroom covers
            # the finalize node and the entry edge.
            recursion_limit = self.config.max_steps * 2 + 8
            final = graph.invoke(initial, config={"recursion_limit": recursion_limit})
            return worker_result_from_state(final)
        finally:
            # Runtime-created temp workspaces are unreachable by the caller (the
            # contract carries relative paths only), so remove them; a
            # caller-supplied workspace_root is left untouched.
            workspace.cleanup()
