"""Worker adapter interface and the two bootstrap workers (spec §18, §19).

**A worker is not a model.** It is a `{harness, model, tools, sandbox,
privacy_tiers}` tuple. The same harness may be registered twice with different
models; the same model may appear behind two harnesses. Routing selects the
tuple, never the model.

Adapters normalize their harness's output into `WorkerResult`. Anything the
worker wants to be durable it writes through `WorkerContext.create_artifact` —
its internal tool loop, scratch context, and reasoning are explicitly *not*
truth (§6).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from .models import (
    Artifact,
    ArtifactChunk,
    ArtifactType,
    PrivacyTier,
    SandboxSpec,
    Subtask,
    Task,
    WorkerLocation,
)


class WorkerCapabilities(BaseModel):
    worker_id: str
    harness: str
    model: Optional[str] = None
    tools: list[str] = Field(default_factory=list)
    sandbox: SandboxSpec = Field(default_factory=SandboxSpec)
    privacy_tiers: list[PrivacyTier] = Field(default_factory=list)
    capability_tags: list[str] = Field(default_factory=list)
    location: WorkerLocation = WorkerLocation.local
    cost_per_1k_tokens_usd: float = 0.0
    availability: float = 1.0
    #: Cloud and rented compute must be approved by a human before it runs (§21).
    requires_approval: bool = False


class WorkerEstimate(BaseModel):
    estimated_cost_usd: float = 0.0
    estimated_tokens_in: int = 0
    estimated_tokens_out: int = 0
    estimated_seconds: float = 0.0


class WorkerHealth(BaseModel):
    available: bool = True
    detail: str = ""


class WorkerArtifactRef(BaseModel):
    artifact_id: str
    type: ArtifactType = ArtifactType.worker_output
    description: str = ""


class WorkerResult(BaseModel):
    """The normalized result shape from §18. Every harness reduces to this."""

    status: str = "succeeded"  # succeeded | failed
    summary: str = ""
    artifacts: list[WorkerArtifactRef] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: Optional[str] = None


@dataclass
class WorkerContext:
    """Everything a worker is allowed to touch: the task it serves, the subtask
    it was given, and a way to persist output. No storage handle, no service."""

    task: Task
    subtask: Subtask
    run_id: str
    create_artifact: Callable[..., Artifact]
    read_artifact_chunk: Callable[[str, int, int], ArtifactChunk]


class WorkerAdapter(abc.ABC):
    """Base class for every worker harness."""

    @property
    @abc.abstractmethod
    def worker_id(self) -> str: ...

    @abc.abstractmethod
    def capabilities(self) -> WorkerCapabilities: ...

    @abc.abstractmethod
    def run(self, subtask: Subtask, context: WorkerContext) -> WorkerResult: ...

    def estimate(
        self, subtask: Subtask, context: Optional[WorkerContext] = None
    ) -> WorkerEstimate:
        """Cost/latency guess used by routing's budget gate.

        The router calls this *before* a run exists, so ``context`` is ``None``
        there; a harness that needs run state must tolerate that."""
        tokens_in = max(1, len(subtask.instructions) // 4)
        tokens_out = max(1, tokens_in // 4)
        caps = self.capabilities()
        cost = (tokens_in + tokens_out) / 1000.0 * caps.cost_per_1k_tokens_usd
        return WorkerEstimate(
            estimated_cost_usd=round(cost, 6),
            estimated_tokens_in=tokens_in,
            estimated_tokens_out=tokens_out,
        )

    def cancel(self, run_id: str) -> None:
        """Best effort. The ledger marks the run cancelled regardless."""
        return None

    def health(self) -> WorkerHealth:
        return WorkerHealth(available=True)


class EchoWorker(WorkerAdapter):
    """Deterministic, offline, zero-cost worker (§18, §24).

    Exists so the whole backplane — routing, runs, artifacts, handoff — is
    testable and demonstrable with no model, no GPU, and no network. It reads
    nothing and writes one report artifact.
    """

    def __init__(self, worker_id: str = "echo_worker") -> None:
        self._worker_id = worker_id

    @property
    def worker_id(self) -> str:
        return self._worker_id

    def capabilities(self) -> WorkerCapabilities:
        return WorkerCapabilities(
            worker_id=self._worker_id,
            harness="echo",
            model=None,
            tools=["write_artifact"],
            sandbox=SandboxSpec(),  # touches nothing
            privacy_tiers=list(PrivacyTier),
            capability_tags=["echo", "inspect", "summarize"],
            location=WorkerLocation.local,
            cost_per_1k_tokens_usd=0.0,
            availability=1.0,
            requires_approval=False,
        )

    def run(self, subtask: Subtask, context: WorkerContext) -> WorkerResult:
        body = (
            f"# Echo worker report\n\n"
            f"Subtask: {subtask.title}\n"
            f"Privacy tier: {subtask.privacy_tier.value}\n\n"
            f"## Instructions received\n{subtask.instructions}\n"
        )
        artifact = context.create_artifact(
            type=ArtifactType.worker_output,
            content=body,
            summary=f"Echo of subtask {subtask.id}",
        )
        return WorkerResult(
            status="succeeded",
            summary=f"Echoed instructions for {subtask.title!r}.",
            artifacts=[
                WorkerArtifactRef(
                    artifact_id=artifact.id,
                    type=ArtifactType.worker_output,
                    description="Verbatim echo of the subtask instructions",
                )
            ],
            metrics={
                "tokens_in": len(subtask.instructions) // 4,
                "tokens_out": len(body) // 4,
                "wall_time_seconds": 0.0,
                "estimated_cost_usd": 0.0,
            },
            warnings=["Echo worker performs no analysis; output is not evidence."],
        )


class RawModelWorker(WorkerAdapter):
    """One generation against an injected callable — the thinnest real harness.

    ``generate`` is any ``(prompt) -> text``: an OpenAI-compatible client, a
    llama.cpp handle, a LiteLLM call. Keeping it a bare callable is what stops
    this package from growing a model-gateway dependency (§3).
    """

    def __init__(
        self,
        worker_id: str,
        generate: Callable[[str], str],
        *,
        model: str,
        harness: str = "raw_model",
        location: WorkerLocation = WorkerLocation.local,
        privacy_tiers: Optional[list[PrivacyTier]] = None,
        capability_tags: Optional[list[str]] = None,
        cost_per_1k_tokens_usd: float = 0.0,
        requires_approval: Optional[bool] = None,
    ) -> None:
        self._worker_id = worker_id
        self._generate = generate
        self._model = model
        self._harness = harness
        self._location = location
        self._privacy_tiers = privacy_tiers or [PrivacyTier.public, PrivacyTier.public_redacted]
        self._capability_tags = capability_tags or ["generate"]
        self._cost = cost_per_1k_tokens_usd
        # Local compute is free and stays on the box; anything else spends money
        # or leaves it, so it needs a human (§21). Overridable, never silent.
        self._requires_approval = (
            requires_approval
            if requires_approval is not None
            else location is not WorkerLocation.local
        )

    @property
    def worker_id(self) -> str:
        return self._worker_id

    def capabilities(self) -> WorkerCapabilities:
        return WorkerCapabilities(
            worker_id=self._worker_id,
            harness=self._harness,
            model=self._model,
            tools=["generate", "write_artifact"],
            sandbox=SandboxSpec(),
            privacy_tiers=list(self._privacy_tiers),
            capability_tags=list(self._capability_tags),
            location=self._location,
            cost_per_1k_tokens_usd=self._cost,
            requires_approval=self._requires_approval,
        )

    def run(self, subtask: Subtask, context: WorkerContext) -> WorkerResult:
        prompt = f"{subtask.title}\n\n{subtask.instructions}"
        try:
            text = self._generate(prompt)
        except Exception as exc:  # a harness failure is a failed run, not a crash
            return WorkerResult(status="failed", summary="Model call failed", error=str(exc))
        artifact = context.create_artifact(
            type=ArtifactType.worker_output,
            content=text,
            summary=f"{self._model} output for subtask {subtask.id}",
        )
        tokens_in = len(prompt) // 4
        tokens_out = len(text) // 4
        return WorkerResult(
            status="succeeded",
            summary=text.strip().splitlines()[0][:200] if text.strip() else "(empty output)",
            artifacts=[
                WorkerArtifactRef(
                    artifact_id=artifact.id,
                    type=ArtifactType.worker_output,
                    description=f"Raw {self._model} output",
                )
            ],
            metrics={
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "estimated_cost_usd": round((tokens_in + tokens_out) / 1000.0 * self._cost, 6),
            },
        )
