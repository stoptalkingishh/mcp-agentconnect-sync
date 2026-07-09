"""External local-model-manager boundary (adapters spec, Part B).

**AgentConnect defines the contract; it does not own the engine.** VRAM
admission, model loading, runtime selection, and queueing are somebody else's
problem — a user's Ollama, vLLM, SGLang, llama.cpp, LM Studio, or bespoke rig.

AgentConnect asks one question — *can local compute handle this?* — and gets
back a model/runtime/queue estimate or a refusal. AgentConnect then decides
whether to run local, try cheap cloud, request approval, fail, or queue. It never
picks a quantization.

The whole subsystem is optional: with no provider configured, nothing here runs
and routing simply never sees a `local_model_manager` worker.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .models import ArtifactType, PrivacyTier, SandboxSpec, Subtask, WorkerLocation
from .workers import (
    WorkerAdapter,
    WorkerArtifactRef,
    WorkerCapabilities,
    WorkerContext,
    WorkerEstimate,
    WorkerHealth,
    WorkerResult,
)

_log = logging.getLogger(__name__)


@dataclass
class LocalModel:
    id: str
    runtime: str
    capabilities: list[str]
    context_tokens: int
    loaded: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalEstimateRequest:
    task_type: str
    privacy_tier: str
    required_capabilities: list[str]
    context_tokens: int
    max_output_tokens: int
    latency_preference: str = "normal"
    quality_preference: str = "good_enough"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalEstimate:
    eligible: bool
    selected_model: Optional[str] = None
    runtime: Optional[str] = None
    loaded: bool = False
    estimated_queue_seconds: Optional[float] = None
    estimated_tokens_per_second: Optional[float] = None
    estimated_quality: Optional[float] = None
    reason: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalRunRequest:
    model: Optional[str]
    task_type: str
    prompt: str
    context: Optional[str] = None
    max_output_tokens: int = 2048
    temperature: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalRunResult:
    status: str
    output: str
    model: Optional[str]
    runtime: Optional[str]
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class LocalComputeProvider(abc.ABC):
    @abc.abstractmethod
    def inventory(self) -> list[LocalModel]: ...

    @abc.abstractmethod
    def loaded(self) -> list[LocalModel]: ...

    @abc.abstractmethod
    def estimate(self, request: LocalEstimateRequest) -> LocalEstimate: ...

    @abc.abstractmethod
    def run(self, request: LocalRunRequest) -> LocalRunResult: ...

    def health(self) -> dict[str, Any]:
        return {"status": "unknown"}


class HttpLocalComputeProvider(LocalComputeProvider):
    """Speaks the local-manager HTTP surface the spec names.

    ``GET /health``, ``GET /models``, ``GET /models/loaded``,
    ``POST /route/estimate``, ``POST /generate``, ``POST /runs/{id}/cancel``.

    The transport is injectable — `(method, url, json) -> dict` — so this is
    testable against a mock without a server, which is exactly how the "local
    manager is optional" tests run.
    """

    def __init__(
        self,
        base_url: str,
        transport: Optional[Callable[[str, str, Optional[dict]], dict]] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._transport = transport
        self._timeout = timeout

    def _call(self, method: str, path: str, payload: Optional[dict] = None) -> dict[str, Any]:
        if self._transport is not None:
            return self._transport(method, f"{self.base_url}{path}", payload) or {}
        import httpx  # lazy

        response = httpx.request(
            method, f"{self.base_url}{path}", json=payload, timeout=self._timeout
        )
        response.raise_for_status()
        return response.json() or {}

    @staticmethod
    def _model(raw: dict[str, Any]) -> LocalModel:
        return LocalModel(
            id=str(raw.get("id", "")), runtime=str(raw.get("runtime", "unknown")),
            capabilities=list(raw.get("capabilities", [])),
            context_tokens=int(raw.get("context_tokens", 0)),
            loaded=bool(raw.get("loaded", False)), metadata=raw.get("metadata") or {},
        )

    def inventory(self) -> list[LocalModel]:
        return [self._model(m) for m in self._call("GET", "/models").get("models", [])]

    def loaded(self) -> list[LocalModel]:
        return [self._model(m) for m in self._call("GET", "/models/loaded").get("models", [])]

    def estimate(self, request: LocalEstimateRequest) -> LocalEstimate:
        body = self._call("POST", "/route/estimate", {
            "task_type": request.task_type, "privacy_tier": request.privacy_tier,
            "required_capabilities": request.required_capabilities,
            "context_tokens": request.context_tokens,
            "max_output_tokens": request.max_output_tokens,
            "latency_preference": request.latency_preference,
            "quality_preference": request.quality_preference,
        })
        return LocalEstimate(
            eligible=bool(body.get("eligible", False)),
            selected_model=body.get("selected_model"), runtime=body.get("runtime"),
            loaded=bool(body.get("loaded", False)),
            estimated_queue_seconds=body.get("estimated_queue_seconds"),
            estimated_tokens_per_second=body.get("estimated_tokens_per_second"),
            estimated_quality=body.get("estimated_quality"),
            reason=body.get("reason") or {},
        )

    def run(self, request: LocalRunRequest) -> LocalRunResult:
        body = self._call("POST", "/generate", {
            "model": request.model, "task_type": request.task_type, "prompt": request.prompt,
            "context": request.context, "max_output_tokens": request.max_output_tokens,
            "temperature": request.temperature,
        })
        return LocalRunResult(
            status=str(body.get("status", "succeeded")), output=str(body.get("output", "")),
            model=body.get("model"), runtime=body.get("runtime"),
            metrics=body.get("metrics") or {}, warnings=list(body.get("warnings", [])),
        )

    def cancel(self, run_id: str) -> None:
        try:
            self._call("POST", f"/runs/{run_id}/cancel")
        except Exception as exc:  # cancellation is best effort
            _log.info("local manager cancel(%s) failed: %s", run_id, exc)

    def health(self) -> dict[str, Any]:
        try:
            return self._call("GET", "/health")
        except Exception as exc:
            return {"status": "unreachable", "detail": str(exc)}


class LocalModelManagerWorkerAdapter(WorkerAdapter):
    """Exposes an external local model manager to routing as one worker.

    Routing decides *whether local compute is eligible*; this adapter asks the
    manager *which model* and reports the answer back as a nested `local_estimate`
    on the route explanation. AgentConnect never reads the manager's internals.

    An outage is a rejected worker (``health.available = False``), not an
    exception: the router then falls back per policy, and the API/MCP survive.
    """

    def __init__(
        self,
        provider: LocalComputeProvider,
        worker_id: str = "local-manager",
        privacy_tiers: Optional[list[PrivacyTier]] = None,
        capability_tags: Optional[list[str]] = None,
        task_type: str = "general",
        max_output_tokens: int = 2048,
    ) -> None:
        self.provider = provider
        self._worker_id = worker_id
        self._privacy_tiers = privacy_tiers or list(PrivacyTier)
        self._capability_tags = capability_tags or ["generate", "code", "summarize"]
        self._task_type = task_type
        self._max_output_tokens = max_output_tokens

    @property
    def worker_id(self) -> str:
        return self._worker_id

    def capabilities(self) -> WorkerCapabilities:
        return WorkerCapabilities(
            worker_id=self._worker_id, harness="local_model_manager", model=None,
            tools=["generate", "write_artifact"], sandbox=SandboxSpec(),
            privacy_tiers=list(self._privacy_tiers),
            capability_tags=list(self._capability_tags),
            location=WorkerLocation.local, cost_per_1k_tokens_usd=0.0,
            requires_approval=False,
        )

    def _estimate_request(self, subtask: Subtask) -> LocalEstimateRequest:
        return LocalEstimateRequest(
            task_type=self._task_type, privacy_tier=subtask.privacy_tier.value,
            required_capabilities=list(subtask.required_capabilities),
            context_tokens=max(1, len(subtask.instructions) // 4),
            max_output_tokens=self._max_output_tokens,
        )

    def estimate(
        self, subtask: Subtask, context: Optional[WorkerContext] = None
    ) -> WorkerEstimate:
        tokens_in = max(1, len(subtask.instructions) // 4)
        estimate = WorkerEstimate(
            estimated_cost_usd=0.0, estimated_tokens_in=tokens_in,
            estimated_tokens_out=tokens_in // 4,
        )
        try:
            local = self.provider.estimate(self._estimate_request(subtask))
        except Exception as exc:
            _log.warning("local manager estimate failed: %s", exc)
            return estimate
        if local.estimated_queue_seconds is not None:
            estimate.estimated_seconds = float(local.estimated_queue_seconds)
        return estimate

    def local_estimate(self, subtask: Subtask) -> Optional[LocalEstimate]:
        """The nested `local_estimate` section of the route explanation."""
        try:
            return self.provider.estimate(self._estimate_request(subtask))
        except Exception as exc:
            _log.warning("local manager estimate failed: %s", exc)
            return None

    def health(self) -> WorkerHealth:
        try:
            body = self.provider.health()
        except Exception as exc:
            return WorkerHealth(available=False, detail=f"local manager unreachable: {exc}")
        status = str(body.get("status", "unknown"))
        if status in ("unreachable", "down", "error"):
            return WorkerHealth(
                available=False,
                detail=f"local manager {status}: {body.get('detail', '')}".strip(),
            )
        return WorkerHealth(available=True, detail=status)

    def run(self, subtask: Subtask, context: WorkerContext) -> WorkerResult:
        estimate = self.local_estimate(subtask)
        if estimate is not None and not estimate.eligible:
            return WorkerResult(
                status="failed", summary="Local compute declined the subtask",
                error=f"local manager reported ineligible: {estimate.reason}",
            )
        try:
            result = self.provider.run(LocalRunRequest(
                model=estimate.selected_model if estimate else None,
                task_type=self._task_type,
                prompt=f"{subtask.title}\n\n{subtask.instructions}",
                max_output_tokens=self._max_output_tokens,
            ))
        except Exception as exc:
            # An outage is a failed run, recorded in the ledger — never a crash
            # that takes the API or the MCP server down with it.
            return WorkerResult(
                status="failed", summary="Local model manager unavailable", error=str(exc)
            )
        if result.status != "succeeded":
            return WorkerResult(
                status="failed", summary="Local model manager reported failure",
                error=result.status, warnings=result.warnings,
            )
        artifact = context.create_artifact(
            type=ArtifactType.worker_output, content=result.output,
            summary=f"{result.model or 'local model'} output for subtask {subtask.id}",
        )
        metrics = dict(result.metrics)
        metrics.setdefault("estimated_cost_usd", 0.0)
        return WorkerResult(
            status="succeeded",
            summary=(result.output.strip().splitlines() or ["(empty output)"])[0][:200],
            artifacts=[WorkerArtifactRef(
                artifact_id=artifact.id, type=ArtifactType.worker_output,
                description=f"Output from {result.model} on {result.runtime}",
            )],
            metrics=metrics, warnings=result.warnings,
        )
