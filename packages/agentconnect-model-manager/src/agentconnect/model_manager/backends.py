"""Model server backend abstraction (handoff §3).

The Local Model Manager is meant to sit in front of a real serving backend —
vLLM (ROCm), llama.cpp server, Ollama, or SGLang. This module defines the
minimal interface the residency executor needs and ships a deterministic
:class:`StubBackend` so the whole system runs end-to-end without a GPU.

Swap in a real backend by implementing :class:`ModelBackend`.
"""

from __future__ import annotations

import abc
import os
from typing import Any, Optional

from ..common.schemas import AvailableModel, GenerateRequest, GenerateResponse


class ModelBackend(abc.ABC):
    @abc.abstractmethod
    def inventory(self) -> list[AvailableModel]:
        """Models installed on this node."""

    @abc.abstractmethod
    def load(self, model_id: str) -> None:
        """Make `model_id` resident. Blocking in real backends; instant in stub."""

    @abc.abstractmethod
    def unload(self, model_id: str) -> None:
        ...

    @abc.abstractmethod
    def generate(self, req: GenerateRequest) -> GenerateResponse:
        ...


# --- Default inventory for the appliance described in the handoff (§5, §17) ---
_DEFAULT_INVENTORY = [
    AvailableModel(
        model_id="qwen3.6-35b-a3b",
        profiles=["main_concurrent_worker", "general_coder", "default_worker", "resident_ok"],
        estimated_load_seconds=45, supports_tools=False, supports_vision=False,
        max_model_len=16384, quantization="q4",
    ),
    AvailableModel(
        model_id="ornith-1.0-35b",
        profiles=["coding_specialist", "coding_patch"],
        estimated_load_seconds=55, supports_tools=False, supports_vision=False,
        max_model_len=16384, quantization="q4",
    ),
    AvailableModel(
        model_id="qwen3.6-27b",
        profiles=["review_worker", "critic", "coding_review"],
        estimated_load_seconds=40, supports_tools=False, supports_vision=True,
        max_model_len=16384, quantization="q4",
    ),
]


class StubBackend(ModelBackend):
    """Deterministic in-process backend for development and tests.

    Generation echoes a compact, structured, deterministic response so callers
    can exercise the full path (dispatch -> generate -> artifact) offline.
    """

    def __init__(self, inventory: list[AvailableModel] | None = None):
        self._inventory = inventory if inventory is not None else list(_DEFAULT_INVENTORY)
        self._loaded: set[str] = set()

    def inventory(self) -> list[AvailableModel]:
        return list(self._inventory)

    def load(self, model_id: str) -> None:
        self._loaded.add(model_id)

    def unload(self, model_id: str) -> None:
        self._loaded.discard(model_id)

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        last_user = ""
        for m in reversed(req.messages):
            if m.get("role") == "user":
                last_user = str(m.get("content", ""))
                break
        text = (
            f"[stub:{req.model_id}] processed task {req.task_id}. "
            f"prompt_chars={len(last_user)}."
        )
        in_tokens = sum(len(str(m.get("content", ""))) for m in req.messages) // 4
        out_tokens = min(req.max_output_tokens, len(text) // 4 + 1)
        return GenerateResponse(
            request_id=req.request_id,
            model_id=req.model_id,
            output_text=text,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            finish_reason="stop",
        )


class OpenAICompatibleBackend(ModelBackend):
    """Real backend for an OpenAI-compatible server (vLLM, llama.cpp server,
    Ollama, SGLang, TGI). Generation POSTs to ``{base_url}/chat/completions``.

    ``client`` is injectable (an ``httpx.Client``) so tests can drive it with a
    mock transport; in production it is built from ``base_url`` + an optional API
    key. Loading/unloading is a no-op here — the external server owns residency —
    but a deployment can front several single-model servers and treat load/unload
    as routing between them.
    """

    def __init__(
        self,
        base_url: str,
        models: list[AvailableModel],
        api_key: Optional[str] = None,
        client: Any = None,
        timeout: float = 120.0,
    ):
        self._base = base_url.rstrip("/")
        self._inventory = list(models)
        self._loaded: set[str] = set()
        if client is not None:
            self._client = client
        else:
            import httpx

            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            self._client = httpx.Client(base_url=self._base, headers=headers, timeout=timeout)

    def inventory(self) -> list[AvailableModel]:
        return list(self._inventory)

    def load(self, model_id: str) -> None:
        self._loaded.add(model_id)

    def unload(self, model_id: str) -> None:
        self._loaded.discard(model_id)

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        payload = {
            "model": req.model_id,
            "messages": req.messages,
            "max_tokens": req.max_output_tokens,
            "temperature": req.temperature,
        }
        resp = self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        text = choice.get("message", {}).get("content", "")
        usage = data.get("usage", {})
        return GenerateResponse(
            request_id=req.request_id,
            model_id=req.model_id,
            output_text=text,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
        )


def backend_from_env() -> ModelBackend:
    """Select a backend from env: MODEL_MANAGER_BACKEND=stub (default) | openai.

    For ``openai``: MODEL_BACKEND_URL (required), MODEL_BACKEND_API_KEY (optional),
    MODEL_BACKEND_MODELS (comma-separated model ids; defaults to the stub inventory
    ids)."""
    kind = os.environ.get("MODEL_MANAGER_BACKEND", "stub").lower()
    if kind in ("stub", ""):
        return StubBackend()
    if kind in ("openai", "openai_compatible", "vllm", "llamacpp", "ollama"):
        base = os.environ.get("MODEL_BACKEND_URL")
        if not base:
            raise RuntimeError("MODEL_MANAGER_BACKEND=openai requires MODEL_BACKEND_URL")
        ids = [m.strip() for m in os.environ.get("MODEL_BACKEND_MODELS", "").split(",") if m.strip()]
        models = (
            [AvailableModel(model_id=i) for i in ids]
            if ids
            else list(_DEFAULT_INVENTORY)
        )
        return OpenAICompatibleBackend(
            base_url=base, models=models, api_key=os.environ.get("MODEL_BACKEND_API_KEY")
        )
    raise RuntimeError(f"Unknown MODEL_MANAGER_BACKEND={kind!r}")
