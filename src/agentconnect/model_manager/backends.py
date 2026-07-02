"""Model server backend abstraction (handoff §3).

The Local Model Manager is meant to sit in front of a real serving backend —
vLLM (ROCm), llama.cpp server, Ollama, or SGLang. This module defines the
minimal interface the residency executor needs and ships a deterministic
:class:`StubBackend` so the whole system runs end-to-end without a GPU.

Swap in a real backend by implementing :class:`ModelBackend`.
"""

from __future__ import annotations

import abc
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
