"""Model residency + admission control (handoff §4.2, §18, §22).

Owns the *local execution* concerns only: which model is loaded, how many active
inference sequences are running, whether a new request can be admitted, and
executing loads/unloads/generations against the backend.

It deliberately does NOT own routing policy, quota, cloud logic, or secrets
(§26: "Do not let the inference machine become the global policy engine").
"""

from __future__ import annotations

import threading
from typing import Optional

from ..common.schemas import (
    AvailableModel,
    CanAcceptRequest,
    CanAcceptResponse,
    GenerateRequest,
    GenerateResponse,
    GpuStatus,
    LoadedModel,
    LoadRequest,
    LoadResponse,
    ManagerStatus,
    QueueStatus,
)
from .backends import ModelBackend, StubBackend


class ResidencyManager:
    def __init__(
        self,
        backend: Optional[ModelBackend] = None,
        node_id: str = "r9700-box-01",
        default_model: str = "qwen3.6-35b-a3b",
        max_active_sequences: int = 4,
        max_model_len: int = 16384,
        vram_total_gb: float = 32.0,
        gpu_name: str = "Radeon AI PRO R9700",
    ):
        self.backend = backend or StubBackend()
        self.node_id = node_id
        self.max_active_sequences = max_active_sequences
        self.max_model_len = max_model_len
        self.vram_total_gb = vram_total_gb
        self.gpu_name = gpu_name

        self._lock = threading.Lock()
        self._active_sequences = 0
        self._waiting = 0
        self._loaded_model: Optional[str] = None
        self._loading = False
        # Real admission: at most max_active_sequences run concurrently; excess
        # generate() calls block (queue) until a slot frees (handoff §18).
        self._slots = threading.BoundedSemaphore(max_active_sequences)

        # Load the default model at startup.
        self.load(LoadRequest(target_model=default_model, reason="startup_default"))

    # ------------------------------------------------------------- inventory
    def inventory(self) -> list[AvailableModel]:
        return self.backend.inventory()

    def _find(self, model_id: str) -> Optional[AvailableModel]:
        return next((m for m in self.backend.inventory() if m.model_id == model_id), None)

    # ---------------------------------------------------------------- status
    def status(self) -> ManagerStatus:
        with self._lock:
            loaded = None
            if self._loaded_model:
                meta = self._find(self._loaded_model)
                loaded = LoadedModel(
                    model_id=self._loaded_model,
                    quantization=meta.quantization if meta else None,
                    max_model_len=meta.max_model_len if meta else self.max_model_len,
                    max_active_sequences=self.max_active_sequences,
                    active_sequences=self._active_sequences,
                )
            state = "loading" if self._loading else ("busy" if self._active_sequences >= self.max_active_sequences else "ready")
            # Crude deterministic VRAM model: base + per-active-sequence.
            used = 20.0 + 2.0 * self._active_sequences if self._loaded_model else 0.0
            gpu = GpuStatus(
                name=self.gpu_name,
                vram_total_gb=self.vram_total_gb,
                vram_free_gb=round(max(0.0, self.vram_total_gb - used), 1),
                gpu_utilization_pct=round(min(100.0, 100.0 * self._active_sequences / max(1, self.max_active_sequences)), 1),
            )
            return ManagerStatus(
                node_id=self.node_id,
                status=state,
                backend=type(self.backend).__name__.replace("Backend", "").lower(),
                gpu=gpu,
                loaded_model=loaded,
                available_models=self.backend.inventory(),
                queue=QueueStatus(local_waiting=self._waiting, oldest_wait_seconds=0),
            )

    # -------------------------------------------------------------- admission
    def can_accept(self, req: CanAcceptRequest) -> CanAcceptResponse:
        meta = self._find(req.model_id)
        if meta is None:
            return CanAcceptResponse(can_accept=False, reason="model_not_available")
        total_ctx = req.estimated_input_tokens + req.estimated_output_tokens
        if total_ctx > meta.max_model_len:
            return CanAcceptResponse(
                can_accept=False,
                reason=f"context_exceeds_max_model_len({total_ctx}>{meta.max_model_len})",
            )
        with self._lock:
            switch_needed = self._loaded_model != req.model_id
            if switch_needed:
                wait = meta.estimated_load_seconds
                return CanAcceptResponse(
                    can_accept=True,
                    estimated_queue_wait_seconds=wait,
                    reason="model_switch_required",
                )
            if self._active_sequences >= self.max_active_sequences:
                # A slot will free up; report an estimated wait proportional to queue.
                return CanAcceptResponse(
                    can_accept=True,
                    estimated_queue_wait_seconds=6 * (self._waiting + 1),
                    reason="queued_no_free_slot",
                )
            return CanAcceptResponse(can_accept=True, estimated_queue_wait_seconds=0, reason="capacity_available")

    # ---------------------------------------------------------- load/unload
    def load(self, req: LoadRequest) -> LoadResponse:
        meta = self._find(req.target_model)
        if meta is None:
            return LoadResponse(accepted=False, reason="model_not_available")
        with self._lock:
            if self._loaded_model == req.target_model:
                return LoadResponse(
                    accepted=True, loaded_model=req.target_model,
                    estimated_load_seconds=0, reason="already_resident",
                )
            self._loading = True
        # In a real backend this blocks while weights load; the stub is instant.
        self.backend.load(req.target_model)
        with self._lock:
            if self._loaded_model:
                self.backend.unload(self._loaded_model)
            self._loaded_model = req.target_model
            self._loading = False
        return LoadResponse(
            accepted=True, loaded_model=req.target_model,
            estimated_load_seconds=meta.estimated_load_seconds, reason=req.reason or "loaded",
        )

    def unload(self, model_id: str) -> LoadResponse:
        with self._lock:
            if self._loaded_model != model_id:
                return LoadResponse(accepted=False, reason="not_loaded")
            self.backend.unload(model_id)
            self._loaded_model = None
        return LoadResponse(accepted=True, loaded_model=None, reason="unloaded")

    # ------------------------------------------------------------- generate
    def generate(self, req: GenerateRequest) -> GenerateResponse:
        # Ensure the requested model is resident (the router decides *whether* to
        # switch; the manager simply executes what it's told).
        if self._loaded_model != req.model_id:
            self.load(LoadRequest(target_model=req.model_id, reason="on_demand_for_generate"))
        # Block for a free inference slot (real admission control, §18). Excess
        # concurrent calls queue here rather than oversubscribing the GPU.
        with self._lock:
            self._waiting += 1
        self._slots.acquire()
        with self._lock:
            self._waiting = max(0, self._waiting - 1)
            self._active_sequences += 1
        try:
            return self.backend.generate(req)
        finally:
            with self._lock:
                self._active_sequences = max(0, self._active_sequences - 1)
            self._slots.release()
