"""Bridge from the router to the worker runtime (agentconnect-runtime).

An agentic task is not a single generation — it is the runtime's act/tool loop,
which calls the model once per step. The router has already selected a provider
and resolved secrets/mTLS behind the gateway, so the loop must reach the model
through that same gateway rather than opening its own path.

:class:`GatewayModelSource` adapts ``ProviderGateway.call`` to the runtime's
``ModelSource`` protocol (``generate(GenerateRequest) -> GenerateResponse``) and
sums token usage across every step, so the router can reconcile quota and record
one evaluation for the whole task the same way it does for a one-shot call.
"""

from __future__ import annotations

from ..common.config import ProviderConfig
from ..common.schemas import GenerateRequest, GenerateResponse
from .gateway import ProviderGateway


class GatewayModelSource:
    """A runtime ModelSource backed by the router's provider gateway.

    Every ``generate`` is a real provider call through the selected ``cfg``;
    ``total_input_tokens`` / ``total_output_tokens`` accumulate across the loop
    so the caller can bill and evaluate the task as a whole. ``model_id`` on the
    incoming request is honored — the runtime sets it from RuntimeConfig — but
    the provider routing (local/cloud, secrets, mTLS) is fixed to ``cfg``.
    """

    def __init__(self, gateway: ProviderGateway, cfg: ProviderConfig, model_id: str):
        self._gateway = gateway
        self._cfg = cfg
        self._model_id = model_id
        self.calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        # Pin the model to the routing decision; the runtime may leave model_id
        # empty or echo its config default, but the provider was chosen for a
        # specific model and must not be silently re-pointed mid-loop.
        call_req = req.model_copy(update={"model_id": self._model_id})
        result = self._gateway.call(self._cfg, call_req)
        self.calls += 1
        self.total_input_tokens += result.input_tokens
        self.total_output_tokens += result.output_tokens
        return GenerateResponse(
            request_id=req.request_id,
            model_id=result.model,
            output_text=result.output_text,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            finish_reason="stop",
        )
