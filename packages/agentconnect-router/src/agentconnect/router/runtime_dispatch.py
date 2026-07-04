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
from .local_client import LocalClient


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


class RentedModelSource:
    """A runtime ModelSource backed by an ALREADY-ACQUIRED rented node's client.

    The gateway cannot serve a rented node (it provisions/bills nothing — it would
    fall back to the owned local client or a bare HTTP endpoint), so agentic runs
    on a rented tier reach the model directly through the ``LocalClient`` the
    router built for the node it acquired. The node is provisioned, billed for its
    rental window, and released EXACTLY ONCE by the caller around the whole loop;
    this source only pins the model and sums usage across steps, mirroring
    :class:`GatewayModelSource` so the caller reconciles the task as a whole.
    """

    def __init__(self, client: LocalClient, model_id: str):
        self._client = client
        self._model_id = model_id
        self.calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        # Pin the model to the routing decision (see GatewayModelSource.generate).
        call_req = req.model_copy(update={"model_id": self._model_id})
        resp = self._client.generate(call_req)
        self.calls += 1
        self.total_input_tokens += resp.input_tokens
        self.total_output_tokens += resp.output_tokens
        return GenerateResponse(
            request_id=req.request_id,
            model_id=resp.model_id,
            output_text=resp.output_text,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            finish_reason="stop",
        )
