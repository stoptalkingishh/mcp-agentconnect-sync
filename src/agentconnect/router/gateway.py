"""Provider gateway — the only component that touches secrets (handoff §7, §24).

Given a provider id and a request, the gateway:
  1. resolves the provider's secret at call time (via :class:`SecretResolver`),
  2. makes the outbound call (local via the Local Model Manager, cloud via HTTP),
  3. returns only the model output + token usage.

The resolved secret NEVER leaves this module: it is not logged, not returned, and
not placed in any artifact or MCP payload. Agents and Claude only ever see the
provider id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..common.config import ProviderConfig
from ..common.schemas import GenerateRequest
from ..common.secrets import SecretResolver
from .local_client import HttpLocalClient, LocalClient


@dataclass
class GatewayResult:
    output_text: str
    input_tokens: int
    output_tokens: int
    provider: str
    model: str


class ProviderGateway:
    def __init__(
        self,
        secret_resolver: Optional[SecretResolver] = None,
        local_client: Optional[LocalClient] = None,
    ):
        # SecretResolver is constructed here and kept private to the gateway.
        self._secrets = secret_resolver or SecretResolver()
        self._local = local_client

    # ----------------------------------------------------------- local wiring
    def bind_local(self, client: LocalClient) -> None:
        self._local = client

    def _local_client_for(self, cfg: ProviderConfig) -> LocalClient:
        if self._local is not None:
            return self._local
        if not cfg.manager_endpoint:
            raise RuntimeError(f"Local provider {cfg.provider_id} has no manager_endpoint configured.")
        # Resolve the manager's bearer token from the secrets manager (§7).
        token = self._secrets.resolve(cfg.secret_ref)
        return HttpLocalClient(cfg.manager_endpoint, token=token)

    # --------------------------------------------------------------- dispatch
    def call(self, cfg: ProviderConfig, req: GenerateRequest) -> GatewayResult:
        if cfg.type == "local":
            client = self._local_client_for(cfg)
            resp = client.generate(req)
            return GatewayResult(
                output_text=resp.output_text,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                provider=cfg.provider_id,
                model=resp.model_id,
            )
        return self._call_cloud(cfg, req)

    def _call_cloud(self, cfg: ProviderConfig, req: GenerateRequest) -> GatewayResult:
        """Cloud call. Resolves the API key at call time and never returns it.

        This scaffold implements an OpenAI-compatible chat call and degrades to a
        deterministic stub when the SDK/network/credentials are unavailable, so
        the pipeline stays exercisable offline. A production build would remove
        the stub fallback and surface errors.
        """
        try:
            api_key = self._secrets.resolve(cfg.secret_ref)  # noqa: F841  (used below, never logged)
        except Exception:
            api_key = None

        if api_key:
            try:
                return self._http_openai_compatible(cfg, req, api_key)
            except Exception:
                # Fall through to the deterministic stub rather than leaking why.
                pass

        text = f"[cloud-stub:{cfg.provider_id}/{req.model_id}] task {req.task_id} (no live call)."
        return GatewayResult(
            output_text=text,
            input_tokens=sum(len(str(m.get("content", ""))) for m in req.messages) // 4,
            output_tokens=len(text) // 4 + 1,
            provider=cfg.provider_id,
            model=req.model_id,
        )

    def _http_openai_compatible(
        self, cfg: ProviderConfig, req: GenerateRequest, api_key: str
    ) -> GatewayResult:
        import httpx

        url = cfg.endpoint.rstrip("/") + "/chat/completions"
        payload = {
            "model": req.model_id,
            "messages": req.messages,
            "max_tokens": req.max_output_tokens,
            "temperature": req.temperature,
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        with httpx.Client(timeout=60.0) as client:
            r = client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return GatewayResult(
            output_text=choice,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            provider=cfg.provider_id,
            model=req.model_id,
        )
