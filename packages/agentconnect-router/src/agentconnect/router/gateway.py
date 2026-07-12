"""Provider gateway — the only component that touches secrets (handoff §7, §24).

Given a provider id and a request, the gateway:
  1. for CLOUD providers, resolves the API key at call time (via
     :class:`SecretResolver`); for LOCAL nodes, uses mutual TLS (no secret),
  2. makes the outbound call (local via the Local Model Manager, cloud via HTTP),
  3. returns only the model output + token usage.

Any resolved cloud secret NEVER leaves this module: it is not logged, not returned,
and not placed in any artifact or MCP payload. Local inference nodes carry no
shared secret at all — identity is the client certificate. Agents and Claude only
ever see the provider id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..common.compression import Compressor
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
        on_call_result: Optional[Callable[[str, bool, Optional[str]], None]] = None,
        compressor: Optional[Compressor] = None,
    ):
        # SecretResolver is constructed here and kept private to the gateway.
        self._secrets = secret_resolver or SecretResolver()
        self._local = local_client
        # Notified (provider_id, success, error) after every real cloud call
        # attempt — including the one currently swallowed into the stub
        # fallback below. Used by RouterService to feed a circuit breaker;
        # never sees secrets, only the provider id and a bool/error string.
        self._on_result = on_call_result
        # Compresses outbound cloud message content (tool-output/prose) before
        # the HTTP payload is built. Cloud-only — local/rented are untouched.
        self._compressor = compressor

    # ----------------------------------------------------------- local wiring
    def bind_local(self, client: LocalClient) -> None:
        self._local = client

    def bind_result_callback(self, callback: Callable[[str, bool, Optional[str]], None]) -> None:
        self._on_result = callback

    def bind_compressor(self, compressor: Compressor) -> None:
        self._compressor = compressor

    def _local_client_for(self, cfg: ProviderConfig) -> LocalClient:
        if self._local is not None:
            return self._local
        if not cfg.manager_endpoint:
            raise RuntimeError(f"Local provider {cfg.provider_id} has no manager_endpoint configured.")
        # Local nodes authenticate via mutual TLS — no secret is resolved here.
        # Identity is the client certificate (see HttpLocalClient / handoff §7).
        return HttpLocalClient(cfg.manager_endpoint, tls=cfg.tls)

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

    def _compressed(self, cfg: ProviderConfig, req: GenerateRequest) -> GenerateRequest:
        """Return `req` with each message's content compressed for this
        provider, or `req` unchanged if no compressor is bound. A message
        whose content starts with "OBSERVATION:" (the tool-output convention
        from runtime/graph.py's act/tool loop) is compressed in tool_output
        mode; everything else is treated as prose."""
        if self._compressor is None:
            return req
        new_messages = []
        for m in req.messages:
            content = m.get("content")
            if not isinstance(content, str):
                new_messages.append(m)
                continue
            kind = "tool_output" if content.startswith("OBSERVATION:") else "prose"
            compressed, _stats = self._compressor.compress_for_provider(cfg.provider_id, content, kind)
            new_messages.append({**m, "content": compressed})
        return req.model_copy(update={"messages": new_messages})

    def _call_cloud(self, cfg: ProviderConfig, req: GenerateRequest) -> GatewayResult:
        """Cloud call. Resolves the API key at call time and never returns it.

        Provider and credential failures are surfaced to the caller. A failed
        cloud request must never be reported as a successful task.
        """
        try:
            api_key = self._secrets.resolve(cfg.secret_ref)
        except Exception as exc:
            if self._on_result:
                self._on_result(cfg.provider_id, False, str(exc))
            raise RuntimeError(f"No usable credential for cloud provider {cfg.provider_id}") from exc

        model_id = cfg.model_map.get(req.model_id, req.model_id)
        if cfg.model_prefix and not model_id.startswith(cfg.model_prefix):
            model_id = cfg.model_prefix + model_id
        call_req = self._compressed(cfg, req.model_copy(update={"model_id": model_id}))
        try:
            result = self._http_openai_compatible(cfg, call_req, api_key)
        except Exception as exc:
            if self._on_result:
                self._on_result(cfg.provider_id, False, str(exc))
            raise RuntimeError(f"Cloud provider {cfg.provider_id} request failed") from exc

        if self._on_result:
            self._on_result(cfg.provider_id, True, None)
        return result

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
            "stream": False,
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
