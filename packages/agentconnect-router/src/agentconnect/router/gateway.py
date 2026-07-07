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
        completion_fn: Optional[Callable[..., object]] = None,
    ):
        # SecretResolver is constructed here and kept private to the gateway.
        self._secrets = secret_resolver or SecretResolver()
        self._local = local_client
        # The cloud call is delegated to LiteLLM (maintained multi-provider I/O).
        # Injectable so tests exercise the mapping/key-isolation without importing
        # the heavy SDK; None -> lazily bind ``litellm.completion`` behind the
        # [cloud] extra. It is a transport seam only — never the routing decision.
        self._completion_fn = completion_fn

    # ----------------------------------------------------------- local wiring
    def bind_local(self, client: LocalClient) -> None:
        self._local = client

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

    def _call_cloud(self, cfg: ProviderConfig, req: GenerateRequest) -> GatewayResult:
        """Cloud call. Resolves the API key at call time and never returns it.

        The call itself is delegated to LiteLLM (maintained multi-provider I/O),
        and degrades to a deterministic stub when the SDK/network/credentials are
        unavailable, so the pipeline stays exercisable offline. A production build
        would remove the stub fallback and surface errors.
        """
        try:
            api_key = self._secrets.resolve(cfg.secret_ref)  # noqa: F841  (used below, never logged)
        except Exception:
            api_key = None

        if api_key:
            try:
                return self._call_via_litellm(cfg, req, api_key)
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

    def _resolve_completion_fn(self) -> Callable[..., object]:
        """The injected completion function, or ``litellm.completion`` (lazy, behind
        the [cloud] extra so stub/local-only deployments never import it)."""
        if self._completion_fn is not None:
            return self._completion_fn
        try:
            from litellm import completion  # type: ignore
        except ImportError as e:  # pragma: no cover - only without the extra
            raise RuntimeError(
                "cloud generation needs the [cloud] extra: pip install "
                "'agentconnect-router[cloud]'"
            ) from e
        return completion

    def _call_via_litellm(
        self, cfg: ProviderConfig, req: GenerateRequest, api_key: str
    ) -> GatewayResult:
        """Route the cloud call through LiteLLM. The API key is passed EXPLICITLY
        per-call and never written to the environment, so the secret stays inside
        this module (handoff §7/§24). ``litellm_model`` selects a NATIVE provider
        handler (e.g. ``gemini/...``); otherwise we call in OpenAI-compatible mode
        against ``cfg.endpoint`` — the prior behavior, now maintained by LiteLLM.
        """
        completion = self._resolve_completion_fn()
        kwargs: dict[str, object] = {
            "messages": req.messages,
            "max_tokens": req.max_output_tokens,
            "temperature": req.temperature,
            "api_key": api_key,
        }
        if cfg.litellm_model:
            kwargs["model"] = cfg.litellm_model
            if cfg.endpoint:
                kwargs["api_base"] = cfg.endpoint.rstrip("/")
        else:
            # OpenAI-compatible endpoint (the default): the "openai/" prefix tells
            # LiteLLM to POST /chat/completions at api_base, matching the old path.
            kwargs["model"] = f"openai/{req.model_id}"
            kwargs["api_base"] = cfg.endpoint.rstrip("/")

        resp = completion(**kwargs)
        choice = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return GatewayResult(
            output_text=choice,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            provider=cfg.provider_id,
            model=req.model_id,
        )
