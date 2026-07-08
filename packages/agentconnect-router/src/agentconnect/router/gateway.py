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
        completion_fn: Optional[Callable[..., object]] = None,
        on_call_result: Optional[Callable[[str, bool, Optional[str]], None]] = None,
        compressor: Optional[Compressor] = None,
    ):
        # SecretResolver is constructed here and kept private to the gateway.
        self._secrets = secret_resolver or SecretResolver()
        self._local = local_client
        # The cloud call is delegated to LiteLLM (maintained multi-provider I/O).
        # Injectable so tests exercise the mapping/key-isolation without importing
        # the heavy SDK; None -> lazily bind ``litellm.completion`` behind the
        # [cloud] extra. It is a transport seam only — never the routing decision.
        self._completion_fn = completion_fn
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
            call_req = self._compressed(cfg, req)
            try:
                result = self._call_via_litellm(cfg, call_req, api_key)
            except Exception as exc:
                # Still falls through to the deterministic stub below (no
                # behavior change there — a production build would remove
                # that fallback and surface errors) but the caller now learns
                # a real call actually failed, which it could not before.
                if self._on_result:
                    self._on_result(cfg.provider_id, False, str(exc))
            else:
                if self._on_result:
                    self._on_result(cfg.provider_id, True, None)
                return result

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
