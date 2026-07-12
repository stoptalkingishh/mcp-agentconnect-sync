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

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..common.compression import Compressor
from ..common.config import CliConfig, ProviderConfig
from ..common.schemas import GenerateRequest
from ..common.secrets import SecretResolver
from .local_client import HttpLocalClient, LocalClient


def _default_cli_scratch_dir() -> Path:
    """Fixed, neutral working directory for cli_subprocess providers with no
    explicit `cli.cwd` configured -- deliberately never the router process's
    own cwd (which could be a real repo checkout). See `CliConfig`'s
    docstring: even a read-only sandbox mode still lets the CLI *read*
    whatever's in its working directory, so this is a real safety property,
    not cosmetic."""
    return Path(tempfile.gettempdir()) / "agentconnect-cli-scratch"


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
        if cfg.type == "cli_subprocess":
            return self._call_cli_subprocess(cfg, req)
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

    # --------------------------------------------------------- cli_subprocess
    def _call_cli_subprocess(self, cfg: ProviderConfig, req: GenerateRequest) -> GatewayResult:
        """CLI-subprocess call for a subscription-authenticated coding-agent
        CLI (claude_cli/codex_cli). No secret is resolved or held here — the
        CLI authenticates via its own ambient login state, entirely outside
        this gateway's secret-resolution path (see
        config/secrets.example.yaml's `cli_subprocess` note).

        Failures are surfaced to the caller, matching `_call_cloud`'s
        convention: a failed CLI invocation must never be reported as a
        successful task. `_on_result` still fires first so the circuit
        breaker learns about the failure either way.
        """
        cli = cfg.cli
        if cli is None:
            raise RuntimeError(f"cli_subprocess provider {cfg.provider_id} has no `cli` config.")

        prompt = self._render_cli_prompt(self._compressed(cfg, req))
        try:
            result = self._run_cli(cfg, cli, prompt)
        except Exception as exc:
            if self._on_result:
                self._on_result(cfg.provider_id, False, str(exc))
            raise RuntimeError(f"cli_subprocess provider {cfg.provider_id} failed") from exc

        if self._on_result:
            self._on_result(cfg.provider_id, True, None)
        return result

    @staticmethod
    def _render_cli_prompt(req: GenerateRequest) -> str:
        """Render `req.messages` into the single text blob a CLI's stdin
        expects. A single message needs no role framing; multiple messages
        get bracketed role headers so the CLI can tell them apart."""
        if len(req.messages) == 1:
            return str(req.messages[0].get("content", ""))
        return "\n\n".join(
            f"[{m.get('role', 'user')}]\n{m.get('content', '')}" for m in req.messages
        )

    def _run_cli(self, cfg: ProviderConfig, cli: CliConfig, prompt: str) -> GatewayResult:
        cwd = Path(cli.cwd) if cli.cwd else _default_cli_scratch_dir()
        cwd.mkdir(parents=True, exist_ok=True)

        args = list(cli.args)
        output_file_path: Path | None = None
        if cli.output_mode == "output_file":
            if not cli.output_file_flag:
                raise RuntimeError(
                    f"cli_subprocess provider {cfg.provider_id}: "
                    "output_mode=output_file requires output_file_flag"
                )
            fd, raw_path = tempfile.mkstemp(prefix="agentconnect-cli-", suffix=".txt")
            import os

            os.close(fd)
            output_file_path = Path(raw_path)
            args = [*args, cli.output_file_flag, str(output_file_path)]

        try:
            proc = subprocess.run(
                [cli.binary, *args],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=cli.timeout_seconds,
                cwd=str(cwd),
            )
            if cli.output_mode == "stdout_json":
                output_text, input_tokens, output_tokens = self._parse_cli_json(
                    cfg, proc.returncode, proc.stdout, proc.stderr
                )
            else:
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"{cfg.provider_id} exited {proc.returncode}: {proc.stderr[:500]}"
                    )
                if cli.output_mode == "output_file":
                    output_text = output_file_path.read_text(encoding="utf-8").strip()
                else:  # "stdout" plain text
                    output_text = proc.stdout.strip()
                # No structured usage available in these modes -- same rough
                # chars/4 heuristic _call_cloud's own stub fallback already
                # uses, not a claim of precision.
                input_tokens = len(prompt) // 4
                output_tokens = len(output_text) // 4
        finally:
            if output_file_path is not None:
                output_file_path.unlink(missing_ok=True)

        return GatewayResult(
            output_text=output_text[: cli.max_output_chars],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider=cfg.provider_id,
            # No separate model_id concept for a CLI subprocess -- the CLI
            # picks/switches its own model internally.
            model=cfg.provider_id,
        )

    @staticmethod
    def _parse_cli_json(
        cfg: ProviderConfig, returncode: int, stdout: str, stderr: str
    ) -> tuple[str, int, int]:
        """Parse Claude Code's `--output-format json` envelope (confirmed
        live: `{"type": "result", "is_error": bool, "result": "<text>",
        "usage": {"input_tokens": int, "output_tokens": int, ...}, ...}`)."""
        if returncode != 0:
            raise RuntimeError(f"{cfg.provider_id} exited {returncode}: {stderr[:500]}")
        data = json.loads(stdout)
        if data.get("is_error"):
            raise RuntimeError(f"{cfg.provider_id} returned is_error=true: {data.get('result')!r}")
        output_text = data.get("result", "")
        usage = data.get("usage") or {}
        return output_text, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
