"""Secret resolution (handoff §7, §24).

The router and agents deal only in `secret_ref` strings (e.g.
``op://AI/gemini/api-key``). Only the provider gateway ever calls
``SecretResolver.resolve()`` — at request time, immediately before the outbound
provider call — and the resolved value is never returned to the agent layer,
never logged, and never placed in a prompt or MCP tool output.

Resolution is driven by ``config/secrets.yaml`` (a copy of
``secrets.example.yaml``). Each ref maps to a resolver kind:

  env      -> read from an environment variable
  literal  -> inline value (dev/testing only)
  op       -> 1Password ref, resolved via the `op` CLI in real deployments

If a ref has no mapping we fall back to treating the ref itself as an `op://`
reference and, absent an `op` CLI, raise a clear error. That keeps the failure
mode "missing credential" rather than "silently no auth".
"""

from __future__ import annotations

import os
import subprocess
from typing import Optional

import yaml

from .config import CONFIG_DIR


class SecretResolutionError(RuntimeError):
    pass


class SecretResolver:
    """Resolves secret references to secret values. Keep instances out of the
    agent layer — construct one inside the gateway only."""

    def __init__(self, mapping: Optional[dict[str, dict]] = None):
        self._mapping = mapping if mapping is not None else self._load_mapping()

    @staticmethod
    def _load_mapping() -> dict[str, dict]:
        # Prefer the real (gitignored) secrets.yaml; fall back to the example so
        # that env-var-based resolution still works out of the box.
        for name in ("secrets.yaml", "secrets.local.yaml", "secrets.example.yaml"):
            path = CONFIG_DIR / name
            if path.exists():
                with path.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
                return data.get("secret_refs", {}) or {}
        return {}

    def resolve(self, secret_ref: str) -> str:
        """Return the secret value for a ref. Raises if it cannot be resolved.

        The returned string is sensitive: do not log it, echo it, or return it
        across the MCP/agent boundary.
        """
        spec = self._mapping.get(secret_ref)
        if spec is None:
            return self._resolve_op(secret_ref)

        kind = spec.get("kind")
        if kind == "env":
            var = spec["var"]
            val = os.environ.get(var)
            if not val:
                raise SecretResolutionError(
                    f"Secret ref {secret_ref!r} maps to env var {var!r}, which is unset."
                )
            return val
        if kind == "literal":
            val = spec.get("value")
            if not val:
                raise SecretResolutionError(f"Secret ref {secret_ref!r} has empty literal value.")
            return val
        if kind == "op":
            return self._resolve_op(spec.get("ref", secret_ref))
        raise SecretResolutionError(f"Unknown secret resolver kind {kind!r} for {secret_ref!r}.")

    @staticmethod
    def _resolve_op(ref: str) -> str:
        """Resolve a 1Password ``op://`` reference via the `op` CLI, if present."""
        if not ref.startswith("op://"):
            raise SecretResolutionError(
                f"No resolver mapping for secret ref {ref!r} and it is not an op:// reference."
            )
        op_bin = os.environ.get("OP_CLI", "op")
        try:
            out = subprocess.run(
                [op_bin, "read", ref],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
        except FileNotFoundError as exc:
            raise SecretResolutionError(
                f"Cannot resolve {ref!r}: `op` CLI not found. Install 1Password CLI or "
                f"add an env/literal mapping in config/secrets.yaml."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise SecretResolutionError(f"`op read {ref}` failed: {exc.stderr.strip()}") from exc
        return out.stdout.strip()

    def has(self, secret_ref: str) -> bool:
        """Whether a ref is resolvable *without* raising (best-effort, no fetch of op://)."""
        spec = self._mapping.get(secret_ref)
        if spec is None:
            return False
        if spec.get("kind") == "env":
            return bool(os.environ.get(spec.get("var", "")))
        if spec.get("kind") == "literal":
            return bool(spec.get("value"))
        return True
