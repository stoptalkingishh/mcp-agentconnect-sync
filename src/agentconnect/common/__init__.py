"""Shared core for the Agent Router and Local Model Manager.

This package holds the pieces that are policy- and framework-agnostic: schemas,
config loaders, the task state machine, shared memory/artifact store, quota
ledger, privacy/redaction, provider registry, secret resolution, and token
estimation. It has no dependency on FastAPI or the MCP SDK, so it can be unit
tested in isolation (handoff §26: keep responsibilities separate).
"""

from . import config, memory, privacy, providers, quota, schemas, secrets, state, tokens  # noqa: F401

__all__ = [
    "config",
    "memory",
    "privacy",
    "providers",
    "quota",
    "schemas",
    "secrets",
    "state",
    "tokens",
]
