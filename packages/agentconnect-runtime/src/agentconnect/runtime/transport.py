"""Runtime transport placeholders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeEndpoint:
    kind: str  # local | http | future-remote
    address: str
