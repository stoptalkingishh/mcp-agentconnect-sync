"""Worker-local policy guards.

Global policy stays in the router. This module is where task-local execution
limits and guardrails will live.
"""

from __future__ import annotations


def max_steps_allowed(default: int = 12) -> int:
    return max(1, default)
