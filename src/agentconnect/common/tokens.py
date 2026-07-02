"""Cheap, deterministic token estimation (handoff §11 step 6).

A rough heuristic (~4 chars/token) good enough for quota reservation and context
cap checks. Deterministic by construction — no model calls, no randomness.
"""

from __future__ import annotations

_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def estimate_io_tokens(prompt: str, max_output_tokens: int) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) estimate for a task."""
    return estimate_tokens(prompt), max(1, max_output_tokens)
