"""Native, dependency-free outbound-payload compression (concepts adapted
from OmniRoute's compression pipeline — no dependency on OmniRoute itself,
and deliberately no ML/ONNX engine, unlike OmniRoute's LLMLingua-2 tier).

Two modes:
  * ``tool_output`` — for tool-observation messages (see ``runtime/graph.py``'s
    ``f"OBSERVATION:\\n{obs}"`` convention): collapses long runs of repeated
    lines and strips ANSI escape codes / progress-bar noise.
  * ``prose`` — light whitespace normalization and filler-phrase trimming.
    Explicitly scoped as a lightweight heuristic layer, not a claim to match
    OmniRoute's Caveman engine's ~70% prose-compression ratio.

Fenced code blocks (```...```) and bare URLs are always preserved byte-for-
byte in both modes.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_URL_RE = re.compile(r"https?://\S+")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# Lines that are just a progress indicator: "Downloading... 45%", "[#### ] 40%".
_PROGRESS_RE = re.compile(r"^\s*[\[\]\|/\\\-#.\s]*\d{1,3}%\s*\]?\s*$")

_FILLER_PHRASES = (
    "I would recommend that ",
    "I would recommend ",
    "It's worth noting that ",
    "It is worth noting that ",
    "In order to ",
    "Please note that ",
    "It should be noted that ",
)


@dataclass(frozen=True)
class CompressionStats:
    original_chars: int
    compressed_chars: int

    @property
    def ratio(self) -> float:
        if self.original_chars == 0:
            return 0.0
        return 1.0 - (self.compressed_chars / self.original_chars)


def _protect(text: str, pattern: re.Pattern, tag: str) -> tuple[str, dict[str, str]]:
    """Replace every match of ``pattern`` with a unique placeholder token, so
    compression can't touch it; returns the placeholder text and a
    token -> original map to restore afterwards.

    ``tag`` must be distinct per call site (e.g. "CODE" vs "URL") — without it,
    two separate ``_protect`` calls each numbering their own tokens from 0
    can produce identical placeholder strings (e.g. both emit ``PROT0``),
    and ``_restore`` would then replace every occurrence of that ambiguous
    token with whichever map processes it, corrupting the other one."""
    tokens: dict[str, str] = {}

    def _sub(m: re.Match) -> str:
        token = f"\x00PROT_{tag}_{len(tokens)}\x00"
        tokens[token] = m.group(0)
        return token

    return pattern.sub(_sub, text), tokens


def _restore(text: str, tokens: dict[str, str]) -> str:
    for token, original in tokens.items():
        text = text.replace(token, original)
    return text


def _collapse_repeated_lines(text: str, *, min_run: int = 4, head: int = 2, tail: int = 1) -> str:
    """Collapse runs of >= min_run consecutive identical lines into a short
    head + an omission marker + a short tail, so e.g. a build tool re-printing
    the same "Waiting..." line 200 times doesn't burn tokens."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        j = i
        while j + 1 < len(lines) and lines[j + 1] == lines[i]:
            j += 1
        run_len = j - i + 1
        if run_len >= min_run:
            out.extend(lines[i : i + head])
            omitted = run_len - head - tail
            if omitted > 0:
                out.append(f"[{omitted} repeated lines omitted]")
                out.extend(lines[j - tail + 1 : j + 1])
            else:
                out.extend(lines[i : j + 1])
        else:
            out.extend(lines[i : j + 1])
        i = j + 1
    return "\n".join(out)


def _compress_tool_output(text: str) -> str:
    text = _ANSI_RE.sub("", text)
    lines = [ln for ln in text.split("\n") if not _PROGRESS_RE.match(ln)]
    return _collapse_repeated_lines("\n".join(lines))


def _compress_prose(text: str) -> str:
    for phrase in _FILLER_PHRASES:
        text = re.sub(r"^" + re.escape(phrase), "", text)
        # After sentence-ending punctuation + a space, OR after a bare
        # newline (no trailing space needed — e.g. right after a protected
        # code-block placeholder followed directly by a new paragraph).
        text = re.sub(r"([.!?] )" + re.escape(phrase), r"\1", text)
        text = re.sub(r"(\n)" + re.escape(phrase), r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass
class Compressor:
    """Config-driven, provider-scoped compressor. One instance is shared by
    the router service and bound into :class:`ProviderGateway`
    (``bind_compressor``)."""

    enabled: bool = True
    apply_to: tuple[str, ...] = ("tool_output", "prose")
    min_chars_to_compress: int = 500
    per_provider: dict[str, dict[str, Any]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _totals: dict[str, list[int]] = field(default_factory=dict)  # provider -> [orig_chars, compressed_chars]

    def enabled_for(self, provider_id: str) -> bool:
        override = self.per_provider.get(provider_id, {})
        return bool(override.get("enabled", self.enabled))

    def compress_text(self, text: str, kind: str) -> tuple[str, CompressionStats]:
        """Compress ``text`` if ``kind`` is in ``apply_to`` and it's long
        enough to be worth it; otherwise returns it unchanged."""
        original_chars = len(text)
        if kind not in self.apply_to or original_chars < self.min_chars_to_compress:
            return text, CompressionStats(original_chars, original_chars)

        protected, code_tokens = _protect(text, _CODE_FENCE_RE, "CODE")
        protected, url_tokens = _protect(protected, _URL_RE, "URL")

        if kind == "tool_output":
            protected = _compress_tool_output(protected)
        elif kind == "prose":
            protected = _compress_prose(protected)

        restored = _restore(_restore(protected, url_tokens), code_tokens)
        return restored, CompressionStats(original_chars, len(restored))

    def compress_for_provider(
        self, provider_id: str, text: str, kind: str
    ) -> tuple[str, CompressionStats]:
        """Like ``compress_text``, but honors the per-provider enable
        override and accumulates rolling stats for ``stats_for``."""
        if not self.enabled_for(provider_id):
            n = len(text)
            return text, CompressionStats(n, n)
        compressed, stats = self.compress_text(text, kind)
        with self._lock:
            totals = self._totals.setdefault(provider_id, [0, 0])
            totals[0] += stats.original_chars
            totals[1] += stats.compressed_chars
        return compressed, stats

    def stats_for(self, provider_id: str) -> dict[str, Any]:
        orig, compressed = self._totals.get(provider_id, [0, 0])
        ratio = 0.0 if orig == 0 else round(1.0 - (compressed / orig), 4)
        return {"original_chars": orig, "compressed_chars": compressed, "ratio": ratio}

    def stats_all(self) -> dict[str, dict[str, Any]]:
        return {pid: self.stats_for(pid) for pid in self._totals}

    @classmethod
    def from_config(cls, compression: Optional[dict[str, Any]]) -> "Compressor":
        c = compression or {}
        return cls(
            enabled=bool(c.get("enabled", True)),
            apply_to=tuple(c.get("apply_to", ["tool_output", "prose"])),
            min_chars_to_compress=int(c.get("min_chars_to_compress", 500)),
            per_provider=dict(c.get("per_provider", {}) or {}),
        )
