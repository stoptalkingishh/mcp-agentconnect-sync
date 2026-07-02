"""Privacy classification & redaction layer (handoff §13, §14, §24).

Two responsibilities:

1. Classify a task payload into a :class:`PrivacyClass`.
2. Run a redaction/policy pass before any external provider call: detect and
   redact secrets/PII, and decide whether the (redacted) payload is cloud-safe.

The detectors are deliberately conservative — when in doubt the caller should
route local (§14: "If redaction is too destructive, route local instead").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .schemas import PrivacyClass, RedactionResult

# --- Denylisted sensitive file markers (§24) --------------------------------
SENSITIVE_FILE_PATTERNS = [
    r"\.env(\.|$)",
    r"\.ssh/",
    r"\.aws/",
    r"\.gcp/",
    r"\.pem\b",
    r"\.key\b",
    r"\bsecrets\.",
    r"\bcredentials\.",
]

# --- Secret / PII detectors --------------------------------------------------
# (label, compiled regex, replacement) — order matters; broad ones last.
_DETECTORS: list[tuple[str, re.Pattern, str]] = [
    ("removed OpenAI-style API key", re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"), "[REDACTED_API_KEY]"),
    ("removed Google API key", re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"), "[REDACTED_API_KEY]"),
    ("removed AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    ("removed GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "[REDACTED_TOKEN]"),
    ("removed JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"), "[REDACTED_JWT]"),
    ("removed SSH private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"), "[REDACTED_PRIVATE_KEY]"),
    ("removed database URL", re.compile(r"\b(?:postgres|postgresql|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s\"']+"), "[REDACTED_DB_URL]"),
    ("removed bearer token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"), "Bearer [REDACTED_TOKEN]"),
    ("removed generic secret assignment", re.compile(r"(?i)\b(api[_-]?key|secret|password|passwd|token)\s*[:=]\s*[\"']?[A-Za-z0-9._\-]{8,}[\"']?"), r"\1=[REDACTED]"),
    ("removed customer email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    ("removed internal hostname", re.compile(r"\b[a-z0-9\-]+\.(?:local|internal|corp|lan)\b"), "[REDACTED_HOST]"),
    ("removed absolute local path", re.compile(r"(?:/home|/Users)/[A-Za-z0-9._\-/]+"), "[REDACTED_PATH]"),
]

# Detectors whose *mere presence* means the payload contains a hard secret and
# must never go to an LLM (§13 secret_sensitive).
_HARD_SECRET_LABELS = {
    "removed OpenAI-style API key",
    "removed Google API key",
    "removed AWS access key id",
    "removed GitHub token",
    "removed JWT",
    "removed SSH private key",
    "removed database URL",
}


@dataclass
class ClassificationHints:
    """Signals the caller can pass to bias classification."""

    file_paths: tuple[str, ...] = ()
    from_private_repo: bool = False
    declared: Optional[PrivacyClass] = None


def classify(payload: str, hints: Optional[ClassificationHints] = None) -> PrivacyClass:
    """Best-effort deterministic privacy classification (§13)."""
    hints = hints or ClassificationHints()
    if hints.declared is not None:
        return hints.declared

    # Denylisted sensitive files => secret_sensitive.
    for path in hints.file_paths:
        for pat in SENSITIVE_FILE_PATTERNS:
            if re.search(pat, path):
                return PrivacyClass.secret_sensitive

    # Hard secrets present in the body => secret_sensitive.
    for label, rx, _ in _DETECTORS:
        if label in _HARD_SECRET_LABELS and rx.search(payload):
            return PrivacyClass.secret_sensitive

    if hints.from_private_repo:
        return PrivacyClass.repo_sensitive

    # Soft PII (emails, internal hosts, local paths) => low_sensitive.
    for label, rx, _ in _DETECTORS:
        if label not in _HARD_SECRET_LABELS and rx.search(payload):
            return PrivacyClass.low_sensitive

    return PrivacyClass.public


def redact(payload: str, privacy_class: PrivacyClass) -> tuple[RedactionResult, str]:
    """Run the redaction/policy pass before an external call (§14).

    Returns ``(result, redacted_text)``. ``result.cloud_safe`` is False when a
    hard secret was present (caller should route local or block). The caller is
    responsible for storing ``redacted_text`` in shared memory and setting
    ``result.payload_ref`` to the artifact id.
    """
    redactions: list[str] = []
    contains_hard_secret = False
    out = payload
    for label, rx, repl in _DETECTORS:
        if rx.search(out):
            if label in _HARD_SECRET_LABELS:
                contains_hard_secret = True
            out, n = rx.subn(repl, out)
            if n:
                redactions.append(f"{label} (x{n})")

    # Estimate lossiness by how much we changed.
    changed = sum(int(r.split("x")[-1].rstrip(")")) for r in redactions) if redactions else 0
    if changed == 0:
        lossiness = "none"
    elif changed <= 2:
        lossiness = "low"
    elif changed <= 6:
        lossiness = "medium"
    else:
        lossiness = "high"

    # secret_sensitive must never reach an LLM even after redaction.
    cloud_safe = (
        privacy_class not in (PrivacyClass.secret_sensitive,)
        and not contains_hard_secret
        and privacy_class != PrivacyClass.restricted
    )

    result = RedactionResult(
        cloud_safe=cloud_safe,
        privacy_class=privacy_class,
        redactions=redactions,
        payload_ref=None,  # caller stores `out` in shared memory and sets the ref
        lossiness=lossiness,
    )
    return result, out
