"""Worker result helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentconnect.common.schemas import WorkerResult


@dataclass
class RuntimeResult:
    summary: str = ""
    confidence: float = 0.0
    changed_artifacts: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    recommended_next_action: str | None = None


def build_worker_result(result: RuntimeResult) -> WorkerResult:
    return WorkerResult(
        summary=result.summary,
        confidence=result.confidence,
        changed_artifacts=list(result.changed_artifacts),
        evidence_refs=list(result.evidence_refs),
        risks=list(result.risks),
        recommended_next_action=result.recommended_next_action,
    )
