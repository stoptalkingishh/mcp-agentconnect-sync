"""Evaluation & learning (handoff §25 Phase 6).

Turns the raw outcome records in shared memory into per-provider **scorecards**
(success rate, latency, cost, confidence, sample count) and a bounded, deterministic
**learned-quality signal** the router folds into scoring. This closes the loop:
deterministic routing chooses/constrains/verifies, while observed outcomes nudge
future provider preference — without introducing randomness into the control plane.

The signal is intentionally conservative: it stays at 0 until a provider has at
least ``min_samples`` observations, and it is clamped to [-1, 1] so a learned prior
can tilt a close call but never override hard constraints or dominate scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .memory import SharedMemory


@dataclass(frozen=True)
class Scorecard:
    provider: str
    samples: int
    success_rate: float  # [0,1]
    avg_latency_ms: float
    avg_cost_usd: float
    avg_confidence: Optional[float]

    def learned_quality(self, min_samples: int = 5, fastest_latency_ms: float = 0.0,
                        slowest_latency_ms: float = 0.0) -> float:
        """A bounded [-1,1] adjustment. Combines success rate (primary) with a
        small relative-latency term. Zero until enough samples accumulate."""
        if self.samples < min_samples:
            return 0.0
        # Success rate centered on 0.5 -> [-1, 1].
        success_term = (self.success_rate - 0.5) * 2.0
        # Relative latency: faster than the field is a small bonus.
        latency_term = 0.0
        span = slowest_latency_ms - fastest_latency_ms
        if span > 1e-6:
            # 0 (slowest) .. 1 (fastest) -> map to [-0.3, 0.3]
            rel = 1.0 - (self.avg_latency_ms - fastest_latency_ms) / span
            latency_term = (rel - 0.5) * 0.6
        return max(-1.0, min(1.0, 0.85 * success_term + 0.15 * latency_term))


class Evaluator:
    """Reads outcome records from shared memory and produces scorecards."""

    def __init__(self, memory: SharedMemory, min_samples: int = 5):
        self.memory = memory
        self.min_samples = min_samples

    def scorecards(self) -> dict[str, Scorecard]:
        out: dict[str, Scorecard] = {}
        for row in self.memory.provider_eval_aggregate():
            samples = row["samples"] or 0
            successes = row["successes"] or 0
            out[row["provider"]] = Scorecard(
                provider=row["provider"],
                samples=samples,
                success_rate=(successes / samples) if samples else 0.0,
                avg_latency_ms=row["avg_latency_ms"] or 0.0,
                avg_cost_usd=row["avg_cost_usd"] or 0.0,
                avg_confidence=row["avg_confidence"],
            )
        return out

    def learned_quality(self) -> dict[str, float]:
        """Map provider -> bounded learned-quality signal, normalized across the
        current field's latency range."""
        cards = self.scorecards()
        eligible = [c for c in cards.values() if c.samples >= self.min_samples]
        if not eligible:
            return {p: 0.0 for p in cards}
        fastest = min(c.avg_latency_ms for c in eligible)
        slowest = max(c.avg_latency_ms for c in eligible)
        return {
            p: c.learned_quality(self.min_samples, fastest, slowest)
            for p, c in cards.items()
        }
