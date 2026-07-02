"""Provider registry access + capability/health helpers (handoff §6).

Thin wrapper over the loaded provider config that the router queries during
eligibility and scoring. Health is tracked in-process (deterministic, no network
in the core); a real deployment would populate it from probes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from .config import ProviderConfig, ProviderRegistryConfig, load_providers


@dataclass
class ProviderRegistry:
    config: ProviderRegistryConfig
    _health: dict[str, str] = field(default_factory=dict)  # provider_id -> healthy|degraded|down

    @classmethod
    def from_config(cls, config: Optional[ProviderRegistryConfig] = None) -> "ProviderRegistry":
        return cls(config=config or load_providers())

    @property
    def policy_version(self) -> str:
        return self.config.policy_version

    def all(self) -> list[ProviderConfig]:
        return list(self.config.providers.values())

    def get(self, provider_id: str) -> Optional[ProviderConfig]:
        return self.config.providers.get(provider_id)

    def by_type(self, type_: str) -> list[ProviderConfig]:
        return [p for p in self.all() if p.type == type_]

    def with_capability(self, capability: str) -> list[ProviderConfig]:
        return [p for p in self.all() if capability in p.capabilities]

    # --------------------------------------------------------------- health
    def set_health(self, provider_id: str, status: str) -> None:
        self._health[provider_id] = status

    def health(self, provider_id: str) -> str:
        return self._health.get(provider_id, "healthy")

    def is_available(self, provider_id: str) -> bool:
        return self.health(provider_id) != "down"

    def capability_overlap(self, cfg: ProviderConfig, needed: Iterable[str]) -> float:
        needed = list(needed)
        if not needed:
            return 1.0
        hits = sum(1 for c in needed if c in cfg.capabilities)
        return hits / len(needed)
