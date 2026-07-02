"""Configuration loaders.

Loads the YAML files under `config/` into typed structures. The router and the
model manager both read from these; nothing here dereferences secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Repo root is three parents up from this file: src/agentconnect/common/config.py
_REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = Path(os.environ.get("AGENTCONNECT_CONFIG_DIR", _REPO_ROOT / "config"))


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass(frozen=True)
class ProviderConfig:
    provider_id: str
    type: str
    endpoint: str
    secret_ref: str
    privacy: str  # local_only | external | external_paid
    capabilities: tuple[str, ...]
    quota: dict[str, Any] = field(default_factory=dict)
    manager_endpoint: str | None = None
    node_id: str | None = None


@dataclass(frozen=True)
class ProviderRegistryConfig:
    policy_version: str
    providers: dict[str, ProviderConfig]


@dataclass(frozen=True)
class ProfilesConfig:
    default_resident_model: str
    profiles: dict[str, dict[str, Any]]
    agent_defaults: dict[str, str]


@dataclass(frozen=True)
class RoutingConfig:
    raw: dict[str, Any]

    @property
    def privacy(self) -> dict[str, Any]:
        return self.raw.get("privacy", {})

    @property
    def mcp_output_policy(self) -> dict[str, Any]:
        return self.raw.get("mcp_output_policy", {})

    @property
    def model_switching(self) -> dict[str, Any]:
        return self.raw.get("model_switching", {})

    @property
    def local_inference_defaults(self) -> dict[str, Any]:
        return self.raw.get("local_inference_defaults", {})

    @property
    def scoring(self) -> dict[str, Any]:
        return self.raw.get("scoring", {})


def load_providers() -> ProviderRegistryConfig:
    data = _load_yaml("providers.yaml")
    providers: dict[str, ProviderConfig] = {}
    for pid, cfg in (data.get("providers") or {}).items():
        providers[pid] = ProviderConfig(
            provider_id=pid,
            type=cfg["type"],
            endpoint=cfg["endpoint"],
            secret_ref=cfg["secret_ref"],
            privacy=cfg["privacy"],
            capabilities=tuple(cfg.get("capabilities", [])),
            quota=cfg.get("quota", {}) or {},
            manager_endpoint=cfg.get("manager_endpoint"),
            node_id=cfg.get("node_id"),
        )
    return ProviderRegistryConfig(
        policy_version=data.get("policy_version", "unknown"),
        providers=providers,
    )


def load_profiles() -> ProfilesConfig:
    data = _load_yaml("profiles.yaml")
    return ProfilesConfig(
        default_resident_model=data.get("default_resident_model", ""),
        profiles=data.get("profiles", {}) or {},
        agent_defaults=data.get("agent_defaults", {}) or {},
    )


def load_routing() -> RoutingConfig:
    return RoutingConfig(raw=_load_yaml("routing.yaml"))


@lru_cache(maxsize=1)
def load_all() -> tuple[ProviderRegistryConfig, ProfilesConfig, RoutingConfig]:
    """Cached bundle of all config. Call `load_all.cache_clear()` in tests."""
    return load_providers(), load_profiles(), load_routing()
