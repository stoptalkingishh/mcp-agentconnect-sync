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

def _discover_config_dir() -> Path:
    """Locate the repo's ``config/`` directory.

    Precedence:
      1. ``AGENTCONNECT_CONFIG_DIR`` env override (explicit; used in deployments).
      2. An upward search from the current working directory for a directory that
         contains ``config/providers.yaml`` (works from a source checkout regardless
         of where the packages live).
      3. An upward search from this file's location (works when installed alongside
         the repo layout).
      4. ``./config`` as a last resort.

    The old ``parents[3]`` repo-root assumption no longer holds now that ``common``
    lives under ``packages/agentconnect-core/src/agentconnect/common``.
    """
    env = os.environ.get("AGENTCONNECT_CONFIG_DIR")
    if env:
        return Path(env)
    for start in (Path.cwd(), Path(__file__).resolve()):
        for base in (start, *start.parents):
            if (base / "config" / "providers.yaml").exists():
                return base / "config"
    return Path.cwd() / "config"


CONFIG_DIR = _discover_config_dir()


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass(frozen=True)
class TlsClientConfig:
    """Client-side mTLS material for reaching a local inference node (handoff §7).

    All fields are filesystem PATHS provided by the platform (cert-manager /
    SPIFFE / secrets-manager sync) — never inline cert/key material. ``${ENV}``
    references are expanded at load time.
    """

    mode: str = "mutual"  # mutual | insecure_localhost
    ca_cert: str | None = None
    client_cert: str | None = None
    client_key: str | None = None
    server_name: str | None = None


def client_ssl_context(tls: "TlsClientConfig | None") -> "ssl.SSLContext | None":
    """SSLContext for mutual-TLS clients: pin the server cert to the private CA
    and present the client cert. Returns ``None`` for ``tls=None`` or
    ``mode != "mutual"`` — plain HTTP, loopback/dev only."""
    import ssl

    if tls is None or tls.mode != "mutual":
        return None
    ctx = (
        ssl.create_default_context(cafile=tls.ca_cert)
        if tls.ca_cert
        else ssl.create_default_context()
    )
    if tls.client_cert and tls.client_key:
        ctx.load_cert_chain(certfile=tls.client_cert, keyfile=tls.client_key)
    return ctx


@dataclass(frozen=True)
class RentalConfig:
    """Lifecycle + cost + trust settings for a rented GPU node (handoff Goal 4).

    ``secret_ref`` here is the rental VENDOR's control-plane API key — a genuine
    third-party secret that lives in the secrets manager and is used only by the
    provisioner. It is NOT used for inference traffic (that is mTLS).
    """

    vendor: str = "generic"  # generic | runpod | lambda | vast
    secret_ref: str | None = None
    instance_type: str | None = None
    min_rental_seconds: int = 900
    max_hourly_usd: float = 0.0
    max_daily_usd: float = 0.0
    terminate_when_idle_seconds: int = 600
    trust: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CliConfig:
    """Non-interactive subprocess invocation settings for a locally installed,
    subscription-authenticated coding-agent CLI (``type == "cli_subprocess"``
    providers -- e.g. Claude Code, Codex).

    No ``secret_ref``: authentication is whatever the OS user already has the
    CLI logged into (OAuth/session state under the CLI's own config dir) --
    a third, distinct trust category from mTLS (local nodes) and
    ``secret_ref`` (cloud providers), see ``config/secrets.yaml``'s header
    note. The router never resolves or sees a credential for this provider
    type; it can't rotate or scope it either, which is why callers must not
    assume it behaves like the other two.

    ``args`` are the full non-interactive invocation flags, NOT including the
    prompt itself -- the prompt is always piped via stdin (both CLIs support
    this; it also sidesteps a real bug found live-testing this: Claude
    Code's ``--tools <tools...>`` is a greedy variadic flag that silently
    swallows a trailing positional prompt argument). ``args`` must never
    include a workspace/write-access flag (``--add-dir``, Codex's
    ``-s workspace-write``/``danger-full-access``) -- that lockdown is
    enforced here in config, not left as a per-call option downstream in the
    gateway, so this provider's privacy tier stays true to what the invoked
    process can actually touch on disk.

    ``cwd`` is a fixed, neutral working directory -- deliberately never the
    router process's own cwd (which could be a real repo checkout). Even a
    read-only sandbox mode still lets the CLI *read* whatever's in its
    working directory; pinning ``cwd`` away from real project content is
    what actually prevents that, not the sandbox flag alone (confirmed by
    live-testing ``codex exec`` without ``-C``: it defaulted to the
    launching process's cwd).
    """

    binary: str
    args: tuple[str, ...] = ()
    output_mode: str = "stdout"  # "stdout_json" | "output_file"
    output_file_flag: str | None = None  # e.g. "-o"; required when output_mode == "output_file"
    cwd: str | None = None
    timeout_seconds: float = 120.0
    max_output_chars: int = 20000


@dataclass(frozen=True)
class ProviderConfig:
    provider_id: str
    type: str
    endpoint: str
    privacy: str  # local_only | private_rented | external | external_paid
    capabilities: tuple[str, ...]
    secret_ref: str | None = None  # cloud-only now; local nodes authenticate via mTLS
    quota: dict[str, Any] = field(default_factory=dict)
    manager_endpoint: str | None = None
    node_id: str | None = None
    node_class: str | None = None  # owned | rented (for type == "local")
    tls: TlsClientConfig | None = None
    rental: RentalConfig | None = None
    model_map: dict[str, str] = field(default_factory=dict)
    model_prefix: str = ""
    cli: CliConfig | None = None  # for type == "cli_subprocess" only


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
    def auto_retrieval(self) -> dict[str, Any]:
        return self.raw.get("auto_retrieval", {})

    @property
    def scoring(self) -> dict[str, Any]:
        return self.raw.get("scoring", {})

    @property
    def resilience(self) -> dict[str, Any]:
        return self.raw.get("resilience", {})

    @property
    def compression(self) -> dict[str, Any]:
        return self.raw.get("compression", {})


def _expand(value: str | None) -> str | None:
    return os.path.expandvars(value) if value else value


def _parse_tls(raw: dict[str, Any] | None) -> TlsClientConfig | None:
    if not raw:
        return None
    return TlsClientConfig(
        mode=raw.get("mode", "mutual"),
        ca_cert=_expand(raw.get("ca_cert")),
        client_cert=_expand(raw.get("client_cert")),
        client_key=_expand(raw.get("client_key")),
        server_name=raw.get("server_name"),
    )


def _parse_rental(raw: dict[str, Any] | None) -> RentalConfig | None:
    if not raw:
        return None
    return RentalConfig(
        vendor=raw.get("vendor", "generic"),
        secret_ref=raw.get("secret_ref"),
        instance_type=raw.get("instance_type"),
        min_rental_seconds=int(raw.get("min_rental_seconds", 900)),
        max_hourly_usd=float(raw.get("max_hourly_usd", 0.0)),
        max_daily_usd=float(raw.get("max_daily_usd", 0.0)),
        terminate_when_idle_seconds=int(raw.get("terminate_when_idle_seconds", 600)),
        trust=raw.get("trust", {}) or {},
    )


def _parse_cli(raw: dict[str, Any] | None) -> CliConfig | None:
    if not raw:
        return None
    return CliConfig(
        binary=_expand(raw["binary"]),
        args=tuple(raw.get("args", [])),
        output_mode=raw.get("output_mode", "stdout"),
        output_file_flag=raw.get("output_file_flag"),
        cwd=_expand(raw.get("cwd")),
        timeout_seconds=float(raw.get("timeout_seconds", 120.0)),
        max_output_chars=int(raw.get("max_output_chars", 20000)),
    )


def load_providers() -> ProviderRegistryConfig:
    data = _load_yaml("providers.yaml")
    providers: dict[str, ProviderConfig] = {}
    for pid, cfg in (data.get("providers") or {}).items():
        endpoint = cfg.get("endpoint", "")
        model_prefix = ""
        gateway = cfg.get("gateway", {}) or {}
        gateway_base = os.environ.get(gateway.get("base_url_env", ""), "").strip()
        if gateway_base:
            gateway_provider = gateway.get("provider", pid)
            endpoint = f"{gateway_base.rstrip('/')}/v1/providers/{gateway_provider}"
            model_prefix = f"{gateway_provider}/"
        providers[pid] = ProviderConfig(
            provider_id=pid,
            type=cfg["type"],
            endpoint=endpoint,
            secret_ref=cfg.get("secret_ref"),  # optional; local nodes use mTLS
            privacy=cfg["privacy"],
            capabilities=tuple(cfg.get("capabilities", [])),
            quota=cfg.get("quota", {}) or {},
            manager_endpoint=cfg.get("manager_endpoint"),
            node_id=cfg.get("node_id"),
            node_class=cfg.get("node_class"),
            tls=_parse_tls(cfg.get("tls")),
            rental=_parse_rental(cfg.get("rental")),
            model_map=dict(cfg.get("model_map", {}) or {}),
            model_prefix=model_prefix,
            cli=_parse_cli(cfg.get("cli")),
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


def load_workers() -> dict[str, str]:
    """Optional worker_id -> attested ``ProviderPrivacyTier`` map
    (``config/workers.yaml``, key ``workers``).

    Soft-optional, unlike ``load_providers``/``load_routing``: a missing file
    returns ``{}`` rather than raising, since a fresh checkout may not federate
    any pull workers yet. Fail-closed downstream — an identity absent from this
    map resolves to no attested tier, so the work-queue's admissible-class
    computation yields nothing and every claim is denied; this map can only
    ever grant a tier, never widen access beyond what routing.yaml's
    ``privacy.classes`` admits for that tier.
    """
    path = CONFIG_DIR / "workers.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return {str(k): str(v) for k, v in (data.get("workers") or {}).items()}


@lru_cache(maxsize=1)
def load_all() -> tuple[ProviderRegistryConfig, ProfilesConfig, RoutingConfig]:
    """Cached bundle of all config. Call `load_all.cache_clear()` in tests."""
    return load_providers(), load_profiles(), load_routing()
