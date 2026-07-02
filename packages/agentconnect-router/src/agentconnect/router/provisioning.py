"""Node provisioning for rented GPU inference (handoff Goal 4).

A rented GPU box runs the **same** ``agentconnect-model-manager`` as an owned node
and is reached over the **same mutual-TLS transport**. The only new machinery is
lifecycle: rent -> wait-for-ready -> (use) -> drain -> terminate.

This module defines the provisioner interface plus a deterministic
:class:`StubProvisioner` so the rented-node path is testable offline. Real vendor
adapters (RunPod / Lambda / Vast) implement :class:`NodeProvisioner` and read the
vendor's control-plane API key from the secrets manager via
``RentalConfig.secret_ref`` — that key is the ONLY secret involved, and it is used
purely to rent/terminate the box, never for inference traffic.
"""

from __future__ import annotations

import abc
import time
from typing import Any, Optional

from ..common.config import ProviderConfig
from ..common.schemas import NodeHandle, NodeSpec, NodeState, NodeTrust


def spec_from_provider(cfg: ProviderConfig, model_id: Optional[str] = None) -> NodeSpec:
    """Build a :class:`NodeSpec` from a rented provider's config entry."""
    rental = cfg.rental
    trust = NodeTrust(**(rental.trust if rental and rental.trust else {}))
    return NodeSpec(
        provider_id=cfg.provider_id,
        vendor=rental.vendor if rental else "generic",
        instance_type=rental.instance_type if rental else None,
        model_id=model_id,
        min_rental_seconds=rental.min_rental_seconds if rental else 900,
        max_hourly_usd=rental.max_hourly_usd if rental else 0.0,
        trust=trust,
    )


class NodeProvisioner(abc.ABC):
    """Lifecycle control for an inference node. Owned nodes are always-on and do not
    need a provisioner; rented nodes do."""

    @abc.abstractmethod
    def provision(self, spec: NodeSpec) -> NodeHandle: ...

    @abc.abstractmethod
    def wait_ready(self, handle: NodeHandle, timeout_seconds: int = 600) -> NodeHandle: ...

    @abc.abstractmethod
    def drain(self, handle: NodeHandle) -> NodeHandle: ...

    @abc.abstractmethod
    def terminate(self, handle: NodeHandle) -> NodeHandle: ...


class StubProvisioner(NodeProvisioner):
    """Deterministic in-memory provisioner for development and tests.

    ``provision`` returns a ``ready`` handle immediately (no real box, no network,
    no randomness). ``clock`` is injectable so ``started_at`` is deterministic in
    tests; it defaults to a fixed epoch rather than wall-clock time.
    """

    def __init__(self, endpoint_template: str = "https://rented-{pid}.local:8443", clock: float = 0.0):
        self._endpoint_template = endpoint_template
        self._clock = clock
        self._counter = 0

    def provision(self, spec: NodeSpec) -> NodeHandle:
        self._counter += 1
        node_id = f"{spec.provider_id}-rented-{self._counter:03d}"
        return NodeHandle(
            node_id=node_id,
            provider_id=spec.provider_id,
            state=NodeState.ready,
            manager_endpoint=self._endpoint_template.format(pid=spec.provider_id),
            started_at=self._clock,
            hourly_usd=spec.max_hourly_usd,
            trust=spec.trust,
        )

    def wait_ready(self, handle: NodeHandle, timeout_seconds: int = 600) -> NodeHandle:
        return handle.model_copy(update={"state": NodeState.ready})

    def drain(self, handle: NodeHandle) -> NodeHandle:
        return handle.model_copy(update={"state": NodeState.draining})

    def terminate(self, handle: NodeHandle) -> NodeHandle:
        return handle.model_copy(update={"state": NodeState.terminated})


class RunPodProvisioner(NodeProvisioner):
    """Real adapter for RunPod's REST API (https://rest.runpod.io/v1).

    Rents a pod that runs your ``agentconnect-model-manager`` image and exposes it
    over mTLS. The RunPod API key (the ONLY secret here) is passed in by the caller
    after resolving ``rental.secret_ref`` from the secrets manager — it is used
    solely for the control plane, never for inference traffic. ``client`` is an
    injectable ``httpx.Client`` so this is testable offline with a mock transport.
    """

    def __init__(
        self,
        api_key: str,
        image: str = "agentconnect/model-manager:latest",
        base_url: str = "https://rest.runpod.io/v1",
        manager_port: int = 8443,
        client: Any = None,
        poll_interval: float = 5.0,
        sleep=time.sleep,
    ):
        self._image = image
        self._manager_port = manager_port
        self._poll = poll_interval
        self._sleep = sleep
        if client is not None:
            self._client = client
        else:
            import httpx

            self._client = httpx.Client(
                base_url=base_url.rstrip("/"),
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=60.0,
            )

    def provision(self, spec: NodeSpec) -> NodeHandle:
        body = {
            "name": f"agentconnect-{spec.provider_id}",
            "imageName": self._image,
            "gpuTypeIds": [spec.instance_type] if spec.instance_type else [],
            "ports": [f"{self._manager_port}/tcp"],
            "env": {"MODEL_MANAGER_TLS_MODE": "mutual"},
        }
        r = self._client.post("/pods", json=body)
        r.raise_for_status()
        data = r.json()
        return NodeHandle(
            node_id=str(data.get("id") or data.get("podId") or f"{spec.provider_id}-pod"),
            provider_id=spec.provider_id,
            state=NodeState.provisioning,
            manager_endpoint=self._endpoint(data),
            hourly_usd=spec.max_hourly_usd,
            trust=spec.trust,
        )

    def wait_ready(self, handle: NodeHandle, timeout_seconds: int = 600) -> NodeHandle:
        deadline_polls = max(1, timeout_seconds // max(1, int(self._poll)))
        for _ in range(deadline_polls):
            r = self._client.get(f"/pods/{handle.node_id}")
            r.raise_for_status()
            data = r.json()
            status = (data.get("desiredStatus") or data.get("status") or "").upper()
            if status == "RUNNING":
                return handle.model_copy(
                    update={"state": NodeState.ready, "manager_endpoint": self._endpoint(data)}
                )
            self._sleep(self._poll)
        raise TimeoutError(f"RunPod node {handle.node_id} not RUNNING within {timeout_seconds}s")

    def drain(self, handle: NodeHandle) -> NodeHandle:
        return handle.model_copy(update={"state": NodeState.draining})

    def terminate(self, handle: NodeHandle) -> NodeHandle:
        r = self._client.delete(f"/pods/{handle.node_id}")
        # Idempotent: a already-gone pod is fine.
        if r.status_code not in (200, 202, 204, 404):
            r.raise_for_status()
        return handle.model_copy(update={"state": NodeState.terminated})

    def _endpoint(self, data: dict) -> Optional[str]:
        host = data.get("publicIp") or data.get("ip")
        # RunPod may expose a proxy host like {id}-{port}.proxy.runpod.net.
        if not host and data.get("id"):
            host = f"{data['id']}-{self._manager_port}.proxy.runpod.net"
            return f"https://{host}"
        if host:
            return f"https://{host}:{self._manager_port}"
        return None


class NodePool:
    """Keeps rented nodes warm across tasks so a batch amortizes one spin-up.

    ``acquire`` reuses a ready node for the provider if one is up, else provisions
    a fresh one (returning ``reused=False`` so the caller bills the rental window
    once). ``reap_idle`` terminates nodes idle past their configured window. Thread
    -safe; ``clock`` is injectable for deterministic tests.
    """

    def __init__(self):
        import threading

        self._lock = threading.Lock()
        self._live: dict[str, NodeHandle] = {}  # provider_id -> handle
        self._last_used: dict[str, float] = {}

    def acquire(
        self, cfg: ProviderConfig, provisioner: NodeProvisioner, spec: NodeSpec, now: float = 0.0
    ) -> tuple[NodeHandle, bool]:
        with self._lock:
            existing = self._live.get(cfg.provider_id)
            if existing is not None and existing.state == NodeState.ready:
                self._last_used[cfg.provider_id] = now
                return existing, True
        handle = provisioner.wait_ready(provisioner.provision(spec))
        with self._lock:
            self._live[cfg.provider_id] = handle
            self._last_used[cfg.provider_id] = now
        return handle, False

    def release(self, cfg: ProviderConfig, now: float = 0.0) -> None:
        with self._lock:
            if cfg.provider_id in self._live:
                self._last_used[cfg.provider_id] = now

    def reap_idle(
        self, provisioner: NodeProvisioner, cfgs: dict[str, ProviderConfig], now: float
    ) -> list[str]:
        """Terminate nodes idle beyond terminate_when_idle_seconds. Returns the
        provider ids reaped."""
        reaped: list[str] = []
        with self._lock:
            items = list(self._live.items())
        for pid, handle in items:
            cfg = cfgs.get(pid)
            idle_cap = cfg.rental.terminate_when_idle_seconds if (cfg and cfg.rental) else 600
            if now - self._last_used.get(pid, now) >= idle_cap:
                provisioner.terminate(handle)
                with self._lock:
                    self._live.pop(pid, None)
                    self._last_used.pop(pid, None)
                reaped.append(pid)
        return reaped

    def live_nodes(self) -> dict[str, NodeHandle]:
        with self._lock:
            return dict(self._live)


def provisioner_for(
    cfg: ProviderConfig, secret_resolver=None, client: Any = None
) -> NodeProvisioner:
    """Select a provisioner for a rented provider by ``rental.vendor``.

    ``generic`` -> StubProvisioner (offline/manual). A real vendor resolves its
    control-plane key from the secrets manager via ``secret_resolver``."""
    vendor = (cfg.rental.vendor if cfg.rental else "generic").lower()
    if vendor in ("generic", "", "stub"):
        return StubProvisioner()
    if vendor == "runpod":
        if secret_resolver is None or not (cfg.rental and cfg.rental.secret_ref):
            raise RuntimeError("RunPod provisioner needs a secret_resolver and rental.secret_ref")
        api_key = secret_resolver.resolve(cfg.rental.secret_ref)
        return RunPodProvisioner(api_key=api_key, client=client)
    raise RuntimeError(f"No provisioner adapter for vendor {vendor!r}")
