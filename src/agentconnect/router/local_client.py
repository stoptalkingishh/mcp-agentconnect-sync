"""Client interface to the Local Model Manager (handoff §5).

The router needs deterministic visibility into local inference status. It talks
to the Local Model Manager through this interface, which has two implementations:

  * :class:`InProcessLocalClient` — wraps a :class:`ResidencyManager` directly.
    Used in tests and single-box deployments; no network, fully deterministic.
  * :class:`HttpLocalClient` — calls the Model Manager's HTTP API over the
    network, resolving its bearer token from the secrets manager.
"""

from __future__ import annotations

import abc
from typing import Optional

from ..common.schemas import (
    CanAcceptRequest,
    CanAcceptResponse,
    GenerateRequest,
    GenerateResponse,
    LoadRequest,
    LoadResponse,
    ManagerStatus,
)


class LocalClient(abc.ABC):
    @abc.abstractmethod
    def status(self) -> ManagerStatus: ...

    @abc.abstractmethod
    def can_accept(self, req: CanAcceptRequest) -> CanAcceptResponse: ...

    @abc.abstractmethod
    def generate(self, req: GenerateRequest) -> GenerateResponse: ...

    @abc.abstractmethod
    def load(self, req: LoadRequest) -> LoadResponse: ...


class InProcessLocalClient(LocalClient):
    def __init__(self, manager):  # manager: ResidencyManager
        self._m = manager

    def status(self) -> ManagerStatus:
        return self._m.status()

    def can_accept(self, req: CanAcceptRequest) -> CanAcceptResponse:
        return self._m.can_accept(req)

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        return self._m.generate(req)

    def load(self, req: LoadRequest) -> LoadResponse:
        return self._m.load(req)


class HttpLocalClient(LocalClient):
    """Talks to a remote Local Model Manager. Resolves the bearer token from the
    secrets manager at construction — the token never crosses into agent state."""

    def __init__(self, base_url: str, token: Optional[str] = None, timeout: float = 30.0):
        import httpx

        self._base = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = httpx.Client(base_url=self._base, headers=headers, timeout=timeout)

    def status(self) -> ManagerStatus:
        r = self._client.get("/status")
        r.raise_for_status()
        return ManagerStatus.model_validate(r.json())

    def can_accept(self, req: CanAcceptRequest) -> CanAcceptResponse:
        r = self._client.post("/can_accept", json=req.model_dump(mode="json"))
        r.raise_for_status()
        return CanAcceptResponse.model_validate(r.json())

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        r = self._client.post("/generate", json=req.model_dump(mode="json"))
        r.raise_for_status()
        return GenerateResponse.model_validate(r.json())

    def load(self, req: LoadRequest) -> LoadResponse:
        r = self._client.post("/load", json=req.model_dump(mode="json"))
        r.raise_for_status()
        return LoadResponse.model_validate(r.json())
