"""Agent Router — the global control plane (handoff §4.1)."""

from .routing import RoutingContext, RoutingEngine  # noqa: F401
from .service import RouterService  # noqa: F401
from .gateway import ProviderGateway  # noqa: F401

__all__ = ["RoutingContext", "RoutingEngine", "RouterService", "ProviderGateway"]
