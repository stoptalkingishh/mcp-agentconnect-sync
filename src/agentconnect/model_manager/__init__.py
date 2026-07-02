"""Local Model Manager — the local inference control plane (handoff §4.2)."""

from .residency import ResidencyManager  # noqa: F401
from .backends import ModelBackend, StubBackend  # noqa: F401

__all__ = ["ResidencyManager", "ModelBackend", "StubBackend"]
