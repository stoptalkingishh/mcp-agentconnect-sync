"""Typed errors for the backplane.

Adapters map these onto their own vocabulary (HTTP status, MCP error text, CLI
exit code) — the service never raises protocol-specific errors.
"""

from __future__ import annotations


class AgentConnectError(Exception):
    """Base class for every error the service raises deliberately."""

    code = "agentconnect_error"


class NotFound(AgentConnectError):
    code = "not_found"


class Conflict(AgentConnectError):
    """The request is well-formed but loses a race or violates an invariant
    (e.g. a second primary_manager claim on a task that already has one)."""

    code = "conflict"


class PolicyViolation(AgentConnectError):
    """Refused by policy: privacy, authority, or approval — never by a bug."""

    code = "policy_violation"


class InvalidRequest(AgentConnectError):
    code = "invalid_request"
