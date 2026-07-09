"""Linear adapter — the human-visible mirror and control surface (spec §14).

AgentConnect stays canonical. Push sync writes issues and compact comments;
webhook ingest turns human actions into approvals, events, and inbox items.
"""

from .client import LinearClient, LinearError, Transport
from .sync import PROVIDER, LinearSync
from .webhooks import ApprovalCommand, WebhookEvent, handle_webhook, parse_command, parse_webhook

__all__ = [
    "ApprovalCommand", "LinearClient", "LinearError", "LinearSync", "PROVIDER", "Transport",
    "WebhookEvent", "handle_webhook", "parse_command", "parse_webhook",
]
