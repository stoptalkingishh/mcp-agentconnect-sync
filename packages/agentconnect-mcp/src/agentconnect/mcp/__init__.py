"""MCP adapter for the AgentConnect backplane (manager-facing tools only)."""

from .server import build_mcp_server, main

__all__ = ["build_mcp_server", "main"]
