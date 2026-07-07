#!/usr/bin/env python3
"""Register (or list) this router with MCP-compatible AI agent clients.

One Python helper shared cross-platform (instead of duplicating OS-specific
config-path logic in bash and PowerShell separately) -- invoked identically
from the end of both deploy/install.sh and deploy/install.ps1.

Usage:
    register_mcp.py list [--project-dir PATH]
    register_mcp.py add <client|all> [--dry-run] [--project-dir PATH]

Known clients: claude-code, claude-desktop, cursor, windsurf, cline
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Clients whose config-file location is a best-effort guess (varies by app
# version/build) rather than a documented, stable path. If the guessed
# directory doesn't already exist, we refuse to fabricate it and fall back to
# printing the manual snippet instead.
LOW_CONFIDENCE = {"cline"}


def venv_router_command() -> str:
    if os.name == "nt":
        return str(REPO_ROOT / ".venv" / "Scripts" / "agentconnect-router.exe")
    return str(REPO_ROOT / ".venv" / "bin" / "agentconnect-router")


def agentconnect_entry() -> dict:
    return {
        "command": venv_router_command(),
        "env": {"AGENTCONNECT_CONFIG_DIR": str(REPO_ROOT / "config")},
    }


def client_paths(project_dir: Path) -> dict[str, Path]:
    system = platform.system()
    home = Path.home()

    if system == "Windows":
        appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
        claude_desktop = appdata / "Claude" / "claude_desktop_config.json"
        vscode_storage = appdata / "Code" / "User" / "globalStorage"
    elif system == "Darwin":
        claude_desktop = home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        vscode_storage = home / "Library" / "Application Support" / "Code" / "User" / "globalStorage"
    else:
        claude_desktop = home / ".config" / "Claude" / "claude_desktop_config.json"
        vscode_storage = home / ".config" / "Code" / "User" / "globalStorage"

    return {
        "claude-code": project_dir / ".mcp.json",
        "claude-desktop": claude_desktop,
        "cursor": home / ".cursor" / "mcp.json",
        "windsurf": home / ".codeium" / "windsurf" / "mcp_config.json",
        "cline": vscode_storage / "saoudrizwan.claude-dev" / "settings" / "cline_mcp_settings.json",
    }


def print_manual_snippet() -> None:
    print(json.dumps({"mcpServers": {"agentconnect": agentconnect_entry()}}, indent=2))


def merge_write(path: Path, dry_run: bool) -> bool:
    entry = agentconnect_entry()

    if dry_run:
        print(f"--- {path} (dry run, not written) ---")
        print(json.dumps({"mcpServers": {"agentconnect": entry}}, indent=2))
        return True

    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict = {}
    if path.exists():
        backup = path.with_name(f"{path.name}.bak.{int(time.time())}")
        shutil.copy2(path, backup)
        try:
            # utf-8-sig transparently strips a BOM if present (common in JSON
            # files written by Windows tools/editors) and is a no-op otherwise.
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            print(f"WARNING: {path} is not valid JSON -- refusing to touch it. Unmodified backup at {backup}.")
            return False
        print(f"Backed up existing {path} -> {backup}")

    data.setdefault("mcpServers", {})
    data["mcpServers"]["agentconnect"] = entry
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Registered agentconnect in {path}")
    return True


def cmd_list(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir) if args.project_dir else REPO_ROOT
    paths = client_paths(project_dir)
    print(f"{'client':<16}{'found':<7}path")
    for client, path in paths.items():
        found = "yes" if path.exists() else "no"
        note = "  (best-effort path)" if client in LOW_CONFIDENCE else ""
        print(f"{client:<16}{found:<7}{path}{note}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir) if args.project_dir else REPO_ROOT
    paths = client_paths(project_dir)
    targets = list(paths) if args.client == "all" else [args.client]

    any_failed = False
    for client in targets:
        if client not in paths:
            print(f"Unknown client {client!r}. Known: {', '.join(paths)}")
            print("Manual snippet -- paste into your client's MCP config yourself:")
            print_manual_snippet()
            any_failed = True
            continue

        path = paths[client]
        if client in LOW_CONFIDENCE and not args.dry_run and not path.parent.parent.exists():
            print(f"{client}: config directory not found at a confident location ({path.parent.parent}).")
            print("Manual snippet -- paste into your client's MCP config yourself:")
            print_manual_snippet()
            continue

        if not merge_write(path, args.dry_run):
            any_failed = True

    return 1 if any_failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="Show known MCP clients, whether their config exists, and its path")
    p_list.add_argument("--project-dir", help="Project dir for claude-code's .mcp.json (default: repo root)")
    p_list.set_defaults(func=cmd_list)

    p_add = sub.add_parser("add", help="Register agentconnect with one or all known clients")
    p_add.add_argument("client", help="claude-code | claude-desktop | cursor | windsurf | cline | all")
    p_add.add_argument("--dry-run", action="store_true", help="Print the snippet/destination without writing")
    p_add.add_argument("--project-dir", help="Project dir for claude-code's .mcp.json (default: repo root)")
    p_add.set_defaults(func=cmd_add)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
