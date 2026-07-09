"""Task-scoped workspaces and the instructions injected into them (§4–§6, §10).

A workspace is the physical half of the compliance rule. It gives the agent a
place to work that AgentConnect knows about, so `agentconnect audit` can later ask
the only question that matters: *you changed these files — where did you record
that?*

Layout (§4)::

    ~/.agentconnect/workspaces/task_123/
      workspace.json      what this is, and where it came from
      .env.agentconnect   session identity + the one credential (0600)
      .mcp.json           AgentConnect MCP server, and nothing else
      AGENTCONNECT.md     the rules, always written
      CLAUDE.md           harness-specific, when the manager is Claude Code
      CODEX.md            harness-specific, when the manager is Codex
      bin/                helper scripts, prepended to PATH
      repo/               a git worktree where possible
      artifacts/
      logs/

Nothing here imports the service: a workspace is a directory and some text.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from .models import RepoMode, SessionMode

_log = logging.getLogger(__name__)

WORKSPACE_ENV = "AGENTCONNECT_WORKSPACE_DIR"
BRANCH_PREFIX = "agentconnect"

#: Exactly the tools a managed agent may see (§10). Anything a backend would
#: expose directly — Temporal, WikiBrain promotion, Cognee/Graphiti writes, the
#: local model manager, the secrets manager — is absent by construction.
EXPOSED_MCP_TOOLS: tuple[str, ...] = (
    "get_task_context_pack", "claim_task", "record_attempt", "record_decision",
    "submit_subtask", "get_subtask_status", "request_review", "list_artifacts",
    "read_artifact_chunk", "release_task",
)

#: Written into `.mcp.json` so the denial is auditable rather than implicit.
DENIED_MCP_TOOLS: tuple[str, ...] = (
    "temporal_signal", "wikibrain_promote", "cognee_write", "graphiti_write",
    "local_model_generate", "secrets_read",
)


def default_workspace_root() -> Path:
    env = os.environ.get(WORKSPACE_ENV)
    if env:
        return Path(env)
    return Path.home() / ".agentconnect" / "workspaces"


def slugify(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (slug[:limit].rstrip("-")) or "work"


def branch_name(entity_id: str, title: str) -> str:
    """`agentconnect/task_123/refactor-auth-expiry` (§5)."""
    return f"{BRANCH_PREFIX}/{entity_id}/{slugify(title)}"


# ------------------------------------------------------------------ git repos
def _git(args: list[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=str(cwd) if cwd else None, capture_output=True, text=True,
        check=False,
    )


def is_git_repo(path: Path) -> bool:
    if not path.is_dir():
        return False
    return _git(["rev-parse", "--is-inside-work-tree"], path).returncode == 0


def _add_worktree(source: Path, dest: Path, branch: str) -> bool:
    """`git worktree add -b <branch> <dest>`. Reuses the branch if it exists."""
    result = _git(["worktree", "add", "-b", branch, str(dest)], source)
    if result.returncode == 0:
        return True
    # A relaunch onto the same task finds its branch already there.
    retry = _git(["worktree", "add", str(dest), branch], source)
    if retry.returncode == 0:
        return True
    _log.warning("git worktree add failed (%s); falling back", result.stderr.strip())
    return False


def changed_files(repo_path: Path) -> list[str]:
    """Working-tree changes, plus anything committed on top of the branch point.

    The audit needs both: an agent that commits its work has still changed files
    that nobody registered as an artifact.
    """
    if not repo_path or not is_git_repo(repo_path):
        return []
    files: set[str] = set()
    status = _git(["status", "--porcelain"], repo_path)
    for line in status.stdout.splitlines():
        if len(line) > 3:
            # Rename lines read `R  old -> new`; the new path is what exists now.
            path = line[3:].split(" -> ")[-1].strip()
            if path:
                files.add(path)
    # Commits made on this worktree that the base branch does not have.
    diff = _git(["diff", "--name-only", "HEAD", "--"], repo_path)
    for line in diff.stdout.splitlines():
        if line.strip():
            files.add(line.strip())
    return sorted(files)


def committed_files(repo_path: Path, base: str = "HEAD~1") -> list[str]:
    result = _git(["diff", "--name-only", base, "HEAD"], repo_path)
    if result.returncode != 0:
        return []
    return sorted(f for f in result.stdout.splitlines() if f.strip())


# ------------------------------------------------------------------ materials
AGENTCONNECT_MD = """# AgentConnect Managed Workspace

This workspace is managed by AgentConnect.

AgentConnect is the source of truth for:

- task state
- manager claims
- decisions
- attempts
- subtasks
- reviews
- artifacts
- approvals
- completion
- handoff summaries

## Before working

1. Call `get_task_context_pack`.
2. Ensure the task or review is claimed.

## During work

- Record meaningful progress with `record_attempt`.
- Record durable decisions with `record_decision`.
- Delegate work with `submit_subtask`.
- Check delegated work with `get_subtask_status`.
- Request review with `request_review`.
- Register important outputs as artifacts.
- Read outputs through `list_artifacts` and `read_artifact_chunk`.

## Before declaring completion

- Ensure important artifacts are registered.
- Ensure required reviews are complete.
- Ensure open subtasks are resolved.
- Ensure handoff is updated.
- Ensure AgentConnect audit passes.

## Do not call these directly

- Temporal
- WikiBrain
- Cognee
- Graphiti
- local model manager
- cloud model providers
- rented GPU providers
- secrets manager

Use AgentConnect tools. Recalled memory is *external context*, not truth: items
labeled `trusted=false` are retrieval leads, not recorded facts.

**If it is not recorded in AgentConnect, it is not complete.**
"""

CLAUDE_MD = """# Claude Code — AgentConnect Managed Session

Read `AGENTCONNECT.md` first. It is the contract; this file is how you honor it.

Use AgentConnect MCP tools as the canonical interface.

Required first action:

- `get_task_context_pack`

Use:

- `claim_task`
- `record_attempt`
- `record_decision`
- `submit_subtask`
- `get_subtask_status`
- `request_review`
- `list_artifacts`
- `read_artifact_chunk`
- `release_task`

`AGENTCONNECT_TASK_ID` and `AGENTCONNECT_MANAGER_ID` are already in your
environment. Every tool infers them when you omit the argument, so you never need
to type an ID.

**Do not rely on chat history as canonical task state.** Your context window is
not durable and does not survive a handoff to another manager. The ledger does.
"""

CODEX_MD = """# Codex — AgentConnect Managed Session

Read `AGENTCONNECT.md` first. It is the contract; this file is how you honor it.

Use the AgentConnect CLI/API as the canonical interface.

Start with:

    agentconnect tasks context-pack $AGENTCONNECT_TASK_ID

Record meaningful work:

    agentconnect attempts add $AGENTCONNECT_TASK_ID --by $AGENTCONNECT_MANAGER_ID \\
        --summary "what you did" --outcome "what happened"

Complete reviews through AgentConnect:

    agentconnect reviews complete $AGENTCONNECT_REVIEW_ID \\
        --by $AGENTCONNECT_MANAGER_ID --summary "verdict"

Helper scripts are on your `PATH`: `ac-context`, `ac-attempt`, `ac-audit`.

**Do not produce final results only in chat or unregistered files.** A file you
changed that no artifact references does not exist as far as the next manager —
or `agentconnect audit` — is concerned.
"""

#: Which harness gets which file. Unknown managers still get AGENTCONNECT.md.
HARNESS_FILES: dict[str, tuple[str, str]] = {
    "claude": ("CLAUDE.md", CLAUDE_MD),
    "claude-code": ("CLAUDE.md", CLAUDE_MD),
    "codex": ("CODEX.md", CODEX_MD),
    "openclaw": ("CODEX.md", CODEX_MD),
}


def harness_file(manager_id: str) -> Optional[tuple[str, str]]:
    key = manager_id.strip().lower()
    if key in HARNESS_FILES:
        return HARNESS_FILES[key]
    for name, spec in HARNESS_FILES.items():
        if key.startswith(name):
            return spec
    return None


_HELPERS = {
    "ac-context": 'exec agentconnect tasks context-pack "${1:-$AGENTCONNECT_TASK_ID}" "${@:2}"\n',
    "ac-attempt": (
        'exec agentconnect attempts add "$AGENTCONNECT_TASK_ID" '
        '--by "$AGENTCONNECT_MANAGER_ID" --summary "$*"\n'
    ),
    "ac-audit": 'exec agentconnect audit "${1:-$AGENTCONNECT_TASK_ID}" "${@:2}"\n',
}


def mcp_config(api_url: str, env: dict[str, str]) -> dict[str, Any]:
    """A *session-local* MCP config naming only the AgentConnect server (§10).

    The agent sees AgentConnect's tools. It does not see Temporal, WikiBrain,
    Cognee, Graphiti, the local model manager, or the secrets manager, because no
    server offering them is configured here.
    """
    return {
        "mcpServers": {
            "agentconnect": {
                "command": "agentconnect-mcp",
                "args": [],
                "env": {k: v for k, v in env.items() if k.startswith("AGENTCONNECT_")},
            }
        },
        "allowedTools": [f"mcp__agentconnect__{t}" for t in EXPOSED_MCP_TOOLS],
        "deniedTools": list(DENIED_MCP_TOOLS),
        "_comment": (
            "Session-local. AgentConnect is the only MCP server a managed agent sees; "
            "durable work must enter the ledger through it."
        ),
    }


class WorkspaceBuilder:
    """Materializes a workspace on disk. Idempotent: relaunching reuses it."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root else default_workspace_root()

    def path_for(self, entity_id: str) -> Path:
        return self.root / entity_id

    # ------------------------------------------------------------------ repo
    def _materialize_repo(
        self, dest: Path, source: Optional[Path], mode: str, entity_id: str, title: str,
    ) -> RepoMode:
        if dest.exists() and any(dest.iterdir()):
            # Already materialized by an earlier launch.
            return RepoMode.git_worktree if is_git_repo(dest) else RepoMode.copy
        if source is None or not source.exists():
            dest.mkdir(parents=True, exist_ok=True)
            return RepoMode.empty

        if mode in ("auto", "git_worktree") and is_git_repo(source):
            dest.parent.mkdir(parents=True, exist_ok=True)
            if _add_worktree(source, dest, branch_name(entity_id, title)):
                return RepoMode.git_worktree
            if mode == "git_worktree":
                raise RuntimeError(f"could not create a git worktree from {source}")

        if mode == "copy":
            shutil.copytree(source, dest, dirs_exist_ok=True)
            return RepoMode.copy

        # `auto` on a non-git directory, or an explicit bind: point at the real
        # tree rather than silently duplicating a large repo.
        dest.parent.mkdir(parents=True, exist_ok=True)
        if mode in ("auto", "bind"):
            try:
                dest.symlink_to(source, target_is_directory=True)
                return RepoMode.bind
            except OSError as exc:
                _log.warning("symlink %s -> %s failed (%s); copying", dest, source, exc)
                shutil.copytree(source, dest, dirs_exist_ok=True)
                return RepoMode.copy
        dest.mkdir(parents=True, exist_ok=True)
        return RepoMode.empty

    # ----------------------------------------------------------------- build
    def build(
        self,
        entity_id: str,
        title: str,
        manager_id: str,
        repo_source: Optional[str] = None,
        repo_mode: str = "auto",
    ) -> tuple[Path, Path, Path, RepoMode]:
        """Create (or reuse) the directory tree. Returns (root, repo, artifacts, mode)."""
        base = self.path_for(entity_id)
        base.mkdir(parents=True, exist_ok=True)
        artifacts = base / "artifacts"
        logs = base / "logs"
        helpers = base / "bin"
        for directory in (artifacts, logs, helpers):
            directory.mkdir(parents=True, exist_ok=True)

        source = Path(repo_source).expanduser().resolve() if repo_source else None
        resolved = self._materialize_repo(base / "repo", source, repo_mode, entity_id, title)

        for name, body in _HELPERS.items():
            script = helpers / name
            script.write_text(f"#!/bin/sh\nset -eu\n{body}", encoding="utf-8")
            script.chmod(0o755)

        return base, base / "repo", artifacts, resolved

    def write_instructions(self, base: Path, manager_id: str) -> list[str]:
        written = ["AGENTCONNECT.md"]
        (base / "AGENTCONNECT.md").write_text(AGENTCONNECT_MD, encoding="utf-8")
        spec = harness_file(manager_id)
        if spec is not None:
            name, body = spec
            (base / name).write_text(body, encoding="utf-8")
            written.append(name)
        return written

    def write_env_file(self, base: Path, rendered: str) -> Path:
        path = base / ".env.agentconnect"
        path.write_text(rendered, encoding="utf-8")
        path.chmod(0o600)  # it holds a session token
        return path

    def write_mcp_config(self, base: Path, api_url: str, env: dict[str, str]) -> Path:
        path = base / ".mcp.json"
        path.write_text(
            json.dumps(mcp_config(api_url, env), indent=2) + "\n", encoding="utf-8"
        )
        path.chmod(0o600)
        return path

    def write_metadata(self, base: Path, metadata: dict[str, Any]) -> Path:
        path = base / "workspace.json"
        path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
        return path

    def read_metadata(self, base: Path) -> dict[str, Any]:
        path = Path(base) / "workspace.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            return {}


def mode_for(review_id: Optional[str], readonly: bool) -> SessionMode:
    if readonly:
        return SessionMode.readonly
    return SessionMode.reviewer if review_id else SessionMode.manager
