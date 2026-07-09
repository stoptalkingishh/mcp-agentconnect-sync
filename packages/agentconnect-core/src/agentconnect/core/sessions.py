"""Session scopes, tokens, and environment sanitization (compliance spec §8–§9).

> Agents may think and work inside their own harness. But durable work must enter
> AgentConnect. **If it is not recorded in AgentConnect, it did not happen.**

This is a *compliance* boundary, not a security boundary. It makes AgentConnect
the normal path and makes bypasses visible; it does not stop a hostile agent. A
determined process can still read `~/.aws/credentials` off the disk. What it does
stop is the overwhelmingly common case: an agent that reaches for
``OPENAI_API_KEY`` because it is sitting right there in the environment.

Two rules do most of the work:

1. **Allowlist, never denylist, for what survives.** An unknown variable is
   dropped. The denylist exists only to police the *explicit* opt-in path, where
   a human might otherwise re-admit a credential by name.
2. **Only a token's hash is stored.** The plaintext exists for exactly as long as
   it takes to write `.env.agentconnect` with mode 0600.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
from typing import Any, Iterable, Optional

from .models import SessionMode

DEFAULT_TOKEN_TTL_SECONDS = 12 * 3600

# --------------------------------------------------------------------- scopes

#: A manager drives the task: it decides, delegates, and asks for review.
MANAGER_ACTIONS: frozenset[str] = frozenset({
    "get_task_context_pack", "claim_task", "record_attempt", "record_decision",
    "submit_subtask", "get_subtask_status", "list_artifacts", "read_artifact_chunk",
    "request_review", "release_task",
})

#: A reviewer reads and judges. It cannot decide, delegate, or complete the task.
REVIEWER_ACTIONS: frozenset[str] = frozenset({
    "get_task_context_pack", "claim_review", "record_attempt", "list_artifacts",
    "read_artifact_chunk", "complete_review",
})

#: Look, do not touch. What `--readonly` and `--force-readonly` grant.
READONLY_ACTIONS: frozenset[str] = frozenset({
    "get_task_context_pack", "list_artifacts", "read_artifact_chunk",
    "get_subtask_status",
})

ACTIONS_BY_MODE: dict[SessionMode, frozenset[str]] = {
    SessionMode.manager: MANAGER_ACTIONS,
    SessionMode.reviewer: REVIEWER_ACTIONS,
    SessionMode.readonly: READONLY_ACTIONS,
}

#: Named so a reviewer of *this file* can see what is being withheld, and so a
#: token that somehow lists one of these is rejected rather than silently obeyed.
#: These are not tools AgentConnect exposes; they are the shapes of the tools a
#: backend would expose if an agent reached it directly.
FORBIDDEN_ACTIONS: frozenset[str] = frozenset({
    "promote_memory_candidate", "wikibrain_promote", "wikibrain_admin",
    "cognee_write", "graphiti_write",
    "temporal_signal", "temporal_admin", "workflow_terminate",
    "local_model_generate", "secrets_read", "admin_settings",
    "grant_approval", "approve_subtask", "deny_subtask",
})


def actions_for(mode: SessionMode) -> frozenset[str]:
    return ACTIONS_BY_MODE[mode]


# ---------------------------------------------------------------- environment

#: Credentials that must never reach a proprietary agent (compliance §8).
SECRET_DENYLIST: frozenset[str] = frozenset({
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "LINEAR_API_KEY",
    "TEMPORAL_ADDRESS", "TEMPORAL_NAMESPACE", "TEMPORAL_CLIENT_CERT",
    "TEMPORAL_CLIENT_KEY",
    "WIKIBRAIN_ADMIN_TOKEN", "WIKIBRAIN_WRITE_TOKEN",
    "COGNEE_WRITE_TOKEN", "GRAPHITI_WRITE_TOKEN",
    "LOCAL_MODEL_MANAGER_TOKEN",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AZURE_CLIENT_SECRET",
    "GCP_SERVICE_ACCOUNT", "SECRETS_MANAGER_TOKEN",
})

#: Anything matching these is refused from the explicit opt-in list too. The
#: denylist above is what the spec names; this is what it *meant*.
_SECRETISH = re.compile(
    r"(^|_)(API_KEY|APIKEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIALS|PRIVATE_KEY)($|_)"
)

#: The minimum a shell and a modern CLI need to function.
BASE_ALLOWLIST: frozenset[str] = frozenset({
    "PATH", "HOME", "SHELL", "TERM", "LANG", "LC_ALL", "USER", "LOGNAME", "TMPDIR", "TZ",
})

#: The session's own identity. `AGENTCONNECT_SESSION_TOKEN` is the one credential
#: an agent gets, and it buys exactly `actions_for(mode)`.
SESSION_VARS: tuple[str, ...] = (
    "AGENTCONNECT_API_URL", "AGENTCONNECT_TASK_ID", "AGENTCONNECT_REVIEW_ID",
    "AGENTCONNECT_MANAGER_ID", "AGENTCONNECT_WORKSPACE_ID", "AGENTCONNECT_SESSION_ID",
    "AGENTCONNECT_SESSION_TOKEN", "AGENTCONNECT_MODE",
)

#: Opt-in extras, comma separated: `AGENTCONNECT_SHELL_ALLOW_ENV=NVM_DIR,PYENV_ROOT`.
ALLOW_ENV_VAR = "AGENTCONNECT_SHELL_ALLOW_ENV"


def is_secretish(name: str) -> bool:
    """True for a name that *looks* like a credential, denylisted or not."""
    upper = name.upper()
    return upper in SECRET_DENYLIST or bool(_SECRETISH.search(upper))


def extra_allowed(environ: dict[str, str]) -> list[str]:
    raw = environ.get(ALLOW_ENV_VAR, "")
    return [n.strip() for n in raw.split(",") if n.strip()]


def sanitize_env(
    environ: dict[str, str],
    session_env: dict[str, str],
    extra_allow: Optional[Iterable[str]] = None,
    helper_bin: Optional[str] = None,
) -> dict[str, str]:
    """Build the agent's environment from an allowlist, then add the session.

    A bare ``env -i`` breaks too many tools to be useful, so the allowlist keeps
    the handful of variables a shell needs. Everything else — every API key, every
    backend address, every cloud credential — is simply not carried over, because
    it was never on the list.

    `extra_allow` is the escape hatch for local tooling (`NVM_DIR`, `PYENV_ROOT`).
    It cannot re-admit anything that looks like a credential: an operator who
    tries is told, not quietly obeyed.
    """
    allowed = set(BASE_ALLOWLIST)
    for name in extra_allow or extra_allowed(environ):
        if is_secretish(name):
            raise ValueError(
                f"{name!r} looks like a credential and cannot be allowed into an "
                f"agent shell (see {ALLOW_ENV_VAR})"
            )
        allowed.add(name)

    clean = {k: v for k, v in environ.items() if k in allowed and not is_secretish(k)}
    # The session's own vars are the point of the exercise; they are added last so
    # nothing in the ambient environment can shadow them.
    clean.update({k: v for k, v in session_env.items() if v})

    if helper_bin:
        clean["PATH"] = f"{helper_bin}{os.pathsep}{clean.get('PATH', '')}".rstrip(os.pathsep)
    return clean


def session_env_vars(
    api_url: str, task_id: Optional[str], review_id: Optional[str], manager_id: str,
    workspace_id: Optional[str], session_id: str, token: Optional[str], mode: SessionMode,
) -> dict[str, str]:
    """The §7 block. Tools infer task/review IDs from these when a caller omits
    them — which is the whole point: an agent that cannot mistype an ID cannot
    record its work against the wrong task."""
    return {
        "AGENTCONNECT_API_URL": api_url,
        "AGENTCONNECT_TASK_ID": task_id or "",
        "AGENTCONNECT_REVIEW_ID": review_id or "",
        "AGENTCONNECT_MANAGER_ID": manager_id,
        "AGENTCONNECT_WORKSPACE_ID": workspace_id or "",
        "AGENTCONNECT_SESSION_ID": session_id,
        "AGENTCONNECT_SESSION_TOKEN": token or "",
        "AGENTCONNECT_MODE": mode.value,
    }


def render_env_file(env: dict[str, str]) -> str:
    lines = [
        "# Written by `agentconnect launch`. Mode 0600: it holds a session token.",
        "# AgentConnect is the source of truth. See AGENTCONNECT.md.",
    ]
    for name in SESSION_VARS:
        lines.append(f"{name}={env.get(name, '')}")
    return "\n".join(lines) + "\n"


def parse_env_file(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


# -------------------------------------------------------------------- tokens

def mint_token() -> str:
    return f"act_{secrets.token_urlsafe(32)}"


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def build_scope(
    session_id: str, manager_id: str, mode: SessionMode,
    task_id: Optional[str], review_id: Optional[str],
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "manager_id": manager_id,
        "mode": mode.value,
        "task_id": task_id,
        "review_id": review_id,
        "actions": sorted(actions_for(mode)),
    }
