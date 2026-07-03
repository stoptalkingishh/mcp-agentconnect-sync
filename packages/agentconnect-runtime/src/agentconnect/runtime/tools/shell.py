"""Shell tool: run a command inside the task workspace.

Availability is gated by ``RuntimeConfig.allow_shell`` (enforced in the graph,
not here). The command runs in its own session so a timeout kills the whole
process group — backgrounded children included, not just the shell. Output is
combined stdout+stderr plus the exit code, formatted as an observation string.
"""

from __future__ import annotations

import os
import signal
import subprocess

from ..workspace import Workspace


def run_shell(ws: Workspace, command: str, timeout: float = 60.0) -> str:
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=ws.root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return f"ERROR: command timed out after {timeout:.0f}s: {command}"
    parts = [f"exit_code={proc.returncode}"]
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    return "\n".join(parts)
