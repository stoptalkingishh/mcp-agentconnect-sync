# Operator guide

How to run one task through AgentConnect, end to end. Every command below was executed
during a manual dogfood run; the failure modes at the bottom are the ones that actually
happened, not ones imagined for the doc.

The rule the whole system exists to enforce:

> Agents may think and work inside their own harness. But durable work must enter
> AgentConnect. **If it is not recorded in AgentConnect, it did not happen.**

## Scope — read this before you trust it

**This is not a sandbox.** It is a compliance layer. It makes AgentConnect the normal
path and makes bypasses visible; it does not contain a hostile process. Concretely:

* **`AGENTCONNECT_DB_PATH` is forwarded into the agent's environment on purpose.**
  Without it the agent's own `agentconnect` CLI opens `~/.agentconnect/agentconnect.db`
  and writes to a second ledger nobody reads. Forwarding it is what makes the agent's
  work land where the audit looks. It is a path, not a credential — it grants no cloud
  spend, no model access, no backend write token. Backend credentials are **never**
  forwarded.
* **Managed-session CLI commands are restricted by `AGENTCONNECT_MODE`.** `launch` and
  `shell` set it, so the CLI knows it is the agent rather than the operator, and refuses
  `complete` and `memory promote`. A reviewer may still complete its own *review*.
* **Direct SQLite, filesystem, and environment tampering are out of scope.** An agent
  that unsets `AGENTCONNECT_MODE`, opens the ledger file with the `sqlite3` binary, or
  edits an artifact on disk is stopped by nothing in this layer. Guarding against that
  needs OS-level isolation — a container, a microVM, a separate user — which this
  deliberately does not provide.

What the layer *does* buy you: an agent cannot casually reach a credential that is not
in its environment, cannot mistype a task id, cannot complete its own work with the
command sitting right there, and cannot make the audit agree that undone work is done.

## Install

`agentconnect-core` alone gives you the library, not the command. Install the CLI too:

```sh
uv pip install --python .venv -e packages/agentconnect-core -e packages/agentconnect-cli
.venv/bin/agentconnect --help
```

The `agentconnect` binary must be **on `PATH` inside the agent's shell** — the generated
`bin/ac-*` helpers exec it by name. Activate the venv (or add its `bin/` to `PATH`)
rather than calling `.venv/bin/agentconnect` by absolute path; the environment the agent
receives inherits your `PATH`, and nothing else.

Optional adapters: `agentconnect-mcp` (an MCP-speaking harness needs the
`agentconnect-mcp` binary on `PATH` too), `agentconnect-api`, `agentconnect-linear`,
`agentconnect-temporal`.

## Configure the ledger

Everything hangs off one environment variable. Set it before any command:

```sh
export AGENTCONNECT_DB_PATH=/srv/agentconnect/agentconnect.db
export AGENTCONNECT_ARTIFACT_DIR=/srv/agentconnect/artifacts    # optional
export AGENTCONNECT_WORKSPACE_DIR=/srv/agentconnect/workspaces  # optional
```

Unset, the CLI falls back to `~/.agentconnect/agentconnect.db`. That fallback is the
single most expensive mistake available here: an agent writing to a second ledger looks
like an agent that recorded nothing, and the audit will say exactly that, blaming the
agent for your configuration. `agentconnect shell` forwards `AGENTCONNECT_DB_PATH` (and
the other `AGENTCONNECT_*` **config** variables — paths and knobs) into the agent's
environment precisely so this cannot happen. It forwards **no credentials**.

## The loop

### 1. Create a task

```sh
agentconnect tasks create \
  --title "Dogfood AgentConnect managed loop" \
  --goal  "Run one full managed loop through launch, shell, subtask, review, audit, completion." \
  --by matthew
```

Returns JSON; keep the `id` (`task_…`).

### 2. Launch a managed agent session

```sh
agentconnect launch codex --task "$TASK" --claim --repo /path/to/repo
```

Creates a workspace, a git worktree on branch `agentconnect/<task_id>/<slug>`,
`AGENTCONNECT.md` plus a harness file (`CODEX.md` / `CLAUDE.md`), a `0600`
`.env.agentconnect` holding a short-lived scoped token, and `.mcp.json` naming
AgentConnect as the only MCP server. With `--claim` it claims the task. It prints the
`shell` command to run next.

Inspect: `agentconnect sessions list --task "$TASK"`, `agentconnect workspaces list`.

For a review, `--review <review_id>`; the manager id must be the review's assignee, or
the claim is refused.

### 3. Shell into the workspace

```sh
agentconnect shell --task "$TASK" -- <agent-command>
```

**A session is consumed by one `shell`.** The command you pass is the agent's whole
life; when it exits, the session ends and its token is revoked. Poking around with
several `shell` invocations does not work — the second reports *no prepared session*.
Use `--print-env` (which does not consume the session) to look, or `launch` again.

The agent's environment is built from an allowlist: `PATH HOME SHELL TERM LANG LC_ALL
USER LOGNAME TMPDIR TZ`, the `AGENTCONNECT_*` session variables, and the forwarded
config pointers. Verify:

```sh
agentconnect shell --task "$TASK" -- env | grep -E 'OPENAI|ANTHROPIC|LINEAR|TEMPORAL|WIKIBRAIN|COGNEE|GRAPHITI|LOCAL_MODEL|AWS|SECRET'
# expect: no output
```

### 4. Retrieve context (the agent)

```sh
agentconnect tasks context-pack "$AGENTCONNECT_TASK_ID" --profile manager_brief
# or the shim on PATH:
ac-context
```

The pack carries the handoff plus **labeled external memory**. Items are labeled
`trusted` or not; untrusted items are retrieval leads, never recorded facts. Warnings
name any scope that could not be resolved. An MCP harness calls `get_task_context_pack`
instead and gets the same object.

### 5. Record an attempt (the agent)

```sh
agentconnect attempts add "$AGENTCONNECT_TASK_ID" \
  --actor "$AGENTCONNECT_MANAGER_ID" \
  --summary "Started the loop and retrieved context." --outcome in_progress
# or: ac-attempt "Started the loop and retrieved context."
```

The audit later asks whether an attempt was recorded **during this session**. An attempt
from a previous session does not excuse the current one.

Record a decision behind any durable change — the audit asks for that too:

```sh
agentconnect decisions add "$AGENTCONNECT_TASK_ID" --by "$AGENTCONNECT_MANAGER_ID" \
  --decision "Prove the loop with one bounded subtask artifact." --rationale "dogfood"
```

### 6. Submit a subtask (the agent)

```sh
agentconnect subtasks submit "$AGENTCONNECT_TASK_ID" \
  --title "Create dogfood artifact" \
  --instructions "Create a short artifact proving the worker received context." \
  --privacy repo_sensitive
```

### 7. Worker context is injected, not fetched

A worker harness usually has no MCP client, so it cannot *pull* context. Before any
worker runs, `service.prepare_worker_context` builds a `worker_brief` and attaches it to
the subtask. Both execution backends call it — `DirectExecutionBackend` directly, the
Temporal `recall_context` activity by delegation — so a worker's memory never depends on
which backend is installed.

```sh
agentconnect subtasks show "$SUBTASK" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["subtask"]["metadata"]["context_pack"])'
```

Note the shape: `subtasks show` returns `{"subtask": …, "runs": […]}`. The pack is under
`subtask.metadata.context_pack`, with `profile: worker_brief` and `scopes_queried`.

### 8. Artifacts

```sh
agentconnect artifacts list "$TASK"
agentconnect artifacts read "$ARTIFACT" --offset 0 --limit 8000   # --all to page to EOF
```

A worker's output is registered as an artifact. A file on disk that is not an artifact
does not exist as far as the audit is concerned.

### 9. Request a review

```sh
agentconnect reviews request "$TASK" --to codex-reviewer --by codex \
  --artifact "$ARTIFACT" \
  --criteria "Verify the artifact exists and worker context was attached before execution."
```

It lands in `agentconnect inbox codex-reviewer`.

### 10–11. Review

```sh
agentconnect launch codex-reviewer --review "$REVIEW" --claim
agentconnect shell --review "$REVIEW" -- <reviewer-command>
#   inside: agentconnect tasks context-pack "$AGENTCONNECT_TASK_ID" --profile reviewer_brief

agentconnect reviews complete "$REVIEW" --file review_result.md --summary approved
```

The review result is stored as an artifact, so the manager reads the verdict through
AgentConnect rather than through chat.

### 12. Audit

```sh
agentconnect audit "$TASK"
```

Asks one question several ways: *you did something — where is it in the ledger?* It
checks the workspace, the session, the claim, an attempt during this session, changed
files registered as artifacts, resolved subtasks, completed reviews, a decision behind
any durable change, Linear agreement, and status consistency. Handoff freshness and
memory capture are **advisory**: they warn, they never block.

The audit **writes nothing**. Run it twice; the database is byte-identical. That is
load-bearing — an audit that repairs what it measures is not an audit.

### 13. Complete (operator only)

```sh
agentconnect complete "$TASK" --by matthew
```

Regenerates the handoff, audits, marks the ledger `succeeded`, and *only then* fires the
completion hooks that tell Linear. A hook that raises is logged, never fatal: a tracker
outage cannot un-complete finished work.

Completion is an **operator** action. A session token cannot reach `complete_task` — it
is in no mode's action list, so MCP and HTTP deny it structurally. The CLI additionally
refuses `complete` and `memory promote` whenever `AGENTCONNECT_MODE` is set, which is
exactly when it is running inside a managed agent session:

```sh
agentconnect shell --task "$TASK" -- agentconnect complete "$TASK"
# forbidden_action: completion is an operator action; this is a managed agent session…
```

This is a compliance guard, not a security control. An agent that edits its environment
or opens the SQLite file directly is stopped by nothing here.

## How Linear fits

AgentConnect is canonical. `complete_task` updates the ledger first and Linear second,
through a completion hook. A Linear issue moved to Done by a human records an *event*
and changes nothing; the issue body says so (`**Canonical status:** AgentConnect-managed`).
Comment commands (`/agentconnect complete`) route through the same audit.

With no Linear configured, `agentconnect linear issue "$TASK"` reports the task is not
synced and completion simply fires no hook. Nothing else changes.

## Common failure modes

**`No module named 'agentconnect.core'` when running the CLI.** A stale editable install
predating the package. Reinstall: `uv pip install --python .venv --force-reinstall
--no-deps -e packages/agentconnect-core`.

**The agent's `ac-*` helpers report `agentconnect: not found`.** The binary is not on the
`PATH` the agent inherited. Activate the venv before `agentconnect shell`.

**`no prepared session for task_…; run 'agentconnect launch' first`.** The session was
already consumed by an earlier `shell`. Launch again.

**Audit: `No record_attempt was made during this session.`** Either the agent recorded
nothing, or you relaunched after it worked, which starts a fresh session. Check
`agentconnect sessions list --task "$TASK"` against the attempt timestamps.

**Audit: `Durable changes were made but no decision was recorded.`** Working as intended.
Record a decision.

**Attempts vanish; the audit says the agent recorded nothing.** The classic. The agent
wrote to the fallback ledger. Confirm `AGENTCONNECT_DB_PATH` is exported *before*
`launch`, and that it appears in `agentconnect shell --task "$TASK" --print-env`.

**`policy_violation: review … is assigned to 'x', not 'y'.`** Launch the reviewer session
as the assignee.

**Context pack warns `no project, repo scope is known for this task`.** Repo- and
project-scoped memory cannot surface. Set `task.metadata` (`repo_id`, `project_id`) or
`memory.default_scopes` in `config/memory.yaml`.
