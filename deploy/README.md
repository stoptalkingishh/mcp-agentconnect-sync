# Deploy tooling

Get `mcp-agentconnect` running from a fresh clone, on Windows or Linux/macOS,
registered with whichever MCP-compatible agent client you use.

## Quick start

Linux/macOS:
```bash
./deploy/install.sh
```

Windows (PowerShell):
```powershell
.\deploy\install.ps1
```

Both scripts do the same thing: create `.venv`, editable-install the router
(and any optional packages you ask for), and walk you through pasting in
provider API keys. Re-running either script is safe — it's a no-op except for
whatever you actually changed.

## Flags

| bash               | PowerShell            | What it does                                             |
|---------------------|------------------------|-----------------------------------------------------------|
| `--with-model-manager` | `-WithModelManager`   | Also install `agentconnect-model-manager` (local GPU node) |
| `--with-runtime`    | `-WithRuntime`         | Also install `agentconnect-runtime` (agentic execution)   |
| `--with-web`        | `-WithWeb`             | Install the router's `[web]` extra (spend-approval host)  |
| `--yes` / `-y`      | `-Yes`                 | Non-interactive: skip all secret prompts                  |
| `--recreate-venv`   | `-RecreateVenv`        | Delete and recreate `.venv`                                |
| `--reconfigure-secrets` | `-ReconfigureSecrets` | Re-prompt for every secret, even ones already set        |

## Where your keys go

Pasted keys never touch `config/providers.yaml` and are never printed back —
only a masked `****abcd` confirmation. They're written to
`config/secrets.local.yaml`, which is gitignored and maps each provider's
`secret_ref` to `{kind: literal, value: "..."}` — the same resolver kind the
project's own `config/secrets.example.yaml` documents as "dev/testing only".
See `packages/agentconnect-core/src/agentconnect/common/secrets.py` for how
resolution actually works, and its `op://` support if you want a real secrets
manager (1Password) instead of local literals.

If `config/secrets.yaml` already exists, it takes precedence over
`secrets.local.yaml` (first-found-wins, not merged) — the installer detects
this, warns you, and leaves it alone.

Note: the very first time you actually add or change a key, the installer
rewrites `secrets.local.yaml` from a plain YAML dump, which drops the inline
comments from the `secrets.example.yaml` template it was seeded from. That's a
one-time cost; the file stays byte-for-byte identical to the template until
then.

Prefer to script it (CI, containers, no TTY)? Set
`AGENTCONNECT_INSTALL_<ENV_VAR>` for any row before running with `--yes`/`-Yes`
— e.g. `AGENTCONNECT_INSTALL_GEMINI_API_KEY=...`. See
`deploy/secrets_prompts.json` for the full list of rows (provider, secret_ref,
env var name) both installers read from — it's the single source of truth,
so add a row there if you wire up another provider.

## Registering with your MCP client

```bash
<venv>/bin/python deploy/register_mcp.py list
<venv>/bin/python deploy/register_mcp.py add <client>
```
```powershell
<venv>\Scripts\python.exe deploy\register_mcp.py list
<venv>\Scripts\python.exe deploy\register_mcp.py add <client>
```

Known clients: `claude-code`, `claude-desktop`, `cursor`, `windsurf`, `cline`,
or `all`. `register_mcp.py` writes an absolute path into your venv's
`agentconnect-router` console script plus an explicit
`AGENTCONNECT_CONFIG_DIR`, because GUI clients spawn the server without your
shell's `PATH` or working directory. It only ever touches the single
`mcpServers.agentconnect` key — any other server you already have registered
is left alone, and it takes a timestamped backup (`<file>.bak.<epoch>`) before
writing. `cline`'s config path is a best-effort guess (varies by VS Code
build); if it can't find a confident location, or for any client you pass
`--dry-run`, it just prints the JSON snippet for you to paste in yourself.

## Verification

```bash
./deploy/verify.sh
```
```powershell
.\deploy\verify.ps1
```

Copies the working tree into a scratch directory, runs the installer twice
(checking idempotency), resolves the pasted dummy keys back through
`SecretResolver`, confirms `gemini_free` / `groq_free` / `openrouter_free` /
`grok_cloud` all show up healthy via the router's own status calls, exercises
`register_mcp.py`'s no-clobber guarantee, and runs the full `pytest` suite.
Nothing it does touches your real `.venv` or `config/secrets*.yaml` — it
cleans up its scratch copy on exit.

## Troubleshooting

- **"already exists and takes precedence"** — you have a real
  `config/secrets.yaml`; edit that file directly, the installer won't touch it.
- **PowerShell execution policy** — not an issue here; the scripts never call
  `.venv\Scripts\Activate.ps1`, they invoke `.venv\Scripts\python.exe` by
  absolute path for every step.
- **`cline` not found** — its config path varies by VS Code build/version;
  run `register_mcp.py add cline --dry-run` and paste the snippet manually
  into your Cline MCP settings.
- **Grok / xAI provider not routing** — `grok_cloud` is real metered billing
  (`privacy: external_paid`), gated behind `set_budget(...)` on the running
  router. It stays blocked until you raise the budget above whatever you've
  set (e.g. the `$0.01` placeholder some setups use to stay free-tier-only).
