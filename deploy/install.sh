#!/usr/bin/env bash
# Installer for mcp-agentconnect on Linux/macOS. Mirrors deploy/install.ps1
# step-for-step -- any change here should be mirrored there. See
# deploy/README.md for usage.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

WITH_MODEL_MANAGER=0
WITH_RUNTIME=0
WITH_WEB=0
NON_INTERACTIVE=0
RECREATE_VENV=0
RECONFIGURE_SECRETS=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

  --with-model-manager   Also install agentconnect-model-manager
  --with-runtime         Also install agentconnect-runtime (agentic execution)
  --with-web             Install the router's [web] extra (spend-approval host)
  --yes, -y              Non-interactive: skip all secret prompts
  --recreate-venv        Delete and recreate .venv
  --reconfigure-secrets  Re-prompt for every secret even if already set
  -h, --help             Show this help
EOF
}

for arg in "$@"; do
  case "$arg" in
    --with-model-manager) WITH_MODEL_MANAGER=1 ;;
    --with-runtime) WITH_RUNTIME=1 ;;
    --with-web) WITH_WEB=1 ;;
    --yes|-y) NON_INTERACTIVE=1 ;;
    --recreate-venv) RECREATE_VENV=1 ;;
    --reconfigure-secrets) RECONFIGURE_SECRETS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage; exit 1 ;;
  esac
done

echo "==> Preflight"
PY_BIN=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1 \
     && "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY_BIN="$cand"
    break
  fi
done
if [ -z "$PY_BIN" ]; then
  echo "ERROR: need Python >= 3.10 on PATH (checked python3, python)." >&2
  exit 1
fi
echo "Using $("$PY_BIN" --version 2>&1) at $(command -v "$PY_BIN")"

echo "==> Virtual environment"
if [ "$RECREATE_VENV" = "1" ] && [ -d "$VENV_DIR" ]; then
  echo "Removing existing $VENV_DIR (--recreate-venv)"
  rm -rf "$VENV_DIR"
fi
if [ ! -d "$VENV_DIR" ]; then
  "$PY_BIN" -m venv "$VENV_DIR"
  echo "Created $VENV_DIR"
else
  echo "Reusing existing $VENV_DIR"
fi
VENV_PY="$VENV_DIR/bin/python"

echo "==> Installing packages"
"$VENV_PY" -m pip install --upgrade pip >/dev/null
INSTALL_ARGS=(-e "$REPO_ROOT/packages/agentconnect-core")
if [ "$WITH_WEB" = "1" ]; then
  INSTALL_ARGS+=(-e "$REPO_ROOT/packages/agentconnect-router[web]")
else
  INSTALL_ARGS+=(-e "$REPO_ROOT/packages/agentconnect-router")
fi
[ "$WITH_MODEL_MANAGER" = "1" ] && INSTALL_ARGS+=(-e "$REPO_ROOT/packages/agentconnect-model-manager")
[ "$WITH_RUNTIME" = "1" ] && INSTALL_ARGS+=(-e "$REPO_ROOT/packages/agentconnect-runtime")
"$VENV_PY" -m pip install "${INSTALL_ARGS[@]}"

echo "==> Configuring secrets (config/secrets.local.yaml)"
CONFIGURE_ARGS=()
[ "$NON_INTERACTIVE" = "1" ] && CONFIGURE_ARGS+=(--non-interactive)
[ "$RECONFIGURE_SECRETS" = "1" ] && CONFIGURE_ARGS+=(--reconfigure-secrets)
# Explicit override, not cwd-based discovery: this is invoked from wherever the
# caller happens to be, and config.py's upward search from cwd could otherwise
# find the wrong repo's config/ (e.g. a parent checkout).
AGENTCONNECT_CONFIG_DIR="$REPO_ROOT/config" "$VENV_PY" "$SCRIPT_DIR/configure_secrets.py" "${CONFIGURE_ARGS[@]}"

echo
echo "==> Done."
echo "Run the router directly:     $VENV_DIR/bin/agentconnect-router"
echo "List MCP client targets:     $VENV_PY $SCRIPT_DIR/register_mcp.py list"
echo "Register with a client:      $VENV_PY $SCRIPT_DIR/register_mcp.py add <client>"
