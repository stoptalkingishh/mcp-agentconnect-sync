#!/usr/bin/env pwsh
<#
.SYNOPSIS
  Installer for mcp-agentconnect on Windows. Mirrors deploy/install.sh
  step-for-step -- any change here should be mirrored there. See
  deploy/README.md for usage.

.PARAMETER WithModelManager
  Also install agentconnect-model-manager.
.PARAMETER WithRuntime
  Also install agentconnect-runtime (agentic execution).
.PARAMETER WithWeb
  Install the router's [web] extra (spend-approval host).
.PARAMETER Yes
  Non-interactive: skip all secret prompts.
.PARAMETER RecreateVenv
  Delete and recreate .venv.
.PARAMETER ReconfigureSecrets
  Re-prompt for every secret even if already set.
#>
param(
  [switch]$WithModelManager,
  [switch]$WithRuntime,
  [switch]$WithWeb,
  [switch]$Yes,
  [switch]$RecreateVenv,
  [switch]$ReconfigureSecrets
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$VenvDir = Join-Path $RepoRoot ".venv"

Write-Host "==> Preflight"
$PyBin = $null
foreach ($cand in @("py -3", "python3", "python")) {
  $parts = $cand.Split(" ")
  $exe = $parts[0]
  $exeArgs = $parts[1..($parts.Length - 1)]
  $cmd = Get-Command $exe -ErrorAction SilentlyContinue
  if ($cmd) {
    $check = & $exe @exeArgs -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
    if ($LASTEXITCODE -eq 0) {
      $PyBin = $cand
      break
    }
  }
}
if (-not $PyBin) {
  Write-Error "Need Python >= 3.10 on PATH (checked py -3, python3, python)."
  exit 1
}
$PyBinParts = $PyBin.Split(" ")
$verOutput = & $PyBinParts[0] @($PyBinParts[1..($PyBinParts.Length - 1)]) --version 2>&1
Write-Host "Using $verOutput ($PyBin)"

Write-Host "==> Virtual environment"
if ($RecreateVenv -and (Test-Path $VenvDir)) {
  Write-Host "Removing existing $VenvDir (-RecreateVenv)"
  Remove-Item -Recurse -Force $VenvDir
}
if (-not (Test-Path $VenvDir)) {
  & $PyBinParts[0] @($PyBinParts[1..($PyBinParts.Length - 1)]) -m venv $VenvDir
  Write-Host "Created $VenvDir"
} else {
  Write-Host "Reusing existing $VenvDir"
}
$VenvPy = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "==> Installing packages"
& $VenvPy -m pip install --upgrade pip | Out-Null
$InstallArgs = @("-e", (Join-Path $RepoRoot "packages\agentconnect-core"))
if ($WithWeb) {
  $InstallArgs += @("-e", (Join-Path $RepoRoot "packages\agentconnect-router") + "[web]")
} else {
  $InstallArgs += @("-e", (Join-Path $RepoRoot "packages\agentconnect-router"))
}
if ($WithModelManager) {
  $InstallArgs += @("-e", (Join-Path $RepoRoot "packages\agentconnect-model-manager"))
}
if ($WithRuntime) {
  $InstallArgs += @("-e", (Join-Path $RepoRoot "packages\agentconnect-runtime"))
}
& $VenvPy -m pip install @InstallArgs

Write-Host "==> Configuring secrets (config\secrets.local.yaml)"
$ConfigureArgs = @()
if ($Yes) { $ConfigureArgs += "--non-interactive" }
if ($ReconfigureSecrets) { $ConfigureArgs += "--reconfigure-secrets" }
# Explicit override, not cwd-based discovery: Push-Location doesn't reliably
# propagate to native child processes' working directory, so config.py's
# upward search from cwd could otherwise resolve the wrong repo's config/.
$env:AGENTCONNECT_CONFIG_DIR = Join-Path $RepoRoot "config"
& $VenvPy (Join-Path $ScriptDir "configure_secrets.py") @ConfigureArgs
Remove-Item Env:\AGENTCONNECT_CONFIG_DIR -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "==> Done."
Write-Host "Run the router directly:     $(Join-Path $VenvDir 'Scripts\agentconnect-router.exe')"
Write-Host "List MCP client targets:     $VenvPy $(Join-Path $ScriptDir 'register_mcp.py') list"
Write-Host "Register with a client:      $VenvPy $(Join-Path $ScriptDir 'register_mcp.py') add <client>"
