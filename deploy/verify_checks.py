#!/usr/bin/env python3
"""Shared in-process verification checks, used by both deploy/verify.sh and
deploy/verify.ps1 so the assertions can't drift between the two platforms.

Run inside the scratch venv that deploy/install.* just built, with
AGENTCONNECT_CONFIG_DIR pointing at the scratch checkout's config/ dir and
AGENTCONNECT_DB pointing at a scratch sqlite path. Expects the same
AGENTCONNECT_INSTALL_<ENV_VAR> values the caller fed to configure_secrets.py,
and checks that they round-trip through SecretResolver and that every
provider they belong to shows up live and healthy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PROMPTS_FILE = Path(__file__).resolve().parent / "secrets_prompts.json"


def main() -> int:
    prompts = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
    failures: list[str] = []

    from agentconnect.common.secrets import SecretResolver
    from agentconnect.router import mcp_server

    resolver = SecretResolver()
    for row in prompts:
        expect_var = f"AGENTCONNECT_INSTALL_{row['env_var']}"
        expected = os.environ.get(expect_var)
        if expected is None:
            failures.append(f"{expect_var} not set in environment -- can't verify {row['provider_id']}")
            continue
        try:
            got = resolver.resolve(row["secret_ref"])
        except Exception as exc:  # noqa: BLE001 - report and keep checking the rest
            failures.append(f"resolve({row['secret_ref']!r}) raised: {exc}")
            continue
        if got != expected:
            failures.append(f"resolve({row['secret_ref']!r}) = {got!r}, expected {expected!r}")
        else:
            print(f"PASS  resolver  {row['provider_id']:<16} resolves to the expected pasted value")

    svc = mcp_server._build_service()
    status = svc.get_router_status()
    provider_status = {p["provider"]: p for p in svc.get_provider_status()}
    for row in prompts:
        pid = row["provider_id"]
        if pid not in status["providers"]:
            failures.append(f"{pid} missing from get_router_status()['providers']")
            continue
        health = provider_status.get(pid, {}).get("health")
        if health != "healthy":
            failures.append(f"{pid} health = {health!r}, expected 'healthy'")
        else:
            print(f"PASS  provider   {pid:<16} present and healthy")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nAll in-process verification checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
