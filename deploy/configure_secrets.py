#!/usr/bin/env python3
"""Bootstrap config/secrets.local.yaml from deploy/secrets_prompts.json.

Shared by both deploy/install.sh and deploy/install.ps1 so the paste-a-key
logic can't drift between the two platforms — each installer just calls this
script with its own venv's python. Never touches config/secrets.yaml (that
file, if present, always wins per SecretResolver's load order and is left
untouched here).

Run standalone any time to add/refresh a key:
    <venv>/bin/python deploy/configure_secrets.py
    <venv>/bin/python deploy/configure_secrets.py --reconfigure-secrets
    <venv>/bin/python deploy/configure_secrets.py --non-interactive   # CI / verify.sh
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
PROMPTS_FILE = Path(__file__).resolve().parent / "secrets_prompts.json"
SECRETS_YAML = CONFIG_DIR / "secrets.yaml"
SECRETS_LOCAL = CONFIG_DIR / "secrets.local.yaml"
SECRETS_EXAMPLE = CONFIG_DIR / "secrets.example.yaml"


def load_prompts() -> list[dict]:
    return json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))


def atomic_write(path: Path, text: str) -> None:
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def tighten_permissions(path: Path) -> None:
    if os.name == "posix":
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 600
    else:
        # Best effort on Windows: NTFS ACLs aren't a strict equivalent of chmod
        # 600, so this is a courtesy, not a guarantee.
        try:
            import subprocess

            subprocess.run(
                ["icacls", str(path), "/inheritance:r", "/grant:r", f"{os.environ.get('USERNAME', '')}:F"],
                capture_output=True,
                check=False,
            )
        except Exception:
            pass


def mask(value: str) -> str:
    if len(value) <= 4:
        return "*" * len(value)
    return "*" * (len(value) - 4) + value[-4:]


def already_set(spec: dict | None) -> bool:
    if spec is None:
        return False
    kind = spec.get("kind")
    if kind == "literal":
        return bool(spec.get("value"))
    if kind == "env":
        return bool(os.environ.get(spec.get("var", "")))
    return kind == "op"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never prompt; only apply AGENTCONNECT_INSTALL_<ENV_VAR> overrides.",
    )
    parser.add_argument(
        "--reconfigure-secrets",
        action="store_true",
        help="Re-prompt for every row even if it already resolves to a value.",
    )
    args = parser.parse_args()

    if SECRETS_YAML.exists():
        print(
            f"[configure_secrets] {SECRETS_YAML} already exists and takes precedence over "
            f"secrets.local.yaml (SecretResolver checks secrets.yaml first, not merged). "
            f"Leaving it untouched -- edit it directly instead."
        )
        return 0

    seeded = False
    if not SECRETS_LOCAL.exists():
        shutil.copyfile(SECRETS_EXAMPLE, SECRETS_LOCAL)
        seeded = True
        print(f"[configure_secrets] seeded {SECRETS_LOCAL} from {SECRETS_EXAMPLE.name}")

    data = yaml.safe_load(SECRETS_LOCAL.read_text(encoding="utf-8")) or {}
    refs = data.setdefault("secret_refs", {})

    prompts = load_prompts()
    interactive = not args.non_interactive and sys.stdin.isatty()
    changed = False

    for row in prompts:
        ref = row["secret_ref"]
        label = row["label"]
        override_var = f"AGENTCONNECT_INSTALL_{row['env_var']}"
        override_val = os.environ.get(override_var)
        existing = refs.get(ref)

        if override_val:
            refs[ref] = {"kind": "literal", "value": override_val}
            changed = True
            print(f"[configure_secrets] {label}: set from ${override_var} ({mask(override_val)})")
            continue

        if already_set(existing) and not args.reconfigure_secrets:
            print(f"[configure_secrets] {label}: already configured, skipping (--reconfigure-secrets to change)")
            continue

        if not interactive:
            print(f"[configure_secrets] {label}: no value provided, leaving unset")
            continue

        pasted = getpass.getpass(f"Paste {label} API key (Enter to skip): ").strip()
        if not pasted:
            print(f"[configure_secrets] {label}: skipped")
            continue
        refs[ref] = {"kind": "literal", "value": pasted}
        changed = True
        print(f"[configure_secrets] {label}: saved ({mask(pasted)})")

    if changed:
        # yaml.safe_dump doesn't round-trip comments, so once we actually
        # rewrite the file it loses secrets.example.yaml's inline docs
        # (a one-time cost the first time a key changes; see deploy/README.md).
        atomic_write(SECRETS_LOCAL, yaml.safe_dump(data, default_flow_style=False, sort_keys=False))
        print(f"[configure_secrets] wrote {SECRETS_LOCAL}")
    elif seeded:
        print(f"[configure_secrets] {SECRETS_LOCAL} left as the unmodified example template")

    tighten_permissions(SECRETS_LOCAL)

    print("\n[configure_secrets] Provider secret status:")
    try:
        from agentconnect.common.secrets import SecretResolver

        resolver = SecretResolver()
        for row in prompts:
            ok = resolver.has(row["secret_ref"])
            status = "OK     " if ok else "MISSING"
            print(f"  {status}  {row['provider_id']:<16} {row['label']}")
    except ImportError:
        print("  agentconnect-core is not importable from this Python -- install it first, then re-run.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
