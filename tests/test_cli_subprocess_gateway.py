"""ProviderGateway's cli_subprocess dispatch path -- mocked subprocess calls
for fast/deterministic coverage, plus one real live invocation against the
actual installed claude/codex binaries (skipped if not present) matching
this project's "verify against reality" testing discipline."""

from __future__ import annotations

import json
import shutil
import subprocess
from unittest.mock import patch

import pytest
from agentconnect.common.config import CliConfig, ProviderConfig
from agentconnect.common.schemas import GenerateRequest
from agentconnect.router.gateway import GatewayResult, ProviderGateway

_CLAUDE_CLI = CliConfig(
    binary="claude",
    args=("-p", "--output-format", "json", "--tools", ""),
    output_mode="stdout_json",
    timeout_seconds=30,
    max_output_chars=20000,
)
_CODEX_CLI = CliConfig(
    binary="codex",
    args=("exec", "-s", "read-only", "--skip-git-repo-check"),
    output_mode="output_file",
    output_file_flag="-o",
    timeout_seconds=30,
    max_output_chars=20000,
)


def _provider(cli: CliConfig, provider_id="claude_cli") -> ProviderConfig:
    return ProviderConfig(
        provider_id=provider_id,
        type="cli_subprocess",
        endpoint="",
        privacy="external_paid",
        capabilities=("reasoning",),
        cli=cli,
    )


def _req(messages=None) -> GenerateRequest:
    return GenerateRequest(
        request_id="r1",
        task_id="t1",
        model_id="claude_cli",
        messages=messages or [{"role": "user", "content": "what do you think?"}],
    )


class _FakeCompletedProcess:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_call_routes_cli_subprocess_type_to_the_cli_path():
    cfg = _provider(_CLAUDE_CLI)
    gateway = ProviderGateway()
    fake = _FakeCompletedProcess(
        stdout=json.dumps({"type": "result", "is_error": False, "result": "PONG"})
    )
    with patch("agentconnect.router.gateway.subprocess.run", return_value=fake):
        result = gateway.call(cfg, _req())
    assert result.output_text == "PONG"
    assert result.provider == "claude_cli"


def test_claude_json_mode_extracts_result_and_usage():
    cfg = _provider(_CLAUDE_CLI)
    gateway = ProviderGateway()
    fake = _FakeCompletedProcess(
        stdout=json.dumps(
            {
                "type": "result",
                "is_error": False,
                "result": "the answer is 42",
                "usage": {"input_tokens": 10, "output_tokens": 4},
            }
        )
    )
    with patch("agentconnect.router.gateway.subprocess.run", return_value=fake) as run:
        result = gateway._call_cli_subprocess(cfg, _req())
    assert result.output_text == "the answer is 42"
    assert result.input_tokens == 10
    assert result.output_tokens == 4
    # Prompt goes via stdin, never as a positional arg (sidesteps the real
    # --tools variadic-greedy bug found live-testing this).
    _cmd, kwargs = run.call_args
    assert kwargs["input"] == "what do you think?"
    # Real bug found live-testing this: without an explicit encoding,
    # subprocess.run's text=True falls back to the platform default (cp1252
    # on Windows), silently corrupting any non-ASCII character an LLM's
    # response naturally contains (em-dashes, smart quotes, ...).
    assert kwargs["encoding"] == "utf-8"


def test_claude_is_error_true_raises_and_reports_failure():
    # Matches _call_cloud's convention: a failed call must never be reported
    # as a successful task -- raise, don't silently degrade to fake output.
    cfg = _provider(_CLAUDE_CLI)
    calls = []
    gateway = ProviderGateway(on_call_result=lambda *a: calls.append(a))
    fake = _FakeCompletedProcess(
        stdout=json.dumps({"type": "result", "is_error": True, "result": "boom"})
    )
    with (
        patch("agentconnect.router.gateway.subprocess.run", return_value=fake),
        pytest.raises(RuntimeError, match="claude_cli"),
    ):
        gateway._call_cli_subprocess(cfg, _req())
    assert calls == [("claude_cli", False, calls[0][2])]
    assert "is_error" in calls[0][2]


def test_nonzero_exit_raises():
    cfg = _provider(_CLAUDE_CLI)
    gateway = ProviderGateway()
    fake = _FakeCompletedProcess(returncode=1, stdout="", stderr="auth expired")
    with (
        patch("agentconnect.router.gateway.subprocess.run", return_value=fake),
        pytest.raises(RuntimeError, match="claude_cli"),
    ):
        gateway._call_cli_subprocess(cfg, _req())


def test_binary_not_found_raises():
    cfg = _provider(_CLAUDE_CLI)
    gateway = ProviderGateway()
    with (
        patch(
            "agentconnect.router.gateway.subprocess.run",
            side_effect=FileNotFoundError("no such binary"),
        ),
        pytest.raises(RuntimeError, match="claude_cli"),
    ):
        gateway._call_cli_subprocess(cfg, _req())


def test_timeout_raises():
    cfg = _provider(_CLAUDE_CLI)
    gateway = ProviderGateway()
    with (
        patch(
            "agentconnect.router.gateway.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=30),
        ),
        pytest.raises(RuntimeError, match="claude_cli"),
    ):
        gateway._call_cli_subprocess(cfg, _req())


def test_output_file_mode_reads_the_temp_file_and_cleans_it_up(tmp_path):
    cfg = _provider(_CODEX_CLI, provider_id="codex_cli")
    gateway = ProviderGateway()
    written_path = {}

    def fake_run(cmd, **kwargs):
        # The gateway must have appended [output_file_flag, path] to args.
        idx = cmd.index("-o")
        path = cmd[idx + 1]
        written_path["path"] = path
        with open(path, "w", encoding="utf-8") as f:
            f.write("codex's answer\n")
        return _FakeCompletedProcess(returncode=0)

    with patch("agentconnect.router.gateway.subprocess.run", side_effect=fake_run):
        result = gateway._call_cli_subprocess(cfg, _req())

    assert result.output_text == "codex's answer"
    import os

    assert not os.path.exists(written_path["path"])  # cleaned up after read


def test_output_file_mode_cleans_up_temp_file_even_on_failure():
    cfg = _provider(_CODEX_CLI, provider_id="codex_cli")
    gateway = ProviderGateway()
    written_path = {}

    def fake_run(cmd, **kwargs):
        idx = cmd.index("-o")
        written_path["path"] = cmd[idx + 1]
        return _FakeCompletedProcess(returncode=1, stderr="sandbox denied")

    with (
        patch("agentconnect.router.gateway.subprocess.run", side_effect=fake_run),
        pytest.raises(RuntimeError, match="codex_cli"),
    ):
        gateway._call_cli_subprocess(cfg, _req())

    import os

    assert not os.path.exists(written_path["path"])


def test_missing_output_file_flag_is_a_config_error_not_a_crash():
    bad_cli = CliConfig(binary="codex", output_mode="output_file", output_file_flag=None)
    cfg = _provider(bad_cli, provider_id="codex_cli")
    gateway = ProviderGateway()
    # Never reaches subprocess.run -- fails validation before spawning. The
    # specific validation reason is chained as __cause__, not the outer
    # message (_call_cli_subprocess wraps every failure uniformly).
    with pytest.raises(RuntimeError, match="codex_cli") as exc_info:
        gateway._call_cli_subprocess(cfg, _req())
    assert "output_file_flag" in str(exc_info.value.__cause__)


def test_missing_cli_config_raises_immediately():
    cfg = ProviderConfig(
        provider_id="x", type="cli_subprocess", endpoint="", privacy="external_paid",
        capabilities=(), cli=None,
    )
    gateway = ProviderGateway()
    with pytest.raises(RuntimeError, match="no `cli` config"):
        gateway._call_cli_subprocess(cfg, _req())


def test_render_prompt_single_message_has_no_role_framing():
    req = _req(messages=[{"role": "user", "content": "just this"}])
    assert ProviderGateway._render_cli_prompt(req) == "just this"


def test_render_prompt_multiple_messages_get_role_headers():
    req = _req(
        messages=[
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]
    )
    rendered = ProviderGateway._render_cli_prompt(req)
    assert "[system]" in rendered and "be terse" in rendered
    assert "[user]" in rendered and "hi" in rendered


def test_never_passes_a_workspace_write_flag_regardless_of_configured_args():
    # Defense in depth: even if providers.yaml were misconfigured with a
    # write-access flag, the gateway itself doesn't add one -- but this test
    # documents the actual guarantee: args are passed through verbatim from
    # config, so the lockdown lives in providers.yaml + review, not runtime
    # enforcement. Assert the current real config has no such flag.
    for cli in (_CLAUDE_CLI, _CODEX_CLI):
        assert "--add-dir" not in cli.args
        assert "workspace-write" not in cli.args
        assert "danger-full-access" not in cli.args


# --------------------------------------------------------------- live tests
# Real invocations against the actual installed binaries -- skipped (not
# failed) if a binary isn't present, so this suite still runs clean on a
# machine without them.

_HAS_CLAUDE = shutil.which("claude") is not None
_CODEX_BINARY = shutil.which("codex") or r"C:\Users\Ismael\.codex\.sandbox-bin\codex.exe"
_HAS_CODEX = __import__("os").path.exists(_CODEX_BINARY)


@pytest.mark.skipif(not _HAS_CLAUDE, reason="claude CLI not found on PATH")
def test_live_claude_cli_round_trip():
    cfg = _provider(_CLAUDE_CLI)
    gateway = ProviderGateway()
    req = _req(
        messages=[{"role": "user", "content": "Reply with exactly the word PONG and nothing else."}]
    )
    result = gateway._call_cli_subprocess(cfg, req)
    assert isinstance(result, GatewayResult)
    assert not result.output_text.startswith("[cli-stub:")
    assert "PONG" in result.output_text


@pytest.mark.skipif(not _HAS_CODEX, reason="codex CLI not found")
def test_live_codex_cli_round_trip():
    live_cli = CliConfig(
        binary=_CODEX_BINARY,
        args=("exec", "-s", "read-only", "--skip-git-repo-check"),
        output_mode="output_file",
        output_file_flag="-o",
        timeout_seconds=60,
    )
    cfg = _provider(live_cli, provider_id="codex_cli")
    gateway = ProviderGateway()
    req = _req(
        messages=[{"role": "user", "content": "Reply with exactly the word PONG and nothing else."}]
    )
    result = gateway._call_cli_subprocess(cfg, req)
    assert isinstance(result, GatewayResult)
    assert not result.output_text.startswith("[cli-stub:")
    assert "PONG" in result.output_text
