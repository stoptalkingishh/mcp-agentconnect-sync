from agentconnect.common.config import CliConfig, ProviderConfig, _parse_cli


def test_parse_cli_round_trips_all_fields():
    raw = {
        "binary": "claude",
        "args": ["-p", "--output-format", "json", "--tools", ""],
        "output_mode": "stdout_json",
        "cwd": "/tmp/council-scratch",
        "timeout_seconds": 90,
        "max_output_chars": 15000,
    }
    cli = _parse_cli(raw)
    assert cli == CliConfig(
        binary="claude",
        args=("-p", "--output-format", "json", "--tools", ""),
        output_mode="stdout_json",
        output_file_flag=None,
        cwd="/tmp/council-scratch",
        timeout_seconds=90.0,
        max_output_chars=15000,
    )


def test_parse_cli_defaults_when_optional_fields_absent():
    cli = _parse_cli({"binary": "codex"})
    assert cli.args == ()
    assert cli.output_mode == "stdout"
    assert cli.output_file_flag is None
    assert cli.cwd is None
    assert cli.timeout_seconds == 120.0
    assert cli.max_output_chars == 20000


def test_parse_cli_returns_none_for_missing_or_empty_block():
    assert _parse_cli(None) is None
    assert _parse_cli({}) is None


def test_parse_cli_expands_env_vars_in_cwd(monkeypatch):
    monkeypatch.setenv("COUNCIL_SCRATCH_DIR", "/scratch/council")
    cli = _parse_cli({"binary": "codex", "cwd": "$COUNCIL_SCRATCH_DIR/codex"})
    assert cli.cwd == "/scratch/council/codex"


def test_parse_cli_expands_env_vars_in_binary(monkeypatch):
    monkeypatch.setenv("CODEX_CLI_BINARY", "/opt/codex/codex.exe")
    cli = _parse_cli({"binary": "$CODEX_CLI_BINARY"})
    assert cli.binary == "/opt/codex/codex.exe"


def test_provider_config_cli_field_defaults_to_none():
    cfg = ProviderConfig(
        provider_id="x", type="cloud", endpoint="", privacy="external", capabilities=()
    )
    assert cfg.cli is None


def test_provider_config_carries_cli_config():
    cli = CliConfig(binary="claude")
    cfg = ProviderConfig(
        provider_id="claude_cli",
        type="cli_subprocess",
        endpoint="",
        privacy="external_paid",
        capabilities=("reasoning",),
        cli=cli,
    )
    assert cfg.cli is cli
    assert cfg.secret_ref is None
