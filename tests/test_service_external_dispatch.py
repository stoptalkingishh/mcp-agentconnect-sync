from types import SimpleNamespace

from agentconnect.router.service import _is_external_provider, _payload_for_provider


def test_cli_subprocess_is_treated_as_external_for_payload_selection():
    cfg = SimpleNamespace(type="cli_subprocess")
    assert _is_external_provider(cfg) is True
    assert _payload_for_provider(cfg, "email me at private@example.com", "email me at [EMAIL]") == (
        "email me at [EMAIL]"
    )


def test_cloud_is_external_but_owned_local_is_not():
    assert _is_external_provider(SimpleNamespace(type="cloud")) is True
    assert _is_external_provider(SimpleNamespace(type="local")) is False
