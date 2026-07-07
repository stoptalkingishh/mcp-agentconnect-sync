"""Cloud generation delegated to LiteLLM, beneath the gateway.

Routing/privacy/spend policy is unchanged and sits ABOVE this seam; here we only
verify the transport swap: the resolved API key is passed EXPLICITLY to LiteLLM and
never leaks into the environment, OpenAI-compatible vs native (`litellm_model`)
routing, the stub fallback, and that real LiteLLM maps through our GatewayResult.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from agentconnect.common.config import ProviderConfig
from agentconnect.common.schemas import GenerateRequest
from agentconnect.router.gateway import ProviderGateway


class _FakeSecrets:
    def __init__(self, key):
        self.key = key

    def resolve(self, ref):
        return self.key


def _cloud_cfg(**kw) -> ProviderConfig:
    base = dict(
        provider_id="openai_paid", type="cloud", endpoint="https://api.openai.com/v1",
        privacy="external_paid", capabilities=("coding",), secret_ref="op://AI/openai/api-key",
    )
    base.update(kw)
    return ProviderConfig(**base)


def _req(model="gpt-4o") -> GenerateRequest:
    return GenerateRequest(
        request_id="r", task_id="t", model_id=model,
        messages=[{"role": "user", "content": "hi"}], max_output_tokens=32, temperature=0.2,
    )


def _resp(content, pt, ct):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=pt, completion_tokens=ct),
    )


# --------------------------------------------------------------------------- #
def test_cloud_call_routes_through_litellm_openai_compatible_with_key_isolated():
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _resp("reply from litellm", 11, 7)

    gw = ProviderGateway(secret_resolver=_FakeSecrets("sk-SECRET"), completion_fn=fake_completion)
    result = gw.call(_cloud_cfg(), _req("gpt-4o"))

    # Response mapping into GatewayResult.
    assert result.output_text == "reply from litellm"
    assert (result.input_tokens, result.output_tokens) == (11, 7)
    assert result.provider == "openai_paid" and result.model == "gpt-4o"
    # Default (no litellm_model) = OpenAI-compatible against the endpoint.
    assert captured["model"] == "openai/gpt-4o"
    assert captured["api_base"] == "https://api.openai.com/v1"
    assert captured["max_tokens"] == 32 and captured["temperature"] == 0.2
    # Key isolation: passed explicitly to LiteLLM, NEVER written to the environment.
    assert captured["api_key"] == "sk-SECRET"
    assert "sk-SECRET" not in os.environ.values()
    assert "OPENAI_API_KEY" not in os.environ


def test_cloud_litellm_model_selects_native_provider():
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _resp("gemini says hi", 5, 3)

    gw = ProviderGateway(secret_resolver=_FakeSecrets("sk-G"), completion_fn=fake_completion)
    cfg = _cloud_cfg(
        provider_id="gemini_free", endpoint="https://generativelanguage.googleapis.com",
        litellm_model="gemini/gemini-1.5-flash",
    )
    result = gw.call(cfg, _req("ignored-when-native"))

    # The native model string is used verbatim (so Gemini's own API is reached).
    assert captured["model"] == "gemini/gemini-1.5-flash"
    assert captured["api_base"] == "https://generativelanguage.googleapis.com"
    assert captured["api_key"] == "sk-G"
    assert result.output_text == "gemini says hi"


def test_cloud_falls_back_to_stub_without_a_key():
    class _NoSecrets:
        def resolve(self, ref):
            raise RuntimeError("no vault reachable")

    called = {"n": 0}

    def fake_completion(**kwargs):
        called["n"] += 1
        return _resp("should not happen", 1, 1)

    gw = ProviderGateway(secret_resolver=_NoSecrets(), completion_fn=fake_completion)
    result = gw.call(_cloud_cfg(), _req())

    assert result.output_text.startswith("[cloud-stub:")
    assert called["n"] == 0  # no key -> LiteLLM is never called


def test_cloud_stub_fallback_when_litellm_raises():
    def boom(**kwargs):
        raise RuntimeError("provider 500")

    gw = ProviderGateway(secret_resolver=_FakeSecrets("sk-x"), completion_fn=boom)
    result = gw.call(_cloud_cfg(), _req())
    # A live-call failure degrades to the deterministic stub, never leaks the reason.
    assert result.output_text.startswith("[cloud-stub:")


def test_cloud_real_litellm_via_mock_response():
    # Exercise REAL LiteLLM (mapping + response shape) offline via its mock_response
    # hook — no network, no credentials. Skipped if the [cloud] extra is absent.
    litellm = pytest.importorskip("litellm")
    import functools

    fn = functools.partial(litellm.completion, mock_response="mocked reply")
    gw = ProviderGateway(secret_resolver=_FakeSecrets("sk-x"), completion_fn=fn)
    result = gw.call(_cloud_cfg(), _req("gpt-4o"))

    assert result.output_text == "mocked reply"
    assert result.input_tokens >= 0 and result.output_tokens >= 0
