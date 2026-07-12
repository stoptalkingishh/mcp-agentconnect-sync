from agentconnect.common.config import load_providers


def test_omniroute_gateway_is_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("OMNIROUTE_BASE_URL", raising=False)
    direct = load_providers().providers["groq_free"]
    assert direct.endpoint == "https://api.groq.com/openai/v1"
    assert direct.model_prefix == ""

    monkeypatch.setenv("OMNIROUTE_BASE_URL", "http://127.0.0.1:20128/")
    proxied = load_providers().providers["groq_free"]
    assert proxied.endpoint == "http://127.0.0.1:20128/v1/providers/groq"
    assert proxied.model_prefix == "groq/"
    assert proxied.model_map["qwen3.6-35b-a3b"] == "qwen/qwen3-32b"
