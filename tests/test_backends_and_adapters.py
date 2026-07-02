"""Real backend + vendor-adapter code paths, exercised offline via httpx mock
transports (no live servers or credentials)."""

import httpx

from agentconnect.common.schemas import AvailableModel, GenerateRequest, NodeSpec, NodeTrust
from agentconnect.model_manager.backends import OpenAICompatibleBackend
from agentconnect.router.provisioning import RunPodProvisioner


def test_openai_compatible_backend_generate():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        body = __import__("json").loads(request.content)
        assert body["model"] == "qwen3.6-35b-a3b"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello from vllm"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 5},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://vllm.local")
    backend = OpenAICompatibleBackend(
        base_url="http://vllm.local",
        models=[AvailableModel(model_id="qwen3.6-35b-a3b")],
        client=client,
    )
    resp = backend.generate(
        GenerateRequest(request_id="r", task_id="t", model_id="qwen3.6-35b-a3b",
                        messages=[{"role": "user", "content": "hi"}], max_output_tokens=32)
    )
    assert resp.output_text == "hello from vllm"
    assert resp.input_tokens == 12 and resp.output_tokens == 5


def test_runpod_provisioner_lifecycle():
    state = {"terminated": False}

    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.path
        if request.method == "POST" and p.endswith("/pods"):
            return httpx.Response(200, json={"id": "pod123", "desiredStatus": "PENDING"})
        if request.method == "GET" and p.endswith("/pods/pod123"):
            return httpx.Response(200, json={"id": "pod123", "desiredStatus": "RUNNING", "publicIp": "1.2.3.4"})
        if request.method == "DELETE" and p.endswith("/pods/pod123"):
            state["terminated"] = True
            return httpx.Response(200, json={})
        return httpx.Response(404, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://rest.runpod.io/v1")
    prov = RunPodProvisioner(api_key="dummy", client=client, poll_interval=0, sleep=lambda *_: None)

    spec = NodeSpec(provider_id="rented_h100_pool", vendor="runpod", instance_type="1xH100-80GB",
                    max_hourly_usd=3.5, trust=NodeTrust(ephemeral=True))
    h = prov.provision(spec)
    assert h.node_id == "pod123" and h.state.value == "provisioning"
    h = prov.wait_ready(h)
    assert h.state.value == "ready" and h.manager_endpoint == "https://1.2.3.4:8443"
    h = prov.terminate(h)
    assert h.state.value == "terminated" and state["terminated"] is True


def test_provisioner_for_selects_stub_for_generic():
    from agentconnect.common.config import load_providers
    from agentconnect.router.provisioning import StubProvisioner, provisioner_for

    cfg = load_providers().providers["rented_h100_pool"]  # vendor: generic
    assert isinstance(provisioner_for(cfg), StubProvisioner)
