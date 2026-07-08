"""Integration tests for the circuit-breaker/compression wiring through
ProviderGateway and RouterService (native ports of OmniRoute resilience/
compression concepts)."""

from unittest.mock import MagicMock

from agentconnect.common.compression import Compressor
from agentconnect.common.config import ProviderConfig
from agentconnect.common.memory import SharedMemory
from agentconnect.common.schemas import GenerateRequest
from agentconnect.router.gateway import GatewayResult, ProviderGateway
from agentconnect.router.service import RouterService

_CFG = ProviderConfig(
    provider_id="test_cloud", type="cloud", endpoint="https://fake",
    privacy="external", capabilities=(), secret_ref="op://x/y",
)


def _req(content="hello"):
    return GenerateRequest(request_id="r1", task_id="t1", model_id="m1", messages=[{"role": "user", "content": content}])


def test_gateway_notifies_failure_even_though_stub_masks_it():
    events = []
    secrets = MagicMock()
    secrets.resolve.return_value = "fake-key"
    gw = ProviderGateway(secret_resolver=secrets, on_call_result=lambda *a: events.append(a))
    gw._http_openai_compatible = MagicMock(side_effect=RuntimeError("connection refused"))

    result = gw.call(_CFG, _req())

    assert result.output_text.startswith("[cloud-stub:")  # stub fallback unchanged
    assert events == [("test_cloud", False, "connection refused")]


def test_gateway_notifies_success():
    events = []
    secrets = MagicMock()
    secrets.resolve.return_value = "fake-key"
    gw = ProviderGateway(secret_resolver=secrets, on_call_result=lambda *a: events.append(a))
    gw._http_openai_compatible = MagicMock(return_value=GatewayResult("ok", 5, 5, "test_cloud", "m1"))

    result = gw.call(_CFG, _req())

    assert result.output_text == "ok"
    assert events == [("test_cloud", True, None)]


def test_gateway_compresses_outbound_message_before_http_call():
    secrets = MagicMock()
    secrets.resolve.return_value = "fake-key"
    compressor = Compressor(min_chars_to_compress=5)
    gw = ProviderGateway(secret_resolver=secrets, compressor=compressor)
    gw._http_openai_compatible = MagicMock(return_value=GatewayResult("ok", 5, 5, "test_cloud", "m1"))

    long_prose = "It is worth noting that " + ("padding " * 10)
    gw.call(_CFG, _req(long_prose))

    sent_req = gw._http_openai_compatible.call_args[0][1]
    assert "It is worth noting that" not in sent_req.messages[0]["content"]


def test_gateway_no_compressor_leaves_message_untouched():
    secrets = MagicMock()
    secrets.resolve.return_value = "fake-key"
    gw = ProviderGateway(secret_resolver=secrets)
    gw._http_openai_compatible = MagicMock(return_value=GatewayResult("ok", 5, 5, "test_cloud", "m1"))

    text = "It is worth noting that this stays untouched."
    gw.call(_CFG, _req(text))

    sent_req = gw._http_openai_compatible.call_args[0][1]
    assert sent_req.messages[0]["content"] == text


def test_router_service_circuit_breaker_trips_after_repeated_gateway_failures():
    svc = RouterService.create(memory=SharedMemory())
    assert svc.circuit_breaker is not None
    for _ in range(svc.circuit_breaker.failure_threshold):
        svc._on_gateway_result("gemini_free", False, "timeout")
    status = svc.circuit_breaker.status("gemini_free")
    assert status["state"] == "open"

    # Circuit-open providers show up as rejected once refreshed into the engine.
    svc._refresh_circuit_state()
    assert "gemini_free" in svc.engine._circuit_open


def test_get_provider_status_includes_circuit_for_cloud_only():
    svc = RouterService.create(memory=SharedMemory())
    statuses = {p["provider"]: p for p in svc.get_provider_status()}
    assert "circuit" in statuses["gemini_free"]
    assert "circuit" not in statuses["local_r9700"]


def test_get_provider_metrics_shape():
    svc = RouterService.create(memory=SharedMemory())
    out = svc.get_provider_metrics("gemini_free")
    assert len(out) == 1
    entry = out[0]
    assert entry["provider"] == "gemini_free"
    assert entry["samples"] == 0
    assert entry["latency_ms"] == {"p50": None, "p95": None, "p99": None}
    assert "circuit" in entry
    assert "compression" in entry


def test_explain_route_returns_recorded_decision():
    svc = RouterService.create(memory=SharedMemory())
    result = svc.simulate_route("write a haiku about testing")
    assert result["task_id"].startswith("sim_")
    # simulate_route is side-effect-free: nothing was recorded for that sim task id.
    assert svc.explain_route(result["task_id"]) == {
        "error": "no_routing_decision_recorded", "task_id": result["task_id"],
    }


def test_get_compression_status_shape():
    svc = RouterService.create(memory=SharedMemory())
    status = svc.get_compression_status()
    assert status["enabled"] is True
    assert "tool_output" in status["apply_to"]
    assert status["providers"] == {}
