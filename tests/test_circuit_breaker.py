import time

from agentconnect.common.circuit_breaker import CLOSED, HALF_OPEN, OPEN, CircuitBreakerRegistry


def test_closed_by_default():
    cb = CircuitBreakerRegistry()
    assert cb.is_open("p1") is False
    assert cb.status("p1")["state"] == CLOSED


def test_trips_open_at_threshold():
    cb = CircuitBreakerRegistry(failure_threshold=3, cooldown_seconds=60)
    cb.record_failure("p1", "timeout")
    cb.record_failure("p1", "timeout")
    assert cb.is_open("p1") is False  # only 2 failures, not yet tripped
    cb.record_failure("p1", "timeout")
    assert cb.status("p1")["state"] == OPEN
    assert cb.is_open("p1") is True


def test_success_resets_and_closes():
    cb = CircuitBreakerRegistry(failure_threshold=2)
    cb.record_failure("p1", "x")
    cb.record_success("p1")
    assert cb.status("p1") == {
        "state": CLOSED, "consecutive_failures": 0, "opened_at": None, "last_failure_reason": None,
    }


def test_half_open_probe_then_close_on_success():
    cb = CircuitBreakerRegistry(failure_threshold=1, cooldown_seconds=0.05)
    cb.record_failure("p1", "down")
    assert cb.is_open("p1") is True
    time.sleep(0.06)
    assert cb.is_open("p1") is False  # cooldown elapsed -> half-open probe allowed
    assert cb.status("p1")["state"] == HALF_OPEN
    cb.record_success("p1")
    assert cb.status("p1")["state"] == CLOSED


def test_half_open_probe_failure_reopens_immediately():
    cb = CircuitBreakerRegistry(failure_threshold=5, cooldown_seconds=0.05)
    # Force into half-open without needing 5 failures first.
    cb.record_failure("p1", "x")
    cb._states["p1"].state = OPEN
    cb._states["p1"].opened_at = time.time() - 1
    assert cb.is_open("p1") is False  # transitions to half-open
    cb.record_failure("p1", "still down")
    assert cb.status("p1")["state"] == OPEN  # re-opened without needing the full threshold again


def test_per_provider_override():
    cb = CircuitBreakerRegistry(failure_threshold=100, overrides={"p1": {"failure_threshold": 1}})
    cb.record_failure("p1", "x")
    assert cb.status("p1")["state"] == OPEN
    cb.record_failure("p2", "x")
    assert cb.status("p2")["state"] == CLOSED  # p2 uses the default threshold of 100


def test_disabled_via_from_config_never_trips():
    cb = CircuitBreakerRegistry.from_config({"circuit_breaker": {"enabled": False}})
    for _ in range(50):
        cb.record_failure("p1", "x")
    assert cb.is_open("p1") is False


def test_from_config_reads_threshold_and_cooldown():
    cb = CircuitBreakerRegistry.from_config(
        {"circuit_breaker": {"enabled": True, "failure_threshold": 2, "cooldown_seconds": 5}}
    )
    assert cb.failure_threshold == 2
    assert cb.cooldown_seconds == 5
