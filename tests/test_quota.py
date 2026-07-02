from agentconnect.common.config import ProviderConfig
from agentconnect.common.memory import SharedMemory
from agentconnect.common.quota import QuotaLedger


def _free_provider():
    return ProviderConfig(
        provider_id="gemini_free", type="cloud", endpoint="x", secret_ref="op://x",
        privacy="external", capabilities=("classification",),
        quota={"kind": "free_tier", "max_daily_requests": 2, "max_daily_tokens": 10000},
    )


def _paid_provider():
    return ProviderConfig(
        provider_id="openai_paid", type="cloud", endpoint="x", secret_ref="op://x",
        privacy="external_paid", capabilities=("hard_reasoning",),
        quota={"kind": "paid", "max_daily_spend_usd": 0.01,
               "price_per_1k_input_usd": 0.0025, "price_per_1k_output_usd": 0.01},
    )


def test_reservation_reduces_remaining_and_blocks_overspend():
    mem = SharedMemory()
    ledger = QuotaLedger(memory=mem)
    cfg = _free_provider()

    r1 = ledger.reserve(cfg, "t1", 100, 100)
    assert r1.granted
    r2 = ledger.reserve(cfg, "t2", 100, 100)
    assert r2.granted
    # third request exceeds the 2-request daily cap while both are reserved
    r3 = ledger.reserve(cfg, "t3", 100, 100)
    assert not r3.granted
    assert r3.reason == "daily_request_quota_exhausted"


def test_reconcile_persists_usage():
    mem = SharedMemory()
    ledger = QuotaLedger(memory=mem)
    cfg = _free_provider()
    r = ledger.reserve(cfg, "t1", 100, 50)
    ledger.reconcile(r, cfg, act_input=120, act_output=40)
    rem = ledger.remaining(cfg)
    assert rem["tokens_remaining"] == 10000 - 160
    assert rem["requests_remaining"] == 1


def test_paid_budget_enforced():
    mem = SharedMemory()
    ledger = QuotaLedger(memory=mem)
    cfg = _paid_provider()
    # 1000 output tokens -> $0.01 exactly, at the cap; a bit more must be blocked.
    ok, reason = ledger.can_reserve(cfg, 0, 2000)
    assert not ok
    assert reason == "daily_spend_budget_exhausted"


def test_local_gpu_is_unlimited():
    mem = SharedMemory()
    ledger = QuotaLedger(memory=mem)
    cfg = ProviderConfig(
        provider_id="local_r9700", type="local", endpoint="x", secret_ref="op://x",
        privacy="local_only", capabilities=("coding",), quota={"kind": "local_gpu"},
    )
    ok, _ = ledger.can_reserve(cfg, 100000, 100000)
    assert ok
