"""Direct-to-user spend authorization: every paid/rented charge passes through a
deterministic human gate (not the stochastic agent), and a missing budget prompts
the user directly."""

from agentconnect.common.authorization import (
    CallbackSpendAuthorizer,
    ChargeRequest,
    ConsoleSpendAuthorizer,
    DenyingSpendAuthorizer,
)
from agentconnect.common.memory import SharedMemory
from agentconnect.common.schemas import TaskConstraints, TaskState, TaskSubmission
from agentconnect.model_manager.residency import ResidencyManager
from agentconnect.router.local_client import InProcessLocalClient
from agentconnect.router.provisioning import StubProvisioner
from agentconnect.router.service import RouterService


def _rented_sub():
    return TaskSubmission(
        task="huge private reasoning job", agent_type="repo_scout",
        constraints=TaskConstraints(privacy_class="repo_sensitive",
                                    allow_external=False, allow_rented=True),
    )


def _rented_svc(authorizer):
    return RouterService.create(
        memory=SharedMemory(), local_client=None,
        provisioner=StubProvisioner(),
        rented_client_factory=lambda cfg, h: InProcessLocalClient(ResidencyManager()),
        authorizer=authorizer,
    )


def test_default_denies_paid_and_rented():
    svc = _rented_svc(DenyingSpendAuthorizer())
    svc.set_budget(50.0, "monthly")  # budget set, but no confirmation channel
    summary = svc.submit_task(_rented_sub())
    assert summary.status == TaskState.REJECTED
    assert "not confirmed" in (summary.summary or "").lower()


def test_callback_confirm_approves_and_receives_charge_details():
    seen = {}

    def confirm(req: ChargeRequest) -> bool:
        seen["provider"] = req.provider
        seen["kind"] = req.kind
        seen["cost"] = req.estimated_cost_usd
        return True

    svc = _rented_svc(CallbackSpendAuthorizer(confirm))
    svc.set_budget(50.0, "monthly")
    summary = svc.submit_task(_rented_sub())
    assert summary.status == TaskState.COMPLETE
    assert seen["provider"] == "rented_h100_pool" and seen["kind"] == "rented_gpu"
    assert seen["cost"] > 0


def test_callback_decline_blocks_charge():
    svc = _rented_svc(CallbackSpendAuthorizer(lambda req: False))
    svc.set_budget(50.0, "monthly")
    assert svc.submit_task(_rented_sub()).status == TaskState.REJECTED


def test_missing_budget_prompts_user_directly_then_proceeds():
    # No budget set. The authorizer's request_budget supplies one directly (deterministic,
    # not via the agent); the router then routes to the rented node and confirms the charge.
    prompted = {"count": 0}

    def request_budget(period):
        prompted["count"] += 1
        return {"amount_usd": 25.0, "period": "monthly"}

    svc = _rented_svc(CallbackSpendAuthorizer(lambda req: True, request_budget_fn=request_budget))
    assert svc.get_budget_status()["configured"] is False
    summary = svc.submit_task(_rented_sub())
    assert prompted["count"] == 1
    assert summary.status == TaskState.COMPLETE
    assert svc.get_budget_status()["configured"] is True  # saved for next time


def test_missing_budget_user_declines():
    svc = _rented_svc(CallbackSpendAuthorizer(lambda req: True, request_budget_fn=lambda p: None))
    summary = svc.submit_task(_rented_sub())
    assert summary.status == TaskState.REJECTED
    assert "budget" in (summary.recommended_next_action or "").lower()


def test_console_authorizer_parses_prompts():
    answers = iter(["12.50", "weekly"])  # amount, period
    auth = ConsoleSpendAuthorizer(input_fn=lambda *_: next(answers), output_fn=lambda *_: None)
    got = auth.request_budget("monthly")
    assert got == {"amount_usd": 12.5, "period": "weekly"}

    yes = ConsoleSpendAuthorizer(input_fn=lambda *_: "y", output_fn=lambda *_: None)
    no = ConsoleSpendAuthorizer(input_fn=lambda *_: "", output_fn=lambda *_: None)
    req = ChargeRequest(provider="openai_paid", kind="paid_cloud", estimated_cost_usd=0.01,
                        task_summary="x")
    assert yes.confirm_charge(req) is True
    assert no.confirm_charge(req) is False


def test_free_local_tasks_need_no_confirmation():
    # A public task on free/local must NOT trigger the spend gate.
    svc = RouterService.create(
        memory=SharedMemory(), local_client=InProcessLocalClient(ResidencyManager()),
        authorizer=DenyingSpendAuthorizer(),
    )
    summary = svc.submit_task(
        TaskSubmission(task="Summarize: the sky is blue.", agent_type="log_summarizer",
                       constraints=TaskConstraints(privacy_class="public"))
    )
    assert summary.status == TaskState.COMPLETE  # denied authorizer irrelevant to free/local
