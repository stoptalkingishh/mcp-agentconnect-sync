"""End-to-end demo — no GPU, no cloud credentials required.

Runs the full router flow against an in-process Local Model Manager (stub
backend) and prints the compact summaries the manager agent would receive, then
demonstrates reading detail back on demand (context virtualization).

    python examples/demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _pkg in ("agentconnect-core", "agentconnect-router", "agentconnect-model-manager"):
    sys.path.insert(0, str(_ROOT / "packages" / _pkg / "src"))

from agentconnect.common.memory import SharedMemory  # noqa: E402
from agentconnect.common.schemas import TaskConstraints, TaskSubmission  # noqa: E402
from agentconnect.model_manager.residency import ResidencyManager  # noqa: E402
from agentconnect.router.local_client import InProcessLocalClient  # noqa: E402
from agentconnect.router.service import RouterService  # noqa: E402


def show(title: str, obj: object) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    from agentconnect.common.authorization import CallbackSpendAuthorizer

    # A direct-to-user spend gate. In a real app this pops a native confirmation UI;
    # here we auto-approve and print, to show the deterministic money gate firing.
    def confirm(req):
        print(f"   [user spend prompt] {req.describe()} -> APPROVED")
        return True

    # Inject an in-process client for the "rented" node too, so Goal 4 runs offline.
    rented_factory = lambda cfg, handle: InProcessLocalClient(ResidencyManager())
    svc = RouterService.create(
        memory=SharedMemory(),  # in-memory; use a path to persist
        local_client=InProcessLocalClient(ResidencyManager()),
        rented_client_factory=rented_factory,
        authorizer=CallbackSpendAuthorizer(confirm),
    )

    show("router status", svc.get_router_status())

    # 1) A private-repo coding task -> must stay local.
    s1 = svc.submit_task(
        TaskSubmission(
            task="Refactor the token refresh path in src/auth/session.py (private repo).",
            agent_type="patch_worker",
            constraints=TaskConstraints(privacy_class="repo_sensitive"),
        )
    )
    show("task 1 summary (repo_sensitive → local)", s1.model_dump(mode="json"))
    show("task 1 routing decision", svc.memory.get_routing_decisions(s1.task_id)[-1])

    # Read the output back on demand — not returned inline.
    out_ref = s1.artifacts.get("output")
    if out_ref:
        show("task 1 output chunk (read on demand)", svc.read_artifact_chunk(out_ref, 0, 200))

    # 2) A public classification task -> free cloud is eligible.
    s2 = svc.submit_task(
        TaskSubmission(
            task="Classify whether this sentence is a question: 'The sky is blue.'",
            agent_type="log_summarizer",
            constraints=TaskConstraints(privacy_class="public"),
        )
    )
    show("task 2 summary (public)", s2.model_dump(mode="json"))

    # 3) A secret-bearing task -> blocked from every LLM.
    s3 = svc.submit_task(
        TaskSubmission(
            task="Update deploy with key sk-ABCD1234EFGH5678IJKL and rotate it.",
            agent_type="patch_worker",
        )
    )
    show("task 3 summary (secret_sensitive → blocked)", s3.model_dump(mode="json"))

    # Budget is mandatory before any paid/rented spend — no silent default.
    show("budget status (before setting)", svc.get_budget_status())
    show("set budget $25/month", svc.set_budget(25.0, "monthly"))

    # 4) Even with a budget + allow_rented, an owned local node is free and wins —
    #    so this stays local and NO spend/confirmation is needed (correct behavior).
    s4 = svc.submit_task(
        TaskSubmission(
            task="Reason over this large private architecture doc and propose a refactor.",
            agent_type="repo_scout",
            constraints=TaskConstraints(
                privacy_class="repo_sensitive", allow_external=False, allow_rented=True
            ),
        )
    )
    show("task 4 routing decision (owned local wins, free)", svc.memory.get_routing_decisions(s4.task_id)[-1])

    # 5) Same task on a node WITHOUT an owned local box -> rented GPU is the only
    #    option. The direct spend gate prompts the user (see the stdout line).
    rented_only = RouterService.create(
        memory=SharedMemory(), local_client=None,
        rented_client_factory=rented_factory,
        authorizer=CallbackSpendAuthorizer(confirm),
    )
    rented_only.set_budget(25.0, "monthly")
    s5 = rented_only.submit_task(
        TaskSubmission(
            task="Reason over this large private doc on rented hardware.",
            agent_type="repo_scout",
            constraints=TaskConstraints(
                privacy_class="repo_sensitive", allow_external=False, allow_rented=True
            ),
        )
    )
    show("task 5 summary (rented; user-approved spend)", s5.model_dump(mode="json"))
    show("task 5 budget status (spend accrued)", rented_only.get_budget_status())

    show("provider status", svc.get_provider_status())
    # Phase 6: learned scorecards accumulated from the dispatches above.
    show("provider scorecards (Phase 6)", svc.get_provider_scorecards())


if __name__ == "__main__":
    main()
