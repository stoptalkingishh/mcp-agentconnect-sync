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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentconnect.common.memory import SharedMemory  # noqa: E402
from agentconnect.common.schemas import TaskConstraints, TaskSubmission  # noqa: E402
from agentconnect.model_manager.residency import ResidencyManager  # noqa: E402
from agentconnect.router.local_client import InProcessLocalClient  # noqa: E402
from agentconnect.router.service import RouterService  # noqa: E402


def show(title: str, obj: object) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    svc = RouterService.create(
        memory=SharedMemory(),  # in-memory; use a path to persist
        local_client=InProcessLocalClient(ResidencyManager()),
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

    show("provider status", svc.get_provider_status())


if __name__ == "__main__":
    main()
