"""Federated work-queue demo — the "open it up to other agents / a friend's
compute" story, end-to-end, offline. No GPU, no cloud credentials, no network.

Run it:

    python examples/federation_demo.py

What it shows, in order:

  1. A broker enqueues tickets across privacy classes (public, low_sensitive,
     repo_sensitive, secret_sensitive), each with capability requirements.
  2. The payload-free OPERATOR VIEW: stats() + open_capability_requirements()
     + list_tickets() — the dashboard a human spot-checks, with no task body
     or task_id ever exposed.
  3. A FRIEND'S EXTERNAL BOX (attested tier=external) drains what it is allowed
     to: it claims the public ticket, is handed the redacted payload under its
     lease, runs it on its OWN box, and reports. Because it is untrusted, the
     result lands `in_review` — never silently accepted as truth.
  4. THE AUTHORIZATION BOUNDARY (the headline): the same external box is refused
     the repo_sensitive ticket — recomputed live from routing.yaml, fail-closed.
     The ticket stays open, its payload never leaves the broker.
  5. MY LOCAL BOX (tier=local_only, most trusted) claims the repo_sensitive
     ticket and its report is accepted immediately (trusted fast-path).
  6. HUMAN SPOT-CHECK: the in_review backlog is triaged; a local_only reviewer
     approves the external result, promoting it to done.
  7. secret_sensitive is parked at enqueue — claimable by NObody, not even
     local_only. It never reaches any LLM/node.

Everything runs against one in-process SQLite-backed SharedMemory and the live
routing config — the exact same authorization mapping the router uses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "agentconnect-core" / "src"))

from agentconnect.common.config import load_routing  # noqa: E402
from agentconnect.common.memory import SharedMemory  # noqa: E402
from agentconnect.common.schemas import PrivacyClass, WorkerResult  # noqa: E402
from agentconnect.common.workqueue import WorkQueue  # noqa: E402

LOCAL = "local_only"        # my own box — most trusted
EXTERNAL = "external"       # a friend's box / rented public compute — least trusted


def banner(n: int, title: str) -> None:
    print(f"\n{'=' * 72}\n  {n}. {title}\n{'=' * 72}")


def show(label: str, obj: object) -> None:
    print(f"\n  {label}:")
    print("   " + json.dumps(obj, ensure_ascii=False, indent=2, default=str).replace("\n", "\n   "))


def run_worker(wq: WorkQueue, *, name: str, identity: str, tier: str, caps: set[str]) -> None:
    """One iteration of a compute contributor: claim -> fetch payload under the
    lease -> 'run' it on this box -> report. The broker never executes; compute
    happens here, on the contributor's machine."""
    claimed = wq.claim_next(identity, tier, capabilities=caps, max=1)
    if not claimed:
        print(f"  [{name}] nothing authorized+available to claim (tier={tier}, caps={sorted(caps)})")
        return
    t = claimed[0]
    tid, token = t["ticket_id"], t["lease_token"]
    print(f"  [{name}] claimed {tid}  (class={t['privacy_class']}, lease held)")

    # Authorized delivery seam: the redacted body crosses to the lease holder;
    # the internal task_id / un-redacted submission never does.
    body = wq.payload_for(identity, tid, token, tier)
    print(f"  [{name}] payload delivered under lease: {body['payload']!r}")

    # ... the contributor runs the task on its own runtime here ...
    result = WorkerResult(status="completed", summary=f"handled by {name}", confidence=0.9)
    rep = wq.report(identity, tier, tid, token, result)
    print(f"  [{name}] reported -> ticket_status={rep['ticket_status']} "
          f"result_status={rep['result_status']}")


def main() -> None:
    mem = SharedMemory()                 # in-memory; pass a path to persist
    routing = load_routing()             # the live authorization mapping
    broker = WorkQueue(mem, routing)

    # -- 1. enqueue across privacy classes --------------------------------
    banner(1, "Broker enqueues work across privacy classes")
    public = broker.add(task="summarize a public RFC", origin="planner",
                        privacy_class=PrivacyClass.public,
                        payload="Summarize RFC 8259 in 3 bullets.",
                        required_capabilities=["python"])
    repo = broker.add(task="refactor an internal module", origin="planner",
                      privacy_class=PrivacyClass.repo_sensitive,
                      payload="Refactor src/billing/ledger.py for idempotency.",
                      required_capabilities=["python"])
    secret = broker.add(task="rotate a live credential", origin="planner",
                        privacy_class=PrivacyClass.secret_sensitive,
                        payload="PROD_DB_PASSWORD=hunter2")
    print(f"  enqueued: public={public['ticket_id']}  repo_sensitive={repo['ticket_id']}  "
          f"secret_sensitive={secret['ticket_id']}")
    print(f"  secret_sensitive parked at enqueue -> park_reason={secret['park_reason']!r} "
          "(never leasable by any tier)")

    # -- 2. payload-free operator view ------------------------------------
    banner(2, "Operator view (payload-free): stats + capability demand")
    show("broker.stats()", broker.stats())
    show("open capability requirements", broker.open_capability_requirements())
    show("list_tickets() — no payload, no task_id",
         [{k: r[k] for k in ("ticket_id", "status", "privacy_class", "allowed_tiers",
                             "required_capabilities")} for r in broker.list_tickets()])

    # -- 3. friend's external box drains what it may ----------------------
    banner(3, "A friend's EXTERNAL box contributes compute")
    run_worker(broker, name="friend-external", identity="friend@box",
               tier=EXTERNAL, caps={"python"})

    # -- 4. the authorization boundary (headline) -------------------------
    banner(4, "Authorization boundary — external is refused repo_sensitive")
    denied = broker.claim("friend@box", EXTERNAL, repo["ticket_id"])
    print(f"  external claim of repo_sensitive -> {denied}")
    still = broker.get(repo["ticket_id"])["status"]
    print(f"  repo_sensitive ticket status is still {still!r} — its payload never left the broker")

    # -- 5. my trusted local box takes the sensitive work -----------------
    banner(5, "MY LOCAL box (local_only) handles the sensitive ticket")
    run_worker(broker, name="my-local", identity="me@laptop",
               tier=LOCAL, caps={"python"})

    # -- 6. human spot-check of the untrusted result ----------------------
    banner(6, "Human spot-check — triage the in_review backlog")
    pending = broker.pending_review()
    print(f"  {len(pending)} ticket(s) awaiting review "
          f"(the untrusted external result did NOT auto-accept):")
    for r in pending:
        print(f"    - {r['ticket_id']}  class={r['privacy_class']}  origin={r['origin']}")
    if pending:
        tid = pending[0]["ticket_id"]
        approved = broker.approve("me@laptop", LOCAL, tid)
        print(f"  local_only reviewer approved {tid} -> {approved}")

    # -- 7. final state ---------------------------------------------------
    banner(7, "Final queue state")
    show("broker.stats()", broker.stats())
    print("\n  Done. Public work ran on a friend's box (reviewed before trust); "
          "\n  sensitive work stayed local; secrets never left the broker.")


if __name__ == "__main__":
    main()
