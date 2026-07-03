# Review Findings

This note captures the main issues found in the current repository review.

## Findings

1. `packages/agentconnect-router/src/agentconnect/router/gateway.py` treats all cloud providers as OpenAI-compatible and always posts to `/chat/completions`. That is fine for OpenAI/Groq-compatible endpoints, but it will not work for providers that require a different request/response shape, which is why provider-specific adapters are needed.

2. `packages/agentconnect-router/src/agentconnect/router/gateway.py` silently falls back to a deterministic stub when a cloud secret is missing or a cloud request fails. That keeps demos working, but in production it can make a failed cloud dispatch look like success.

3. `packages/agentconnect-router/src/agentconnect/router/service.py` clamps the output before writing the artifact to shared memory. That means the stored artifact is not actually the full output, which weakens the “read large outputs back in chunks” story.

4. `packages/agentconnect-router/src/agentconnect/router/provisioning.py` stores rented-node timestamps with a default of `0.0`, and `packages/agentconnect-router/src/agentconnect/router/service.py` calls the pool without passing a real time value. That can make warm rented nodes look stale when idle reaping runs.

5. `packages/agentconnect-core/src/agentconnect/common/quota.py` keeps live reservations in process memory only. If more than one router process runs against the same shared memory database, quota oversubscription becomes possible.

6. `README.md` and `docs/ARCHITECTURE.md` overstate maturity relative to the implementation in a few places. The stack is solid, but it is still closer to a structured prototype than a production-ready control plane.

## Verification

The current test suite passes in the local sandbox with the installed dependencies:

`89 passed, 1 warning`
