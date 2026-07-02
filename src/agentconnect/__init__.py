"""agentconnect — MCP Agent Router + Local Model Manager.

A two-service agent infrastructure:

  * :mod:`agentconnect.router` — the global control plane (Agent Router MCP):
    classifies tasks, enforces privacy/quota policy, scores providers
    deterministically, dispatches work, and returns compact summaries + artifact
    references.
  * :mod:`agentconnect.model_manager` — the local inference control plane (Local
    Model Manager): model residency, admission control, and generation.

Design rule (handoff §26): the router owns decisions, the model manager owns
local execution, the secrets manager owns credentials, agents own task work,
shared memory owns state and artifacts.
"""

__version__ = "0.1.0"
