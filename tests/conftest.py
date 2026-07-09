import sys
from pathlib import Path

# Make the packages importable from a source checkout without installing.
# PEP 420 namespace packages: all `agentconnect/` dirs on sys.path merge into
# one `agentconnect` namespace, so `agentconnect.common`, `agentconnect.core`,
# `agentconnect.router`, `agentconnect.model_manager`, `agentconnect.runtime`, and the
# backplane adapters (`agentconnect.api`, `.cli`, `.mcp`, `.linear`) all resolve.
ROOT = Path(__file__).resolve().parents[1]
for _pkg in (
    "agentconnect-core",
    "agentconnect-router",
    "agentconnect-model-manager",
    "agentconnect-runtime",
    "agentconnect-api",
    "agentconnect-cli",
    "agentconnect-mcp",
    "agentconnect-linear",
    "agentconnect-temporal",
):
    _src = ROOT / "packages" / _pkg / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
