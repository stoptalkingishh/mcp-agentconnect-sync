"""The Router is the product: it must build and run with the Model Manager package
absent (handoff Goal 2/3). We simulate absence by blocking the import."""

import builtins

import pytest

from agentconnect.router import mcp_server


def test_try_embedded_manager_returns_none_when_uninstalled(monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        # Relative imports pass a truncated name (e.g. "model_manager.residency"),
        # so match on substring rather than a full dotted prefix.
        if "model_manager" in name:
            raise ImportError("simulated: agentconnect-model-manager not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    monkeypatch.delenv("MODEL_MANAGER_URL", raising=False)

    assert mcp_server._try_embedded_manager() is None


def test_build_service_standalone_cloud_only(monkeypatch, tmp_path):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if "model_manager" in name:
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    monkeypatch.delenv("MODEL_MANAGER_URL", raising=False)
    monkeypatch.setenv("AGENTCONNECT_DB", str(tmp_path / "mem.sqlite"))

    svc = mcp_server._build_service()
    status = svc.get_router_status()
    assert status["local_manager"] is None  # no local node, but the router works
    assert "gemini_free" in status["providers"]


def test_public_task_fails_closed_without_cloud_credentials(monkeypatch, tmp_path):
    from agentconnect.common.schemas import TaskConstraints, TaskState, TaskSubmission
    from agentconnect.router.service import RouterService

    svc = RouterService.create(memory=None, local_client=None)  # no local node at all
    sub = TaskSubmission(
        task="Classify: is 'the sky is blue' a question?",
        agent_type="log_summarizer",
        constraints=TaskConstraints(privacy_class="public"),
    )
    summary = svc.submit_task(sub)
    assert summary.status == TaskState.FAILED
    decisions = svc.memory.get_routing_decisions(summary.task_id)
    assert decisions[-1]["selected_provider"] in {
        "gemini_free",
        "groq_free",
        "openai_paid",
        "openrouter_free",
        "xai_free",
    }
