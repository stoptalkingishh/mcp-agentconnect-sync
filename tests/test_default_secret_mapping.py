from pathlib import Path

import yaml


def test_default_cloud_secret_refs_map_to_documented_environment_variables():
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "config" / "secrets.example.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["secret_refs"] | {
        "op://AI/gemini/api-key": {"kind": "env", "var": "GEMINI_API_KEY"},
        "op://AI/groq/api-key": {"kind": "env", "var": "GROQ_API_KEY"},
        "op://AI/openrouter/api-key": {"kind": "env", "var": "OPENROUTER_API_KEY"},
        "op://AI/xai/api-key": {"kind": "env", "var": "XAI_API_KEY"},
    } == config["secret_refs"]
