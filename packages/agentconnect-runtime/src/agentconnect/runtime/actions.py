"""The action protocol between the model and the runtime.

Local models served through `GenerateRequest` are plain text — there is no
native tool-calling API — so the model replies with one JSON object per turn:

    {"action": "read_file",  "path": "src/app.py"}
    {"action": "write_file", "path": "src/app.py", "content": "..."}
    {"action": "list_dir",   "path": "."}
    {"action": "shell",      "command": "pytest -q"}
    {"action": "finish",     "summary": "...", "confidence": 0.8,
     "risks": [], "recommended_next_action": "..."}

Parsing is forgiving on transport (code fences, surrounding prose) but strict
on shape: a JSON object with an unknown/malformed action becomes an ``invalid``
action so the loop can show the model an error observation and let it retry.
A reply with no JSON object at all is treated as a free-form final answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

KNOWN_ACTIONS = ("read_file", "write_file", "list_dir", "shell", "finish")

_REQUIRED_ARGS = {
    "read_file": ("path",),
    "write_file": ("path", "content"),
    "list_dir": (),
    "shell": ("command",),
    "finish": (),
}


@dataclass(frozen=True)
class Action:
    kind: str  # one of KNOWN_ACTIONS, or "invalid"
    args: dict[str, Any] = field(default_factory=dict)
    # True when the model answered in prose and we coerced it to finish.
    freeform: bool = False


# Fields where an empty string is a legitimate value (e.g. creating an empty file).
_ALLOW_EMPTY = frozenset({"content"})


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the best JSON object out of `text`: the first one carrying a known
    action, else the first with an ``action`` key (for a precise invalid error),
    else the first object at all. Replies often mix prose or incidental JSON
    (scores, quoted data) with the real action — the action must still win."""
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    start = text.find("{")
    while start != -1:
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(obj, dict):
            if obj.get("action") in KNOWN_ACTIONS:
                return obj
            objects.append(obj)
        start = text.find("{", start + end)
    for obj in objects:
        if "action" in obj:
            return obj
    return objects[0] if objects else None


def parse_action(text: str) -> Action:
    obj = _extract_json_object(text or "")
    if obj is None:
        # Prose reply: treat the whole text as the final answer.
        return Action("finish", {"summary": (text or "").strip()}, freeform=True)
    kind = obj.get("action")
    if kind not in KNOWN_ACTIONS:
        return Action("invalid", {"error": f"unknown action {kind!r}", "raw": obj})
    missing = [
        k
        for k in _REQUIRED_ARGS[kind]
        if not isinstance(obj.get(k), str) or (not obj[k] and k not in _ALLOW_EMPTY)
    ]
    if missing:
        return Action(
            "invalid",
            {"error": f"action {kind!r} missing required field(s): {', '.join(missing)}", "raw": obj},
        )
    args = {k: v for k, v in obj.items() if k != "action"}
    return Action(kind, args)
