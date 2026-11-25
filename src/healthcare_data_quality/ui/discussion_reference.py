"""Immutable, non-actionable reference to an earlier displayed Agent B suggestion.

Retain evidence and model guidance during a reviewer-led alternative discussion,
without restoring stale decision controls or bypassing source/model review gates.
"""
from __future__ import annotations

from typing import Any


def capture_discussion_reference(suggestion: dict[str, Any], reviewed_options: list[dict[str, Any]]) -> dict[str, Any]:
    """Store display data only, never executable operations or model approval tokens.

    The caller must pass only options that passed the existing model assessment,
    source scope and cross-finding conflict filters.
    """
    text = str(suggestion.get("suggestion") or "").strip()
    choices = []
    for option in reviewed_options:
        preview = []
        for row in option.get("rows") or []:
            if not isinstance(row, dict):
                continue
            preview.append({key: ("" if row.get(key) is None else str(row.get(key))) for key in ("Location", "Before", "After")})
        choices.append({
            "label": str(option.get("label") or "Previously proposed action"),
            "rows": preview,
            "provenance": str(option.get("provenance") or ""),
        })
    return {"summary": text, "choices": choices}


def should_show_discussion_reference(mode: str | None, completed: bool, reference: dict | None) -> bool:
    """Superseded proposals are retained for reasoning/audit, never rendered."""
    return False
