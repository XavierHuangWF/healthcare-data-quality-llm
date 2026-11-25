"""Local, review-only memory of human-approved Agent C finding resolutions.

This is intentionally separate from LLM conversation history, remediation
execution, and audit caches. Records are suggestions, never authorization to
change a new dataset. No external service is used.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import re
import threading
from typing import Any
from uuid import uuid4

LEGACY_PATH = (Path.home() / ".local" / "share" /
               "healthcare-data-quality-llm" / "agent-c-review-memory.jsonl")
DEFAULT_PATH = Path(__file__).resolve().parents[3] / "memory" / "approved_decisions.jsonl"
_VERSION = 1
_LOCK = threading.RLock()
_ALLOWED_OPERATIONS = {"replace_cells", "rename_headers", "drop_column", "drop_rows", "merge_rows", "flag_cells"}


def _location(path: Path | str | None = None) -> Path:
    if path is not None:
        return Path(path)
    configured = os.environ.get("HDQ_AGENT_C_REVIEW_MEMORY")
    if configured:
        return Path(configured)
    # First use after upgrading: retain existing approved decisions instead
    # of silently losing them when moving to the project-local memory folder.
    if not DEFAULT_PATH.exists() and LEGACY_PATH.is_file():
        with _LOCK:
            if not DEFAULT_PATH.exists():
                DEFAULT_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                temporary = DEFAULT_PATH.with_suffix(".migrating")
                temporary.write_bytes(LEGACY_PATH.read_bytes())
                temporary.chmod(0o600)
                os.replace(temporary, DEFAULT_PATH)
    return DEFAULT_PATH


def _fingerprint(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


def _identity(finding: dict, source: Path) -> dict[str, str]:
    # Do not rely on the ephemeral sequential finding ID across investigations.
    check_key = str(finding.get("check_key") or "").casefold()
    # `find_invalid_counts` can report very different issues (fractional,
    # negative, missing). Match the *actual problem description*, not just
    # the broad tool name; ignore changing row/column/count numbers.
    title = " ".join(str(finding.get("title") or "").casefold().split())
    pattern = re.sub(r"\b\d+(?:\.\d+)?\b", "#", title)
    return {
        "source_sha256": sha256(Path(source).read_bytes()).hexdigest(),
        "finding_type": check_key + ":" + pattern,
        "evidence_sha256": _fingerprint({
            "title": finding.get("title"),
            "source": finding.get("source"),
            "method": finding.get("method"),
            "evidence": finding.get("evidence") or [],
        }),
    }


def save_reviewed_solution(
    finding: dict, source: Path, *, operations: list[dict] | None = None,
    decision: str = "approved_edit", user_request: str = "",
    agent_response: str = "", path: Path | str | None = None,
    button_label: str | None = None,
) -> dict:
    """Persist only a human-approved result; caller owns the approval gate."""
    if decision not in {"approved_edit", "reviewed_no_change"}:
        raise ValueError("Only completed human review outcomes can be remembered")
    if decision == "approved_edit" and (not isinstance(operations, list) or
                                         not operations or len(operations) > 16):
        raise ValueError("A reviewed edit must include 1–16 validated operations")
    for operation in operations or []:
        if not isinstance(operation, dict) or operation.get("operation") not in _ALLOWED_OPERATIONS:
            raise ValueError("Unsupported remembered cleaning operation")
    record = {
        "schema_version": _VERSION,
        "memory_id": str(uuid4()),
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        **_identity(finding, source),
        "finding_title": str(finding.get("title") or "")[:300],
        "decision": decision,
        "user_request": str(user_request)[:3000],
        "agent_response": str(agent_response)[:3000],
        "operations": operations or [],
        "execution_authorized": False,
    }
    if button_label is not None:
        from healthcare_data_quality.ui.clear_review_actions import reflected_action_label
        record["button_label"] = reflected_action_label(operations, finding, button_label,
            decision="no_change" if decision == "reviewed_no_change" else None)
    serialized = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    if len(serialized.encode("utf-8")) > 1_000_000:
        raise ValueError("Review too large to store in interaction memory")
    destination = _location(path)
    with _LOCK:
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Private, local-only storage; no raw CSV data is copied to this log.
        try:
            destination.parent.chmod(0o700)
        except OSError:
            pass
        fd = os.open(str(destination), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            # fdopen takes ownership of fd in its context; do not mask errors.
            raise
    # Learning is advisory and must never affect transaction success.
    if path is None and decision == "approved_edit":
        try:
            from healthcare_data_quality.learning.adaptive_skill_memory import record_review
            record_review(finding, decision, operations)
        except (OSError, ValueError, RuntimeError, TypeError):
            pass
    return record


def review_memory_identity(record: dict) -> tuple[str, str, str]:
    """Exact evidence identity; used to invalidate superseded historic rules."""
    return (str(record.get("source_sha256") or ""),
            str(record.get("finding_type") or ""),
            str(record.get("evidence_sha256") or ""))


def revoke_reviewed_solution(finding: dict, source: Path,
                             path: Path | str | None = None) -> dict:
    """Append a revision marker so older suggestions aren't silently reused.

    Only applies to the exact original CSV and finding evidence, never to
    unrelated findings or newly approved revisions written after this marker.
    """
    record = {
        "schema_version": _VERSION,
        "memory_id": str(uuid4()),
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        **_identity(finding, source),
        "decision": "revision_reset",
        "operations": [],
        "execution_authorized": False,
    }
    destination = _location(path)
    with _LOCK:
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            destination.parent.chmod(0o700)
        except OSError:
            pass
        fd = os.open(str(destination), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return record


def find_reviewed_solution(finding: dict, source: Path,
                           path: Path | str | None = None) -> dict | None:
    """Prefer exact original dataset+evidence; otherwise show a nonactionable analogy."""
    destination = _location(path)
    if not destination.is_file():
        return None
    wanted = _identity(finding, source)
    recent = deque(maxlen=2000)
    with _LOCK:
        with destination.open("r", encoding="utf-8") as stream:
            for raw in stream:
                if len(raw) <= 1_000_000:
                    recent.append(raw)
    similar = None
    revoked: set[tuple[str, str, str]] = set()
    for raw in reversed(recent):
        try:
            record = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(record, dict) or record.get("schema_version") != _VERSION:
            continue
        identity = review_memory_identity(record)
        if record.get("decision") == "revision_reset":
            revoked.add(identity)
            continue
        if identity in revoked:
            continue
        if record.get("decision") not in {"approved_edit", "reviewed_no_change"}:
            continue
        if record.get("finding_type") != wanted["finding_type"]:
            continue
        if (record.get("source_sha256") == wanted["source_sha256"] and
                record.get("evidence_sha256") == wanted["evidence_sha256"]):
            return {"match": "exact", "record": record}
        if similar is None:
            similar = {"match": "similar", "record": record}
    return similar


def _shown_value(change: dict, field: str) -> str:
    """Display precisely the stored value, including empty strings and NA."""
    if field not in change:
        return "(not saved)"
    return json.dumps(change[field], ensure_ascii=False, default=str)


def _change_description(operation: str, change: dict) -> str:
    """Describe an original-CSV operation, without guessing any value."""
    if operation == "replace_cells":
        return (f"row {change.get('row', '?')}, column {change.get('column', '?')}: "
                f"{_shown_value(change, 'before')} → {_shown_value(change, 'after')}")
    if operation == "rename_headers":
        return (f"column {change.get('position', '?')}: "
                f"{_shown_value(change, 'before')} → {_shown_value(change, 'after')}")
    if operation == "drop_column":
        return (f"column {change.get('position', '?')} "
                f"(header {_shown_value(change, 'before')}) — delete entire column")
    if operation == "drop_rows":
        return f"CSV row {change.get('row', '?')} — delete entire row"
    if operation == "merge_rows":
        return (f"CSV rows {change.get('row', '?')} + {change.get('source_row', '?')} "
                "— sum counts; retain first row")
    if operation == "flag_cells":
        return (f"CSV row {change.get('row', '?')}, column {change.get('column', '?')} — "
                + ("bold red and yellow fill" if change.get('fill', True) else "bold red font only")
                + "; value unchanged")
    return "Unsupported operation"


def exact_change_rows(record: dict) -> list[dict]:
    """All stored original coordinates and values for a reviewable UI table.

    No values are inferred from the currently displayed finding. A memory
    entry may refer to another dataset, so missing fields stay visibly blank.
    """
    result = []
    for action_index, operation in enumerate((record.get("operations") or [])[:16], 1):
        if not isinstance(operation, dict) or operation.get("operation") not in _ALLOWED_OPERATIONS:
            continue
        op = operation["operation"]
        for change in operation.get("changes") or []:
            if not isinstance(change, dict):
                continue
            result.append({
                "Action": action_index,
                "Type": op,
                "CSV row": change.get("row"),
                "CSV column": change.get("column", change.get("position")),
                "Source row": change.get("source_row"),
                "Before": change.get("before") if "before" in change else "(not saved)",
                "After": (change.get("before") if op == "flag_cells" else
                          change.get("after") if "after" in change else "(not saved)"),
                **({"Style": "Bold red + yellow fill" if change.get("fill", True) else "Bold red font only"} if op == "flag_cells" else {}),
            })
    return result


def short_description(record: dict) -> str:
    """Indexed actions with exact stored before/after values for small edits.

    Large batches show examples, while exact_change_rows exposes every change
    for a UI detail table. This only changes presentation, not stored memory.
    """
    if record.get("decision") == "reviewed_no_change":
        return "No changes required"

    descriptions = []
    for operation in (record.get("operations") or [])[:16]:
        if not isinstance(operation, dict):
            continue
        name = str(operation.get("operation") or "")
        changes = operation.get("changes") or []
        if name not in _ALLOWED_OPERATIONS or not isinstance(changes, list):
            continue
        changes = [change for change in changes if isinstance(change, dict)]
        count = len(changes)
        label = {
            "replace_cells": f"Replace {count} cell{'s' if count != 1 else ''}",
            "rename_headers": f"Rename {count} header{'s' if count != 1 else ''}",
            "drop_column": f"Delete {count} column{'s' if count != 1 else ''}",
            "drop_rows": f"Delete {count} row{'s' if count != 1 else ''}",
            "merge_rows": f"Merge {count} row group{'s' if count != 1 else ''}",
            "flag_cells": f"Highlight {count} cell{'s' if count != 1 else ''} (values unchanged)",
        }[name]
        # Keep the banner brief for large operations, without hiding the exact
        # before/after values of a small (e.g., two-cell) change.
        preview_limit = 6 if count <= 6 else 3
        details = "; ".join(_change_description(name, c)
                            for c in changes[:preview_limit])
        remainder = count - preview_limit
        suffix = (f"; +{remainder} more (expand all exact changes)"
                  if remainder > 0 else "")
        descriptions.append(f"{len(descriptions) + 1}. {label} — {details}{suffix}" if details
                            else f"{len(descriptions) + 1}. {label} — no stored details")
    return "\n\n".join(descriptions) or "Previously reviewed cleaning solution"
