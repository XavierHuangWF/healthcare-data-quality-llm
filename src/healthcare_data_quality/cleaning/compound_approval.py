"""Atomic, human-reviewed compound cleaning proposals.

Compute every proposed change against immutable source coordinates. R-dependent
edits must have passed the existing isolated R runner. This module never writes
source files and never mutates Streamlit state. The caller commits its returned
state only after ALL operations validate and the user approves the group.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .finding_cleaning import validate_scoped_plan
from .in_memory_cleaning import (
    _records, assess_verified_in_memory, apply_verified_in_memory,
    apply_approved_column_deletions, apply_approved_row_operations, sha256_bytes,
)
from healthcare_data_quality.r_runtime.scoped_r_execution import (
    prepare_and_execute_scoped_r_preview, verify_prepared_r_insertion,
)

OPS = {"replace_cells", "rename_headers", "drop_column", "drop_rows", "merge_rows"}


def proposal_digest(plans: list[dict], original: bytes, staged: bytes,
                    existing_columns: list[dict], existing_rows: list[dict]) -> str:
    payload = {
        "plans": plans,
        "source_sha256": sha256_bytes(original),
        "staged_sha256": sha256_bytes(staged),
        "columns": existing_columns,
        "rows": existing_rows,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     default=str).encode("utf-8")).hexdigest()


def describe_operation(plan: dict) -> str:
    op, changes = plan.get("operation"), plan.get("changes") or []
    if op == "drop_column":
        return ", ".join(f"Delete column {c['position']} ({c['before'] or 'unnamed'})" for c in changes)
    if op == "drop_rows":
        numbers = [str(c["row"]) for c in changes]
        preview = ", ".join(numbers[:12])
        if len(numbers) > 12:
            preview += f", … (+{len(numbers) - 12} more)"
        return f"Delete {len(numbers)} CSV row{'s' if len(numbers) != 1 else ''}: {preview}"
    if op == "merge_rows":
        return ", ".join(f"Merge CSV rows {c['row']} + {c['source_row']} (sum counts)"
                         for c in changes)
    if op == "replace_cells":
        if len(changes) == 1:
            c = changes[0]
            return f"Row {c['row']}, column {c['column']}: {c['before'] or 'blank'} → {c['after'] or 'blank'}"
        return f"Replace {len(changes)} cells"
    if op == "rename_headers":
        if len(changes) == 1:
            c = changes[0]
            return f"Rename column {c['position']}: {c['before']} → {c['after']}"
        return f"Rename {len(changes)} headers"
    raise ValueError("Unsupported operation")



def validate_compound_targets(plans: list[dict]) -> None:
    """Permit repeated operation types, but never overlapping source targets.

    Coordinates remain 1-based and refer to the immutable original CSV.
    This is checked both when Agent C drafts plans and when approving them,
    independently of the model and without modifying any dataset.
    """
    if not isinstance(plans, list):
        raise ValueError("Compound plans must be a list")
    seen_cells: set[tuple[int, int]] = set()
    deleted_columns: set[int] = set()
    deleted_rows: set[int] = set()
    row_operations: set[int] = set()

    for plan in plans:
        if not isinstance(plan, dict) or plan.get("operation") not in OPS:
            raise ValueError("Unsupported compound operation")
        op = plan["operation"]
        changes = plan.get("changes")
        if (not isinstance(changes, list) or not changes
                or not all(isinstance(c, dict) for c in changes)):
            raise ValueError("A compound operation requires structured changes")
        if op in {"replace_cells", "rename_headers"}:
            for change in changes:
                target = ((1, change["position"]) if op == "rename_headers"
                          else (change["row"], change["column"]))
                if target in seen_cells:
                    raise ValueError(
                        f"CSV row {target[0]}, column {target[1]} is edited more "
                        "than once in this request; specify one final value")
                seen_cells.add(target)
        elif op == "drop_column":
            for change in changes:
                position = change["position"]
                if position in deleted_columns:
                    raise ValueError(f"Column {position} is scheduled for deletion twice")
                deleted_columns.add(position)
        elif op in {"drop_rows", "merge_rows"}:
            for change in changes:
                rows = ([change["row"]] if op == "drop_rows" else
                        [change["row"], change["source_row"]])
                for row in rows:
                    if row in row_operations:
                        raise ValueError(
                            f"CSV row {row} is targeted by multiple row operations")
                    row_operations.add(row)
                    if op == "drop_rows":
                        deleted_rows.add(row)

    for row, col in seen_cells:
        if col in deleted_columns:
            raise ValueError(
                f"Column {col} is scheduled for deletion and editing; "
                "choose one action for that column")
        if row in deleted_rows:
            raise ValueError(
                f"CSV row {row} is scheduled for deletion and editing; "
                "choose one action for that row")

def evaluate_compound(
    plans: list[dict], finding: dict, source: Path, staged: bytes,
    existing_columns: list[dict], existing_rows: list[dict],
    *, reviewer: str = "pending_reviewer",
    requested_rows: frozenset[int] | None = None,
    requested_columns: frozenset[int] | None = None,
    r_previews: dict[int, dict] | None = None,
    allow_single: bool = False,
) -> dict:
    """Verify all actions; return a fully validated, still-uncommitted snapshot.

    `r_previews` is a previously tested R output cache. Even when supplied,
    every R output and plan is reverified against the immutable source.
    """
    if (not isinstance(plans, list) or
            not (1 if allow_single else 2) <= len(plans) <= 16):
        raise ValueError("A review requires 1–16 operations if single approval is enabled, otherwise 2–16")
    validate_compound_targets(plans)
    if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 120:
        raise ValueError("Reviewer identifier is required")
    original = Path(source).read_bytes()
    original_records, staged_records = _records(original), _records(staged)
    if len(original_records) != len(staged_records) or any(
        len(a) != len(b) for a, b in zip(original_records, staged_records)
    ):
        raise PermissionError("The staged CSV dimensions changed")

    pending_columns = [dict(entry) for entry in existing_columns]
    pending_rows = [dict(entry) for entry in existing_rows]
    verified_r = {}
    events = []
    current = staged
    for idx, plan in enumerate(plans):
        if not isinstance(plan, dict) or plan.get("operation") not in OPS:
            raise ValueError("Unsupported compound operation")
        checked = validate_scoped_plan(
            plan, finding, source,
            user_requested_rows=requested_rows,
            user_requested_columns=requested_columns,
        )
        if checked != plan:
            raise PermissionError("Plan changed after source validation")
        op = checked["operation"]
        event = {
            "operation": op,
            "changes": len(checked["changes"]),
            "modified_positions": [],
            "already_resolved": 0,
        }
        if op in {"replace_cells", "rename_headers"}:
            if r_previews is not None:
                prepared = r_previews.get(idx)
                if not isinstance(prepared, dict):
                    raise PermissionError("Missing verified R preview; review the proposal again")
            else:
                prepared = prepare_and_execute_scoped_r_preview(checked, finding, source)
            insertion = verify_prepared_r_insertion(prepared, checked, finding, source)
            assessment = assess_verified_in_memory(original, current, insertion)
            if assessment["conflicts"]:
                first = assessment["conflicts"][0]
                raise PermissionError(
                    f"Conflicting earlier edit at row {first['csv_row']}, "
                    f"column {first['column']}; revise this proposal before approval")
            current, applied = apply_verified_in_memory(
                original, current, insertion, reviewer=reviewer)
            verified_r[idx] = prepared
            event["changes"] = applied["changes_applied"]
            event["already_resolved"] = applied["already_resolved"]
            event["approved_cell_changes"] = [
                {"row": c["csv_row"], "column": c["column"],
                 "before": c["current"], "after": c["proposed"]}
                for c in assessment["ready_to_apply"]
            ]
            event["modified_positions"] = [
                {"csv_row": c["csv_row"], "column": c["column"]}
                for c in assessment["ready_to_apply"]
            ]
            event["r_code_sha256"] = hashlib.sha256(
                prepared["insertion_r_function"].encode("utf-8")).hexdigest()
        elif op == "drop_column":
            item = checked["changes"][0]
            if any(p.get("position") == item["position"] for p in pending_columns):
                raise PermissionError("Column already scheduled for deletion")
            pending_columns.append({
                "position": item["position"], "before": item["before"],
                "reviewer": reviewer, "source_sha256": checked["source_sha256"],
                "finding_id": finding["id"],
            })
            event["modified_positions"] = [{"column": item["position"]}]
        elif op in {"drop_rows", "merge_rows"}:
            pending_rows.append({
                "operation": op, "changes": checked["changes"],
                "reviewer": reviewer, "source_sha256": checked["source_sha256"],
                "finding_id": finding["id"],
            })
            event["modified_positions"] = [
                {k: v for k, v in c.items() if k in {"row", "source_row"}}
                for c in checked["changes"]
            ]
        events.append(event)

    # Cross-operation verification: row merges use the updated count values;
    # column removals happen last, preserving original column coordinates.
    exported_rows = apply_approved_row_operations(original, current, pending_rows)
    exported = apply_approved_column_deletions(
        original, exported_rows, pending_columns,
        approved_row_operations=pending_rows or None,
        staged_before_rows=current if pending_rows else None,
    )
    exported_records = _records(exported)
    return {
        "digest": proposal_digest(plans, original, staged, existing_columns, existing_rows),
        "original_sha256": sha256_bytes(original),
        "staged_sha256": sha256_bytes(staged),
        "staged_after": current,
        "columns_after": pending_columns,
        "rows_after": pending_rows,
        "export_sha256": sha256_bytes(exported),
        "export_rows": len(exported_records) - 1,
        "export_columns": len(exported_records[0]),
        "events": events,
        "r_previews": verified_r,
    }
