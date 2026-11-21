"""Safely undo one finding's staged approvals before a researcher revises it.

This operates on immutable original CSV coordinates, without modifying disk
files. The caller commits the fully verified new snapshot as one transaction.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

from .in_memory_cleaning import (
    _records, apply_approved_column_deletions, apply_approved_row_operations,
    sha256_bytes,
)


def reverse_approved_finding(source: Path, staged: bytes, *,
                             finding_id: str, dataset_id: str,
                             history: list[dict],
                             columns: list[dict], rows: list[dict]) -> dict:
    """Return a validated revision snapshot; never mutate the caller's state.

    Revert only cell/header values actually modified under this finding. Verify
    every last staged value before restoring it and revalidate other approved
    deletions/merges on the resulting dataset. Refuse legacy edits that lack
    enough provenance for a safe inverse.
    """
    original = Path(source).read_bytes()
    original_rows, stage_rows = _records(original), _records(staged)
    if len(original_rows) != len(stage_rows) or any(
            len(a) != len(b) for a, b in zip(original_rows, stage_rows)):
        raise PermissionError("Current staged CSV differs in shape from the original")
    events = [event for event in history if event.get("finding_id") == finding_id
              and event.get("dataset_id") == dataset_id]
    if not events:
        raise ValueError("There are no approved edits to reverse")
    source_hash = sha256_bytes(original)
    for event in events:
        if event.get("source_sha256") != source_hash:
            raise PermissionError("Approval is tied to a different original CSV")
    # A different, more recent approval can depend on a value this finding
    # changed. Refuse to silently undo that person's later decision.
    modified_other_later = set()
    candidate = [list(row) for row in stage_rows]
    selected_indexes = [i for i, e in enumerate(history)
                        if e.get("finding_id") == finding_id
                        and e.get("dataset_id") == dataset_id]
    for idx in reversed(selected_indexes):
        event = history[idx]
        operation = event.get("operation")
        if operation not in {"replace_cells", "rename_headers"}:
            continue
        changes = event.get("approved_cell_changes")
        if not isinstance(changes, list):
            if int(event.get("changes") or 0) == 0:
                continue
            raise PermissionError(
                "This older approval lacks exact before/after values. "
                "Its edits cannot be safely reversed automatically")
        for change in reversed(changes):
            row = change.get("row")
            col = change.get("column")
            if (type(row) is not int or type(col) is not int
                    or not 1 <= row <= len(candidate)
                    or not 1 <= col <= len(candidate[0])
                    or not isinstance(change.get("before"), str)
                    or not isinstance(change.get("after"), str)):
                raise PermissionError("Incomplete original-coordinate audit for this approval")
            target = row, col
            for future in history[idx + 1:]:
                if (future.get("dataset_id") != dataset_id
                        or future.get("finding_id") == finding_id):
                    continue
                if any(item.get("row") == row and item.get("column") == col
                       for item in (future.get("approved_cell_changes") or [])):
                    modified_other_later.add(target)
            if target in modified_other_later:
                raise PermissionError(
                    f"Another approved finding subsequently edited row {row}, column {col}. "
                    "Review that dependent edit before reversing this one")
            if candidate[row - 1][col - 1] != change["after"]:
                raise PermissionError(
                    f"Row {row}, column {col} has changed since approval. "
                    "The previous value cannot be safely restored")
            candidate[row - 1][col - 1] = change["before"]
    # Stage remains coordinate-stable: physical row/column removals are applied
    # only to export, using separate approved-operation lists.
    next_columns = [dict(item) for item in columns
                    if item.get("finding_id") != finding_id]
    next_rows = [dict(item) for item in rows
                 if item.get("finding_id") != finding_id]
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(candidate)
    next_staged = (b"\xef\xbb\xbf" if staged.startswith(b"\xef\xbb\xbf") else b"") + buffer.getvalue().encode("utf-8")
    if _records(next_staged) != candidate:
        raise AssertionError("Reverted CSV does not reproduce the approved inverse")
    verified_rows = apply_approved_row_operations(original, next_staged, next_rows)
    final = apply_approved_column_deletions(
        original, verified_rows, next_columns,
        approved_row_operations=next_rows or None,
        staged_before_rows=next_staged if next_rows else None,
    )
    if not _records(final):
        raise AssertionError("Reverted export is empty")
    next_history = [dict(event) for event in history if not (
        event.get("finding_id") == finding_id
        and event.get("dataset_id") == dataset_id)]
    # Preserve the original signed stage hash; append a new, verified
    # reconciliation hash rather than rewriting historical approval evidence.
    for event in reversed(next_history):
        if event.get("dataset_id") == dataset_id:
            event["revision_reconciled_sha256"] = sha256_bytes(next_staged)
            break
    return {
        "staged": next_staged, "columns": next_columns, "rows": next_rows,
        "history": next_history,
        "reversed_actions": len(events), "export_sha256": sha256_bytes(final),
    }
