"""Show whether approved Before→After decisions survive in the actual final CSV.

A later independently approved deletion can supersede an earlier edit, but
that old edit should never be presented as a change still present in the file.
"""
from __future__ import annotations

from pathlib import Path

from .in_memory_cleaning import (
    _records, apply_approved_row_operations, apply_approved_column_deletions,
)
from .styled_export import _exported_positions


def final_review_rows(plans: list[dict], source: Path, staged: bytes | None,
                      approved_rows: list[dict], approved_columns: list[dict]) -> list[dict]:
    """Return exact final-state audit rows, with original source coordinates.

    This is a display only; verify_final_csv independently fails closed on any
    approval ledger or final byte mismatch before the download is offered.
    """
    original = Path(source).read_bytes()
    staging = staged if staged is not None else original
    after_rows = apply_approved_row_operations(original, staging, approved_rows)
    final = apply_approved_column_deletions(
        original, after_rows, approved_columns,
        approved_row_operations=approved_rows, staged_before_rows=staging,
    )
    source_values, final_values = _records(original), _records(final)
    row_map, col_map = _exported_positions(
        source_values, final_values, approved_rows, approved_columns)
    output = []
    for plan in plans:
        kind = plan.get("operation")
        for change in plan.get("changes") or []:
            if kind == "replace_cells":
                row, col = change["row"], change["column"]
                location = f"R{row} · C{col}"
                before = source_values[row - 1][col - 1]
                if row not in row_map or col not in col_map:
                    after = "Deleted by later approved action"
                else:
                    after = final_values[row_map[row] - 1][col_map[col] - 1]
            elif kind in {"rename_headers", "drop_column"}:
                col = change["position"]
                location, before = f"Column {col}", source_values[0][col - 1]
                after = (final_values[0][col_map[col] - 1] if col in col_map
                         else "Deleted column")
            elif kind == "drop_rows":
                row = change["row"]
                location, before = f"Row {row}", source_values[row - 1][0]
                after = ("Deleted row" if row not in row_map
                         else "Still present — verify deletion")
            elif kind == "merge_rows":
                row, child = change["row"], change["source_row"]
                location, before = f"Rows {row} + {child}", source_values[row - 1][0]
                after = ("Merged / source row removed" if child not in row_map
                         else "Source row still present — verify merge")
            else:
                continue
            output.append({"Location": location, "Before": before, "After (final)": after})
    return output


def verify_unchanged_headers(original_rows: list[list[str]], final_rows: list[list[str]],
                             review_statuses: dict, findings: dict[str, dict],
                             column_map: dict[int, int]) -> None:
    """Fail closed if 'keep blank header' conflicts with the ACTUAL export.

    A later approved deletion can supersede the header, but an unapproved or
    still-present changed header is not a truthful no-change review.
    """
    for fid, finding in (findings or {}).items():
        if review_statuses.get(fid) != "Reviewed — no change":
            continue
        if not isinstance(finding, dict) or finding.get("check_key") != "inspect_headers":
            continue
        for evidence in finding.get("evidence") or []:
            if not isinstance(evidence, dict):
                continue
            col = evidence.get("Column")
            if (type(col) is int and col in column_map
                    and 1 <= col <= len(original_rows[0])
                    and original_rows[0][col - 1] == ""):
                current = final_rows[0][column_map[col] - 1]
                if current != "":
                    raise PermissionError(
                        f"Column {col} was reviewed to remain blank, "
                        "but its exported header was changed. Revise that review first.")
