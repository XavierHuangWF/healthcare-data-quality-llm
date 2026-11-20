"""Read-only, source-backed UI summaries for cleaning review.

Never grants edit scope or authorizes changes. The validators in the cleaning
workflow remain responsible for target verification and atomic approval.
"""
from __future__ import annotations

import csv
from io import StringIO


def completed_finding_label(finding_id: str, history: list[dict], statuses: dict) -> str:
    """Return a visible marker for *actual* approvals, not remembered advice."""
    approvals = [event for event in history
                 if isinstance(event, dict) and event.get("finding_id") == finding_id]
    if approvals:
        n = len(approvals)
        return f"Staged ✓ · {n} approved action{'s' if n != 1 else ''}"
    if statuses.get(finding_id) == "Reviewed — no change":
        return "Reviewed ✓ · no changes"
    return ""


def _csv_rows(content: bytes) -> list[list[str]]:
    if not isinstance(content, bytes):
        raise TypeError("The source preview requires original CSV bytes")
    rows = list(csv.reader(StringIO(content.decode("utf-8-sig"), newline="")))
    if not rows or not rows[0]:
        raise ValueError("Original CSV has no header")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Original CSV contains inconsistent row widths")
    return rows


def deletion_preview(source: bytes, plans: list[dict]) -> dict:
    """Enumerate complete, real source rows/columns targeted by reviewed plans.

    Row indices are 1-based CSV coordinates, including the header as row 1.
    Returns source metadata for DISPLAY ONLY; it never rewrites a plan.
    """
    data = _csv_rows(source)
    headers = data[0]
    rows_to_delete, columns_to_delete = set(), set()
    merges = []
    for plan in plans:
        if not isinstance(plan, dict):
            raise ValueError("Invalid review plan")
        operation = plan.get("operation")
        changes = plan.get("changes") or []
        if not isinstance(changes, list):
            raise ValueError("Invalid review changes")
        if operation == "drop_rows":
            for item in changes:
                rows_to_delete.add(item["row"])
        elif operation == "merge_rows":
            for item in changes:
                merges.append((item["row"], item["source_row"]))
                rows_to_delete.add(item["source_row"])
        elif operation == "drop_column":
            for item in changes:
                columns_to_delete.add(item["position"])
    for row in rows_to_delete:
        if type(row) is not int or row <= 1 or row > len(data):
            raise ValueError("A deletion references an invalid original CSV row")
    for col in columns_to_delete:
        if type(col) is not int or col <= 0 or col > len(headers):
            raise ValueError("A deletion references an invalid original CSV column")
    # Taxonomy/feature lives in the first column. All count-column values
    # remain intact for this preview, including blank strings and literal NA.
    row_details = []
    full_rows = []
    column_labels = [f"C{i} · {h or '(blank header)'}"
                     for i, h in enumerate(headers, start=1)]
    for row_id in sorted(rows_to_delete):
        record = data[row_id - 1]
        empty_count = sum(not str(value).strip() for value in record[1:])
        populated = sum(bool(str(value).strip()) for value in record[1:])
        samples = [f"C{index} {headers[index - 1]}={value}"
                   for index, value in enumerate(record[1:], start=2)
                   if str(value).strip()][:3]
        row_details.append({
            "CSV row": row_id,
            "Taxon / feature": record[0],
            "Empty count cells": empty_count,
            "Populated count cells": populated,
            "Sample values": "; ".join(samples) or "(none)",
        })
        full_rows.append({"CSV row": row_id, **dict(zip(column_labels, record))})
    column_details = []
    for col in sorted(columns_to_delete):
        values = [r[col - 1] for r in data[1:]]
        nonempty = [v for v in values if str(v).strip()]
        column_details.append({
            "CSV column": col,
            "Header": headers[col - 1] or "(blank)",
            "Populated rows": len(nonempty),
            "Empty rows": len(values) - len(nonempty),
            "Sample values": ", ".join(nonempty[:3]) or "(none)",
        })
    return {"rows": row_details, "full_rows": full_rows,
            "columns": column_details, "merges": merges}
