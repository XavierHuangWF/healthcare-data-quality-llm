"""Render *approved* visual annotations in a cleaned Excel workbook.

The source CSV and export-stage CSV are immutable. A flag records coordinates in
that original source; rows/columns removed at export must be remapped before a
style can be applied. Workbook cells never contain HTML or font markup.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from pathlib import Path

_NUMERIC = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\Z")


def approved_flags_for_export(
    approval_history: list[dict], dataset_id: str, original_csv: bytes,
) -> list[dict]:
    """Derive export styles only from current named, source-bound approvals.

    A temporary session-state styling list is not an approval ledger, and may
    be missing after restoration or may contain flags from a reopened review.
    The current history is authoritative; revisions remove their old events.
    """
    source_hash = hashlib.sha256(original_csv).hexdigest()
    latest: dict[tuple[int, int], dict] = {}
    for event in approval_history:
        if (not isinstance(event, dict) or event.get("operation") != "flag_cells"
                or event.get("dataset_id") != dataset_id):
            continue
        if (event.get("source_sha256") != source_hash or
                not str(event.get("reviewer") or "").strip() or
                not isinstance(event.get("flags"), list)):
            raise PermissionError("Approved visual annotation does not match this source or reviewer")
        for flag in event["flags"]:
            if not isinstance(flag, dict):
                raise PermissionError("Malformed approved visual annotation")
            r, c = flag.get("row"), flag.get("column")
            if type(r) is not int or type(c) is not int:
                raise PermissionError("Invalid approved visual annotation coordinates")
            latest[(r, c)] = dict(flag)
    return [latest[loc] for loc in sorted(latest)]


def _read_csv(data: bytes) -> list[list[str]]:
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True))
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Cleaned or original CSV is not rectangular")
    return rows


def _exported_positions(
    source_rows: list[list[str]], final_rows: list[list[str]],
    row_approvals: list[dict], column_approvals: list[dict],
) -> tuple[dict[int, int], dict[int, int]]:
    """Map 1-based physical original CSV coordinates to their final positions."""
    removed_rows: set[int] = set()
    for approval in row_approvals:
        operation = approval.get("operation")
        if operation not in {"drop_rows", "merge_rows"}:
            raise ValueError("Unknown row approval type")
        for change in approval.get("changes", []):
            position = change.get("row") if operation == "drop_rows" else change.get("source_row")
            if type(position) is not int or not 2 <= position <= len(source_rows):
                raise ValueError("Invalid original CSV row in export approval")
            if position in removed_rows:
                raise ValueError("Duplicate deleted row")
            removed_rows.add(position)
    removed_cols: set[int] = set()
    for approval in column_approvals:
        position = approval.get("position")
        if type(position) is not int or not 1 <= position <= len(source_rows[0]):
            raise ValueError("Invalid original CSV column in export approval")
        if position in removed_cols:
            raise ValueError("Duplicate deleted column")
        removed_cols.add(position)
    kept_rows = [index for index in range(1, len(source_rows) + 1) if index not in removed_rows]
    kept_cols = [index for index in range(1, len(source_rows[0]) + 1) if index not in removed_cols]
    if len(kept_rows) != len(final_rows) or len(kept_cols) != len(final_rows[0]):
        raise PermissionError("Approved structural operations do not match cleaned CSV dimensions")
    return ({original: final for final, original in enumerate(kept_rows, 1)},
            {original: final for final, original in enumerate(kept_cols, 1)})


def surviving_flag_count(
    cleaned_csv: bytes, *, original_csv: bytes, approved_flags: list[dict],
    approved_row_operations: list[dict], approved_column_deletions: list[dict],
) -> int:
    """Count original flag targets that actually survive structural cleaning."""
    original = _read_csv(original_csv)
    final = _read_csv(cleaned_csv)
    rows, cols = _exported_positions(
        original, final, approved_row_operations, approved_column_deletions)
    surviving = set()
    for flag in approved_flags:
        r, c = flag.get("row"), flag.get("column")
        if r in rows and c in cols:
            surviving.add((rows[r], cols[c]))
    return len(surviving)


def _excel_cell_value(text: str, column: int):
    """Keep identifiers/taxonomies as text; use numeric types for safe counts."""
    if column == 1 or not text or text.upper() == "NA" or not _NUMERIC.fullmatch(text):
        return text
    # Excel stores at most fifteen significant decimal digits precisely.
    digits = re.sub(r"[^0-9]", "", text.split("e")[0].split("E")[0]).lstrip("0")
    if len(digits) > 15:
        return text
    try:
        number = float(text) if any(x in text for x in (".", "e", "E")) else int(text)
        return number if not isinstance(number, float) or math.isfinite(number) else text
    except (ValueError, OverflowError):
        return text


def build_styled_excel_download(
    cleaned_csv: bytes, *, original_csv: bytes, original_filename: str,
    approved_flags: list[dict], approved_row_operations: list[dict],
    approved_column_deletions: list[dict],
) -> tuple[str, bytes]:
    """Make a one-sheet XLSX using the approved FINAL dataset and exact flags.

    Never infer a cleaning operation from a flag. Flag coordinates and original
    values must still match the original source. A flag on a deleted row/column
    does not move to an unrelated surviving cell.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    original_rows = _read_csv(original_csv)
    final_rows = _read_csv(cleaned_csv)
    row_map, col_map = _exported_positions(
        original_rows, final_rows, approved_row_operations, approved_column_deletions)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Cleaned dataset"
    for row_num, csv_row in enumerate(final_rows, 1):
        for col_num, text in enumerate(csv_row, 1):
            cell = sheet.cell(row_num, col_num, _excel_cell_value(text, col_num))
            # Data originated in a CSV and is never executable content.
            # openpyxl interprets a leading '=' string as an Excel formula by
            # default; force the literal original value instead of running it.
            if isinstance(text, str) and text.startswith("="):
                cell.data_type = "s"
            if row_num == 1:
                cell.font = Font(bold=True)
                cell.fill = PatternFill(fill_type="solid", fgColor="E8EDF4")
    sheet.freeze_panes = "B2"
    sheet.auto_filter.ref = sheet.dimensions
    sheet.column_dimensions["A"].width = 70
    for col_num in range(2, min(len(final_rows[0]), 100) + 1):
        sheet.column_dimensions[get_column_letter(col_num)].width = 14
    flagged_positions = set()
    for flag in approved_flags:
        row, col = flag.get("row"), flag.get("column")
        if (type(row) is not int or type(col) is not int or
                not 2 <= row <= len(original_rows) or
                not 1 <= col <= len(original_rows[0]) or
                flag.get("style") != "bold_red" or
                flag.get("before") != original_rows[row-1][col-1]):
            raise PermissionError("A visual flag no longer matches the immutable original CSV")
        if row not in row_map or col not in col_map:
            continue  # Its explicitly deleted source row/column is absent.
        position = row_map[row], col_map[col]
        if position in flagged_positions:
            continue
        flagged_positions.add(position)
        cell = sheet.cell(*position)
        cell.font = Font(bold=True, color="FFDC2626")
        cell.fill = PatternFill(fill_type="solid", fgColor="FFFFE699")
        cell.alignment = Alignment(vertical="top", wrap_text=True)
    output = io.BytesIO()
    workbook.save(output)
    name = Path(original_filename).name
    if not name.lower().endswith(".csv"):
        raise ValueError("Expected a CSV filename")
    return "cleaned_highlighted_" + name[:-4] + ".xlsx", output.getvalue()
