"""Embed a self-contained, source-reconciled reviewer audit in the exported XLSX.

Only approved events become audit operations. Conversation messages are retained
for human review, but are NOT treated as authorization. The workbook is
reopened and cross-checked before a download is made available.
"""
from __future__ import annotations

import hashlib
import io
from collections import defaultdict
from datetime import datetime, timezone

from .in_memory_cleaning import _records
from .styled_export import _exported_positions
from .review_reconciliation import (independently_reconcile,
                                    DETAIL_HEADERS, OVERVIEW_HEADERS)

AUDIT_HEADERS = ["Finding", "Decision", "Reviewer", "Location (source)",
                 "Before", "After (approved)", "Final outcome", "Final location"]
DISCUSSION_HEADERS = ["Finding / check", "Speaker", "Exact message", "Reviewer", "Turn #"]
VERIFY_HEADERS = ["Check", "Value"]


def _safe(value: object) -> str:
    """Keep potentially formula-like user-supplied audit text literal."""
    text = str(value if value is not None else "")[:32000]
    return "'" + text if text.startswith(("=", "+", "-", "@")) else text


def audit_tables(*, original_csv: bytes, final_csv: bytes, history: list[dict],
                 review_statuses: dict, findings: dict[str, str],
                 approved_rows: list[dict], approved_columns: list[dict],
                 discussions: dict[str, list[dict]] | None = None,
                 reviewer: str = "", verified: dict | None = None) -> dict:
    """Bind every ledger action to original and final physical CSV coordinates."""
    original, final = _records(original_csv), _records(final_csv)
    row_map, col_map = _exported_positions(original, final, approved_rows, approved_columns)
    data = []
    named = str(reviewer).strip()
    if not named:
        raise PermissionError("Named reviewer required for the exported audit")
    row_by_finding = defaultdict(list)
    col_by_finding = defaultdict(list)
    for action in approved_rows:
        row_by_finding[(str(action.get("finding_id")), action.get("operation"))].append(action)
    for action in approved_columns:
        col_by_finding[str(action.get("finding_id"))].append(action)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    audit_history_findings = set()

    def add(fid: str, kind: str, who: str, location: str, before: str,
            approved: str, outcome: str, final_location: str = ""):
        if not who.strip():
            raise PermissionError("An approval is missing its reviewer")
        data.append([findings.get(fid, fid), kind, who, location,
                     before, approved, outcome, final_location])

    for event in history:
        if not isinstance(event, dict) or event.get("dataset_id") != "dataset_1":
            continue
        fid = str(event.get("finding_id") or "")
        kind = str(event.get("operation") or "")
        who = str(event.get("reviewer") or "")
        audit_history_findings.add(fid)
        if kind in {"replace_cells", "rename_headers"}:
            for delta in event.get("approved_cell_changes") or []:
                r, c = delta.get("row"), delta.get("column")
                if type(r) is not int or type(c) is not int or not (1 <= r <= len(original) and 1 <= c <= len(original[0])):
                    raise PermissionError("Audit has an invalid approved cell coordinate")
                surviving = r in row_map and c in col_map
                actual = final[row_map[r]-1][col_map[c]-1] if surviving else ""
                after = str(delta.get("after") or "")
                if surviving and actual != after:
                    # Later approved edits can replace an older approved value;
                    # the audit must describe the final outcome honestly.
                    outcome = "Superseded by later approved edit"
                else:
                    outcome = "Present in final dataset" if surviving else "Superseded by approved deletion"
                add(fid, kind, who, f"R{r}:C{c}", str(delta.get("before") or ""),
                    after, outcome, f"R{row_map[r]}:C{col_map[c]}" if surviving else "")
        elif kind == "drop_column":
            # Approval events may group multiple positions. Reconcile against
            # the independent named column ledger, not a UI summary string.
            positions = [e.get("column") for e in event.get("modified_positions") or []]
            matches = [a for a in col_by_finding[fid] if a.get("position") in positions]
            for approval in matches:
                c = approval["position"]
                if c in used_cols:
                    continue
                used_cols.add(c)
                add(fid, kind, who, f"Column {c}", original[0][c-1], "Delete column",
                    "Deleted from final dataset" if c not in col_map else "Still present")
        elif kind in {"drop_rows", "merge_rows"}:
            positions = event.get("modified_positions") or []
            match_ids = {e.get("row") for e in positions if isinstance(e, dict)}
            matched = [a for a in row_by_finding[(fid, kind)] if
                       any(ch.get("row") in match_ids for ch in a.get("changes") or [])]
            for approval in matched:
                for delta in approval.get("changes") or []:
                    r = delta["row"]
                    target = delta.get("source_row", r) if kind == "merge_rows" else r
                    if target in used_rows:
                        continue
                    used_rows.add(target)
                    add(fid, kind, who, f"Row {target}", original[target-1][0],
                        "Merge counts into retained row" if kind == "merge_rows" else "Delete row",
                        "Deleted from final dataset" if target not in row_map else "Still present")
        elif kind == "flag_cells":
            for flag in event.get("flags") or []:
                r, c = flag["row"], flag["column"]
                survives = r in row_map and c in col_map
                add(fid, "flag_cells", who, f"R{r}:C{c}", str(flag.get("before") or ""),
                    "Bold red + yellow highlight (value unchanged)",
                    "Flagged in final Excel" if survives else "Superseded by approved deletion",
                    f"R{row_map[r]}:C{col_map[c]}" if survives else "")
        else:
            raise PermissionError("Unknown approval operation in audit history")

    # Review-only completions must be auditable without claiming another edit.
    for fid, status in sorted(review_statuses.items()):
        if status == "Reviewed — no change":
            add(str(fid), "no_change", named, "Original data", "Unchanged", "Unchanged",
                "Reviewed — no change")
        elif status == "Resolved by earlier approved edit":
            add(str(fid), "review_resolved_by_prior_approvals", named,
                "Previously approved source deletions", "Earlier deletion(s)",
                "No new edits", "Reviewed — resolved by earlier approved deletions")
    discussions_out = []
    for fid, thread in (discussions or {}).items():
        for turn, entry in enumerate(thread, 1):
            if (not isinstance(entry, dict) or entry.get("role") not in {"user", "assistant"}
                    or entry.get("automatic")):
                continue
            message = str(entry.get("content") or "")
            if message.strip():
                discussions_out.append([findings.get(str(fid), str(fid)),
                                        "Reviewer" if entry["role"] == "user" else "Agent C",
                                        message, named if entry["role"] == "user" else "", turn])
    reconciliation = independently_reconcile(
        original_csv=original_csv, final_csv=final_csv,
        history=history, review_statuses=review_statuses,
        findings=findings, approved_rows=approved_rows,
        approved_columns=approved_columns, discussions=discussions)
    # The overview is deliberately one row per finding/check, instead of a
    # sprawling flat list of technical ledger operations.
    # Group all actions for a finding/check together; keep approval order
    # within each finding. No original action or conversation text is dropped.
    by_title = {title: index for index, title in enumerate(findings.values())}
    data.sort(key=lambda row: by_title.get(row[0], len(by_title)))
    reconciliation['details'].sort(key=lambda row: by_title.get(row[0], len(by_title)))
    verification = [
        ["Reviewer", named],
        ["Exported at (UTC)", datetime.now(timezone.utc).isoformat()],
        ["Source SHA256", hashlib.sha256(original_csv).hexdigest()],
        ["Final CSV SHA256", hashlib.sha256(final_csv).hexdigest()],
        ["Final data rows", str(len(final)-1)],
        ["Final columns", str(len(final[0]))],
        ["Approved ledger events", str(len(history))],
        ["Reviewed no-change decisions", str(sum(v == "Reviewed — no change" for v in review_statuses.values()))],
        ["Reviews resolved by earlier approvals", str(sum(v == "Resolved by earlier approved edit" for v in review_statuses.values()))],
        ["Audit operation records", str(len(data))],
        ["Discussion records", str(len(discussions_out))],
        ["Final CSV independently verified", "Yes" if verified is not None else "No"],
        ["Original data rows", str(reconciliation['original_rows'])],
        ["Original columns", str(reconciliation['original_columns'])],
        ["Surviving cells unchanged", str(reconciliation['surviving_unchanged_cells'])],
        ["Surviving cells changed by approved actions", str(reconciliation['surviving_changed_cells'])],
        ["Rows removed by approval", str(reconciliation['deleted_rows'])],
        ["Columns removed by approval", str(reconciliation['deleted_columns'])],
        ["Action-by-action verification", "See Source-to-final checks: one row per final change or deletion"],
        ["Reviewer decisions", "See Review overview and Audit trail for each finding/check"],
        ["Conversation evidence", "See Interaction log; reviewer messages are recorded verbatim"],
        ["Note", "All original coordinates are physical CSV coordinates including header row 1."],
        ["Scientific interpretation", "The model is advisory; this verifies approved file operations, not biological taxon correctness."],
    ]
    # Keep the verification summary at the top for quick sign-off, then include
    # the complete source-to-final proof *on the Verification sheet itself*.
    # Reviewers need not switch tabs to inspect every original/approved/final
    # value, superseded action or explicitly retained unchanged finding.
    verification.extend([
        ["", ""],
        ["SOURCE-TO-FINAL DETAILS — Every approved difference and deletion", ""],
        DETAIL_HEADERS,
        *reconciliation['details'],
        ["", ""],
        ["REVIEW DECISIONS — Includes approved no-change findings", ""],
        OVERVIEW_HEADERS,
        *reconciliation['summary'],
    ])
    return {"Review overview": [OVERVIEW_HEADERS, *reconciliation['summary']],
            "Audit trail": [AUDIT_HEADERS, *data],
            "Source-to-final checks": [DETAIL_HEADERS, *reconciliation['details']],
            "Interaction log": [DISCUSSION_HEADERS, *discussions_out],
            "Verification": [VERIFY_HEADERS, *verification]}


def embed_audit_sheets(xlsx: bytes, tables: dict) -> bytes:
    """Attach review records as normal visible Excel tabs to the same file."""
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    workbook = load_workbook(io.BytesIO(xlsx))
    if workbook.sheetnames != ["Cleaned dataset"]:
        raise PermissionError("Unexpected workbook structure before audit attachment")
    order = ("Review overview", "Audit trail", "Source-to-final checks", "Interaction log", "Verification")
    for name in order:
        sheet = workbook.create_sheet(name)
        for row in tables[name]:
            sheet.append([_safe(item) for item in row])
        sheet.freeze_panes = "A2"
        if name != "Verification":
            sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="17324D")
        widths = ([37, 28, 24, 19, 19, 15, 40] if name == "Review overview" else
                  [32, 21, 18, 17, 46, 46, 31, 19] if name == "Audit trail" else
                  [34, 18, 20, 20, 25, 49, 49, 40, 19, 29, 35, 21]
                  if name == "Source-to-final checks" else
                  [35, 15, 82, 17, 12] if name == "Interaction log" else
                  [36, 21, 20, 23, 45, 48, 40, 34, 22, 31, 32, 22])
        for idx, width in enumerate(widths, 1):
            sheet.column_dimensions[get_column_letter(idx)].width = width
        sheet.sheet_view.showGridLines = False
        if name in {"Review overview", "Source-to-final checks", "Audit trail"}:
            # Outline by finding, without altering the schema used by export
            # integrity tests. Distinct light banding improves scanability.
            previous = None
            band = False
            for row_i in range(2, sheet.max_row + 1):
                finding_name = sheet.cell(row_i, 1).value
                if finding_name != previous:
                    band = not band
                    previous = finding_name
                if band:
                    for cell in sheet[row_i]:
                        cell.fill = PatternFill('solid', fgColor='EAF2F8')
                for cell in sheet[row_i]:
                    cell.alignment = Alignment(wrap_text=True, vertical='top')
        if name == "Verification":
            # Mixed summary, exhaustive audit-detail, and review-decision
            # tables deliberately share this single printable worksheet.
            for row in sheet.iter_rows(min_row=2):
                title = str(row[0].value or "")
                if title.startswith(("SOURCE-TO-FINAL DETAILS", "REVIEW DECISIONS")):
                    for cell in row:
                        cell.font = Font(bold=True, color="FFFFFF")
                        cell.fill = PatternFill("solid", fgColor="17324D")
                elif title in {DETAIL_HEADERS[0], OVERVIEW_HEADERS[0]}:
                    for cell in row:
                        cell.font = Font(bold=True, color="FFFFFF")
                        cell.fill = PatternFill("solid", fgColor="2676B8")
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
            sheet.sheet_properties.tabColor = "159947"
        if name == "Review overview":
            sheet.sheet_properties.tabColor = "159947"
        elif name == "Source-to-final checks":
            sheet.sheet_properties.tabColor = "2676B8"
        if name != "Verification":
            for row in sheet.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
    saved = io.BytesIO()
    workbook.save(saved)
    workbook.close()
    return saved.getvalue()


def verify_audit_sheets(xlsx: bytes, tables: dict) -> None:
    """Fail export if the attached human-readable audit loses or changes an entry."""
    from openpyxl import load_workbook
    workbook = load_workbook(io.BytesIO(xlsx), read_only=True, data_only=False)
    try:
        if workbook.sheetnames != ["Cleaned dataset", "Review overview", "Audit trail",
                                   "Source-to-final checks", "Interaction log", "Verification"]:
            raise PermissionError("Export is missing its attached review audit")
        for name, rows in tables.items():
            actual = [tuple('' if value is None else value for value in row)
                      for row in workbook[name].values]
            # A mixed-summary worksheet has variable-width sections. Excel
            # pads shorter rows with empty cells; normalize both representations.
            width = max((len(row) for row in rows), default=0)
            expected = [tuple(_safe(value) for value in row) + ('',) * (width - len(row))
                        for row in rows]
            if actual != expected:
                raise PermissionError(f"Audit sheet {name} does not match verified reviewer history")
    finally:
        workbook.close()
