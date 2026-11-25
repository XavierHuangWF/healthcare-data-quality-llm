"""Source-reconciled reviewer ledger for the cleaned CSV and optional styled Excel.

A separate audit of approved decisions is necessary because source rows/columns
may be deleted later. Such flags must be preserved as marked ORIGINAL EVIDENCE,
not transferred onto an unrelated remaining cell or encoded into CSV strings.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from decimal import Decimal, InvalidOperation
from typing import Any

from healthcare_data_quality.cleaning.styled_export import _exported_positions, _read_csv


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def build_reviewer_audit(
    *, original_csv: bytes, cleaned_csv: bytes, original_filename: str,
    history: list[dict], decisions: list[dict], review_statuses: dict[str, str],
    conversations: dict[str, list[dict]], approved_row_operations: list[dict],
    approved_column_deletions: list[dict], reviewer: str = "", dataset_id: str = "dataset_1",
    finding_titles: dict[str, str] | None = None,
    completed_audits: dict[str, dict] | None = None,
) -> dict:
    """Build a complete, validated source-linked review, including superseded edits.

    `history` and `decisions` are the authoritative *current-session* review
    records, not previous model suggestions. Original coordinates are preserved.
    """
    original = _read_csv(original_csv)
    final = _read_csv(cleaned_csv)
    row_map, col_map = _exported_positions(
        original, final, approved_row_operations, approved_column_deletions)
    sha = hashlib.sha256(original_csv).hexdigest()
    audit_rows: list[dict] = []
    marked_evidence: list[dict] = []
    finding_titles = finding_titles or {}
    completed_audits = completed_audits or {}

    def location(r: int | None, c: int | None) -> str:
        if type(r) is int and type(c) is int:
            return f"R{r}:C{c}"
        if type(r) is int:
            return f"Row {r}"
        if type(c) is int:
            return f"Column {c}"
        return ""

    def record(event: dict, operation: str, *, row: int | None = None,
               col: int | None = None, before: str = "", after: str = "") -> dict:
        if type(row) is int and not 1 <= row <= len(original):
            raise PermissionError("Audit references a nonexistent source CSV row")
        if type(col) is int and not 1 <= col <= len(original[0]):
            raise PermissionError("Audit references a nonexistent source CSV column")
        if type(row) is int and type(col) is int:
            original_value = original[row-1][col-1]
        elif type(row) is int:
            original_value = original[row-1][0]
        elif type(col) is int:
            original_value = original[0][col-1]
        else:
            original_value = ""
        # An approved edit's 'before' can be a previously staged value; keep
        # BOTH the immutable original and the historical reviewer proposal.
        if type(row) is int and row not in row_map or type(col) is int and col not in col_map:
            target_row = target_col = None
            disposition = "superseded by approved deletion"
            final_value = ""
        else:
            target_row = row_map.get(row) if type(row) is int else None
            target_col = col_map.get(col) if type(col) is int else None
            if type(target_row) is int and type(target_col) is int:
                final_value = final[target_row - 1][target_col - 1]
            else:
                final_value = ""
            disposition = ("source data retained; style applied" if operation == "flag_cells" else
                           "approved structural action" if operation in {"drop_rows", "drop_column", "merge_rows"} else
                           "reconciled to final dataset")
        out = {
            "finding_id": _safe_text(event.get("finding_id")),
            "finding": _safe_text(event.get("finding") or finding_titles.get(event.get("finding_id", ""), "")),
            "reviewer": _safe_text(event.get("reviewer") or reviewer),
            "operation": operation,
            "original_location": location(row, col),
            "final_location": location(target_row, target_col),
            "original_csv_value": original_value,
            "approved_before": _safe_text(before),
            "approved_after": _safe_text(after),
            "final_csv_value": final_value,
            "outcome": disposition,
        }
        audit_rows.append(out)
        return out

    for event in history or []:
        if not isinstance(event, dict) or event.get("dataset_id") != dataset_id:
            continue
        operation = str(event.get("operation") or "")
        event_hash = event.get("source_sha256")
        if event_hash and event_hash != sha:
            raise PermissionError("Approved event is not bound to the current immutable source")
        if operation == "flag_cells":
            for flag in event.get("flags") or []:
                row, col = flag.get("row"), flag.get("column")
                if (type(row) is not int or type(col) is not int or
                        not (2 <= row <= len(original)) or not (1 <= col <= len(original[0])) or
                        flag.get("style") != "bold_red" or
                        flag.get("before") != original[row-1][col-1]):
                    raise PermissionError("Audited flag does not match its original source cell")
                result = record(event, operation, row=row, col=col,
                                before=flag["before"], after=("Bold red / yellow highlight (no value change)" if flag.get("fill", True) else "Bold red font only (no value change)"))
                marked_evidence.append({**result, 'fill': flag.get('fill', True)})
        elif operation in {"replace_cells", "rename_headers"}:
            cells = event.get("approved_cell_changes") or []
            if not cells and isinstance(event.get("changes"), list):
                cells = [c for c in event["changes"] if isinstance(c, dict)
                         and "row" in c and "column" in c]
            if not cells:
                saved = (completed_audits.get(event.get("finding_id")) or {}).get("changes") or []
                cells = [c for c in saved if isinstance(c, dict)
                         and "row" in c and "column" in c]
            if not cells:
                # Preserve the event without inventing missing Before/After.
                record(event, operation, after="Approved edit (per review history; detail unavailable)")
            for c in cells:
                record(event, operation, row=c.get("row"), col=c.get("column"),
                       before=_safe_text(c.get("before")), after=_safe_text(c.get("after")))
        elif operation == "drop_column":
            for item in event.get("modified_positions") or []:
                pos = item.get("column")
                record(event, operation, col=pos,
                       before=original[0][pos-1] if type(pos) is int and 1 <= pos <= len(original[0]) else "",
                       after="Delete entire column")
        elif operation in {"drop_rows", "merge_rows"}:
            for item in event.get("modified_positions") or []:
                row = item.get("row")
                src = item.get("source_row")
                deleted_row = src if operation == "merge_rows" else row
                result = record(event, operation, row=deleted_row,
                                before=original[deleted_row-1][0] if type(deleted_row) is int and 2 <= deleted_row <= len(original) else "",
                                after=(f"Merge counts into source row {row}; remove row {src}"
                                       if operation == "merge_rows" else "Delete entire row"))
                if type(deleted_row) is int and deleted_row not in row_map:
                    result["outcome"] = "removed from final dataset by approved operation"
        else:
            record(event, operation or "unknown", after="See original approval record")

    # Independently reconcile the FINAL exported CSV against the final
    # approved value for every surviving, source-bound cell/header action.
    # An earlier operation at the same source coordinate can be superseded by
    # a later named approval; never claim such an older proposal is final.
    by_target: dict[str, list[dict]] = {}
    for item in audit_rows:
        if (item["operation"] in {"replace_cells", "rename_headers"}
                and item["original_location"] and item["final_location"]
                and item["approved_after"] != ""):
            by_target.setdefault(item["original_location"], []).append(item)
    for records in by_target.values():
        latest = records[-1]
        final_value = latest["final_csv_value"]
        expected = latest["approved_after"]
        equal = expected == final_value
        if not equal and expected and final_value:
            try:
                equal = Decimal(expected).is_finite() and Decimal(expected) == Decimal(final_value)
            except InvalidOperation:
                equal = False
        if not equal:
            raise PermissionError(
                f"Audit/export mismatch at {latest['original_location']}: "
                "the final CSV differs from the last approved cell value")
        for superseded in records[:-1]:
            superseded["outcome"] = "superseded by later approved cell edit"

    # A zero-edit closure is NOT a new deletion. Verify its reference to
    # existing named approvals and show it as a separate review decision.
    for decision in decisions or []:
        if not isinstance(decision, dict) or decision.get('decision') != 'resolved_by_prior_approvals':
            continue
        if (decision.get('source_sha256') != sha
                or not str(decision.get('reviewer') or '').strip()):
            raise PermissionError('Resolved review lacks a source-bound named approval')
        resolved = decision.get('original_csv_rows')
        if (not isinstance(resolved, list) or not resolved
                or any(type(r) is not int or r <= 1 or r > len(original) for r in resolved)
                or len(set(resolved)) != len(resolved)):
            raise PermissionError('Resolved review has invalid original row references')
        if any(r in row_map for r in resolved):
            raise PermissionError('Resolved review references a row still in the final CSV')
        sources = set(decision.get('earlier_approved_findings') or [])
        covered = set()
        for approval in approved_row_operations or []:
            if (approval.get('source_sha256') != sha or
                    approval.get('finding_id') not in sources or
                    not str(approval.get('reviewer') or '').strip()):
                continue
            kind = approval.get('operation')
            for delta in approval.get('changes') or []:
                row = delta.get('row') if kind == 'drop_rows' else (
                    delta.get('source_row') if kind == 'merge_rows' else None)
                if type(row) is int:
                    covered.add(row)
        if not set(resolved).issubset(covered):
            raise PermissionError('Resolved review no longer matches earlier approved deletions')
        fid = str(decision.get('finding_id') or '')
        audit_rows.append({
            'finding_id': fid, 'finding': finding_titles.get(fid, ''),
            'reviewer': str(decision['reviewer']),
            'operation': 'review_resolved_by_prior_approvals',
            'original_location': 'Rows ' + ', '.join(str(r) for r in resolved),
            'final_location': '(removed by earlier approved findings)',
            'original_csv_value': '(see earlier original row deletions)',
            'approved_before': 'Prior named deletions already removed these rows',
            'approved_after': 'No additional changes',
            'final_csv_value': '(not present in final dataset)',
            'outcome': 'Reviewed — resolved by earlier approved deletions',
        })

    # A reviewer may intentionally accept an issue without correction. This
    # is NOT a validated remediation and must be distinguishable from an
    # approved data edit or a no-issue Gemma check in the exported audit.
    for decision in decisions or []:
        if not isinstance(decision, dict) or decision.get('review_outcome') != 'reviewer_accepted_without_correction':
            continue
        if decision.get('source_sha256') != sha:
            raise PermissionError('No-edit close review is not bound to this source CSV')
        if not str(decision.get('reviewer') or '').strip() or decision.get('new_edits') != 0:
            raise PermissionError('No-edit close review lacks a named zero-edit decision')
        fid = str(decision.get('finding_id') or '')
        if review_statuses.get(fid) != 'Reviewed — no change':
            raise PermissionError('No-edit close decision is inconsistent with review status')
        audit_rows.append({
            'finding_id': fid, 'finding': finding_titles.get(fid, ''),
            'reviewer': str(decision['reviewer']),
            'operation': 'review_accepted_unchanged',
            'original_location': 'Finding/check (original source)',
            'final_location': 'No new changes authorized',
            'original_csv_value': '(see immutable original CSV)',
            'approved_before': 'Observed issue or check presented to reviewer',
            'approved_after': 'Reviewer explicitly accepted unchanged state',
            'final_csv_value': '(existing approved changes retained; none added by this review)',
            'outcome': 'Reviewed — accepted unchanged; not corrected',
        })

    # Reviewer decisions are independent from cell edits: a keep-as-is choice
    # must still be retained, with no fabricated replacement value.
    findings = [
        {"finding_id": _safe_text(fid), "title": finding_titles.get(fid, ""), "status": status}
        for fid, status in (review_statuses or {}).items()
    ]
    conversations_out = []
    for fid, messages in (conversations or {}).items():
        for index, message in enumerate(messages or [], 1):
            if isinstance(message, dict):
                conversations_out.append({"finding_id": _safe_text(fid), "sequence": index,
                                          "speaker": _safe_text(message.get("role")),
                                          "message": _safe_text(message.get("content"))})
    return {
        "schema_version": "hdq-review-audit-1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "original_filename": Path(original_filename).name,
        "original_sha256": sha,
        "cleaned_sha256": hashlib.sha256(cleaned_csv).hexdigest(),
        "original_rows_including_header": len(original),
        "cleaned_rows_including_header": len(final),
        "original_columns": len(original[0]),
        "cleaned_columns": len(final[0]),
        "reviewer": reviewer,
        "finding_reviews": findings,
        "decisions": [dict(d) for d in (decisions or []) if isinstance(d, dict)],
        "approved_operations": [dict(e) for e in (history or [])
                                if isinstance(e, dict) and e.get("dataset_id") == dataset_id],
        "change_reconciliation": audit_rows,
        "flagged_source_evidence": marked_evidence,
        "conversation": conversations_out,
    }


def reviewer_audit_json(audit: dict) -> bytes:
    return json.dumps(audit, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def embed_reviewer_audit_sheets(excel_bytes: bytes, audit: dict) -> bytes:
    """Attach readable review records and original flagged evidence to XLSX.

    The cleaned sheet remains exactly as originally exported. Highlighted
    deleted source rows remain visible ONLY as source evidence, marked removed.
    """
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    book = load_workbook(io.BytesIO(excel_bytes))
    if "Cleaned dataset" not in book.sheetnames:
        raise ValueError("Cleaned dataset worksheet missing")
    sheet = book["Cleaned dataset"]
    # Only insert visual audit sheets; never rewrite data cells.
    meta = book.create_sheet("Audit summary")
    meta.append(["Source", audit["original_filename"]])
    meta.append(["Original CSV SHA-256", audit["original_sha256"]])
    meta.append(["Cleaned CSV SHA-256", audit["cleaned_sha256"]])
    meta.append(["Reviewer", audit["reviewer"]])
    meta.append(["Findings reviewed", len(audit["finding_reviews"])])
    meta.append(["Approved action records", len(audit["approved_operations"])])
    meta.append(["Reviewer interactions", len(audit["conversation"])])
    meta.append(["Note", "Deleted rows/columns cannot be highlighted inside the cleaned dataset. Their approved flags remain on Flagged source evidence."])
    meta.column_dimensions["A"].width = 29
    meta.column_dimensions["B"].width = 105
    for row in meta:
        row[0].font = Font(bold=True)

    def table(name: str, headers: list[str], records: list[dict], fields: list[str], widths: list[int]):
        ws = book.create_sheet(name)
        ws.append(headers)
        for item in records:
            ws.append([_safe_text(item.get(field)) for field in fields])
        for idx, width in enumerate(widths, 1):
            from openpyxl.utils import get_column_letter
            ws.column_dimensions[get_column_letter(idx)].width = width
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFFFF")
            cell.fill = PatternFill("solid", fgColor="FF1F4E78")
        return ws

    changes = table("Audit trail",
        ["Finding ID", "Finding", "Reviewer", "Action", "Original location", "Final location", "Original CSV", "Approved before", "Approved after", "Final CSV", "Result"],
        audit["change_reconciliation"],
        ["finding_id", "finding", "reviewer", "operation", "original_location", "final_location",
         "original_csv_value", "approved_before", "approved_after", "final_csv_value", "outcome"],
        [24, 50, 18, 20, 22, 20, 45, 45, 55, 45, 44])
    for row in changes.iter_rows(min_row=2):
        if "superseded" in _safe_text(row[-1].value) or "removed" in _safe_text(row[-1].value):
            row[-1].font = Font(bold=True, color="FFB45309")
    table("Finding decisions", ["Finding ID", "Title", "Review status"],
          audit["finding_reviews"], ["finding_id", "title", "status"], [24, 55, 36])
    table("Conversation", ["Finding ID", "Sequence", "Speaker", "Message"],
          audit["conversation"], ["finding_id", "sequence", "speaker", "message"], [26, 12, 18, 100])

    flagged = table("Flagged source evidence",
        ["Original location", "Original value (highlight approved)", "Survives at", "Disposition", "Reviewer", "Finding"],
        audit["flagged_source_evidence"],
        ["original_location", "original_csv_value", "final_location", "outcome", "reviewer", "finding"],
        [24, 92, 20, 45, 18, 55])
    for row, record in zip(flagged.iter_rows(min_row=2), audit["flagged_source_evidence"]):
        value = row[1]
        value.font = Font(bold=True, color="FFDC2626")
        if record.get('fill', True):
            value.fill = PatternFill("solid", fgColor="FFFFE699")
        value.alignment = Alignment(wrap_text=True, vertical="top")
    output = io.BytesIO()
    book.save(output)
    return output.getvalue()
