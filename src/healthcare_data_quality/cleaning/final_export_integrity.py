"""Fail-closed final-export reconciliation against named reviewer approvals.

The fact that Streamlit marks a finding reviewed is not proof the exported
CSV/XLSX contains its approved actions.  Replay the independent approval ledger
and check the final file before making the download button available.
"""
from __future__ import annotations

import csv
import io
import hashlib
from pathlib import Path

from .cleaning_stage_checkpoint import _approval_ledger_matches_stage
from .in_memory_cleaning import (
    _records, apply_approved_row_operations, apply_approved_column_deletions,
)
from .styled_export import (
    _excel_cell_value, _exported_positions, approved_flags_for_export,
)


def verify_final_csv(
    source: Path, staged: bytes | None, final_csv: bytes, *,
    history: list[dict], review_statuses: dict, finding_titles: dict[str, str],
    approved_rows: list[dict], approved_columns: list[dict],
    dataset_id: str = "dataset_1",
    finding_checks: dict[str, dict] | None = None,
    review_decisions: list[dict] | None = None,
) -> dict:
    """Reconcile every approved action and actual exported CSV; never trust UI state alone.

    Reject missing/forged/revoked approval events, changed original CSVs,
    unapproved staged edits, missing/duplicated deletions and stray data edits.
    Deliberately approved *flags* are styling only, not corrected data.
    """
    original = Path(source).read_bytes()
    staging = staged if staged is not None else original
    if not isinstance(final_csv, bytes) or not isinstance(staging, bytes):
        raise PermissionError("The final dataset must be verified as CSV bytes")
    if not isinstance(history, list) or not isinstance(review_statuses, dict):
        raise PermissionError("Missing reviewer approval history")
    if not isinstance(approved_rows, list) or not isinstance(approved_columns, list):
        raise PermissionError("Missing verified structural approvals")
    if dataset_id != "dataset_1":
        raise ValueError("Only a single registered dataset can be exported")
    # Existing recovery validator independently replays every named Before→After
    # approval in chronological order; reuse its stringent guarantees in the
    # active export path, rather than limiting them to session restoration.
    _approval_ledger_matches_stage(
        [Path(source)], {dataset_id: staging}, {dataset_id: approved_columns},
        {dataset_id: approved_rows}, history, review_statuses, finding_titles,
        review_decisions=review_decisions,
    )
    recomputed = apply_approved_row_operations(original, staging, approved_rows)
    recomputed = apply_approved_column_deletions(
        original, recomputed, approved_columns,
        approved_row_operations=approved_rows,
        staged_before_rows=staging,
    )
    if _records(recomputed) != _records(final_csv):
        raise PermissionError("Download differs from the approved final CSV values")
    flags = approved_flags_for_export(history, dataset_id, original)
    flagged_findings = {
        str(event["finding_id"]) for event in history
        if event.get("dataset_id") == dataset_id and event.get("operation") == "flag_cells"
    }
    edited_findings = {
        str(event["finding_id"]) for event in history
        if event.get("dataset_id") == dataset_id and
        event.get("operation") in {"replace_cells", "rename_headers", "drop_rows", "merge_rows", "drop_column"}
        and (event.get("changes", 1) or 0) > 0
    }
    flag_only = flagged_findings - edited_findings
    original_rows, final_rows = _records(original), _records(final_csv)
    row_map, col_map = _exported_positions(
        original_rows, final_rows, approved_rows, approved_columns)
    if finding_checks:
        from .previous_approval_resolution import source_verified_prior_row_resolution
        for fid, status in review_statuses.items():
            if status != 'Resolved by earlier approved edit' or any(
                    event.get('finding_id') == fid for event in history):
                continue
            finding = finding_checks.get(fid)
            proven = source_verified_prior_row_resolution(
                finding, Path(source), approved_rows) if finding else None
            if not proven:
                raise PermissionError(f'Finding {fid} no longer has complete source-verified overlap evidence')
            referenced = [d for d in (review_decisions or []) if isinstance(d, dict)
                          and d.get('finding_id') == fid
                          and d.get('decision') == 'resolved_by_prior_approvals']
            if (len(referenced) != 1 or
                referenced[0].get('original_csv_rows') != proven['rows'] or
                sorted(referenced[0].get('earlier_approved_findings') or [])
                    != proven['earlier_finding_ids']):
                raise PermissionError(f'Finding {fid} has stale or mismatched overlap review proof')
        from .final_review_trace import verify_unchanged_headers
        verify_unchanged_headers(original_rows, final_rows, review_statuses,
                                 finding_checks, col_map)
    # Deletion is a *separate, explicitly named reviewer approval*. It removes
    # the target itself, so an older annotation on that target is no longer an
    # applicable export instruction. Record the supersession in the audit; do
    # not block the entire download or paint an unrelated surviving cell.
    # The approval ledger has already validated the source, reviewer and exact
    # row/column removals. Validate original flag values even for removed rows.
    for flag in flags:
        r, c = flag["row"], flag["column"]
        if (type(r) is not int or type(c) is not int or
                not 2 <= r <= len(original_rows) or
                not 1 <= c <= len(original_rows[0]) or
                flag.get("style") != "bold_red" or
                flag.get("before") != original_rows[r - 1][c - 1]):
            raise PermissionError("Approved flag does not match original source evidence")
    orphaned_flags = [(f["row"], f["column"]) for f in flags
                      if f["row"] not in row_map or f["column"] not in col_map]
    # Formatting is attached to a surviving physical coordinate, independently
    # of a later named value approval. Original annotation evidence remains
    # immutable; the verified final value is exported with the approved style.
    surviving = {(row_map[f["row"]], col_map[f["column"]]) for f in flags
                 if f["row"] in row_map and f["column"] in col_map}
    # A separately approved structural deletion supersedes an older value
    # change on the removed target. Preserve it in the audit trail instead of
    # incorrectly requiring a second approval to acknowledge the deletion.
    superseded = []
    for event in history:
        if event.get("dataset_id") != dataset_id or event.get("operation") not in {
            "replace_cells", "rename_headers"}:
            continue
        for delta in event.get("approved_cell_changes") or []:
            if delta["row"] not in row_map or delta["column"] not in col_map:
                superseded.append((delta["row"], delta["column"]))
    return {
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "final_sha256": hashlib.sha256(final_csv).hexdigest(),
        "rows": len(final_rows) - 1,
        "columns": len(final_rows[0]),
        "removed_rows": len(original_rows) - len(final_rows),
        "removed_columns": len(original_rows[0]) - len(final_rows[0]),
        "flagged_cells": len(surviving),
        "superseded_flags": orphaned_flags,
        "superseded_cell_edits": superseded,
        "flag_only_findings": len(flag_only),
        "flags": flags,
    }


def verify_final_excel(xlsx: bytes, final_csv: bytes, *, original_csv: bytes,
                       approved_flags: list[dict], approved_rows: list[dict],
                       approved_columns: list[dict]) -> None:
    """Independently read the downloadable XLSX to check values AND styling.

    This is the final delivery verification, not a proof about a preview table.
    Deleted targets cannot move their highlights onto surviving unrelated cells.
    """
    from openpyxl import load_workbook

    raw_rows, final_rows = _records(original_csv), _records(final_csv)
    row_map, col_map = _exported_positions(raw_rows, final_rows, approved_rows, approved_columns)
    try:
        workbook = load_workbook(io.BytesIO(xlsx), read_only=False, data_only=False)
        if workbook.sheetnames not in (
                ["Cleaned dataset"],
                ["Cleaned dataset", "Audit trail", "Interaction log", "Verification"],
                ["Cleaned dataset", "Review overview", "Audit trail",
                 "Source-to-final checks", "Interaction log", "Verification"]):
            raise PermissionError("Excel contains an unexpected worksheet")
        sheet = workbook["Cleaned dataset"]
        if sheet.max_row != len(final_rows) or sheet.max_column != len(final_rows[0]):
            raise PermissionError("Excel rows or columns differ from the approved CSV")
        for row_number, row in enumerate(final_rows, 1):
            for column_number, value in enumerate(row, 1):
                # XLSX does not round-trip CSV empty strings as Python "":
                # openpyxl returns None for a blank cell after saving and
                # reopening. The verified CSV is still authoritative. Accept
                # this *one* representation equivalence without weakening
                # checks for a missing nonempty header or changed data value.
                expected = (None if value == "" else
                            _excel_cell_value(value, column_number))
                observed = sheet.cell(row_number, column_number).value
                if sheet.cell(row_number, column_number).data_type == "f" or observed != expected or type(observed) is not type(expected):
                    raise PermissionError(
                        f"Excel does not preserve approved value at R{row_number}:C{column_number}")
        expected_flags: dict[tuple[int, int], bool] = {}
        for flag in approved_flags:
            row, col = flag["row"], flag["column"]
            if (flag.get("style") != "bold_red" or flag.get("before") != raw_rows[row - 1][col - 1]):
                raise PermissionError("Excel flag does not match original source evidence")
            if row not in row_map or col not in col_map:
                continue
            expected_flags[(row_map[row], col_map[col])] = flag.get("fill", True)
        for row_number in range(2, sheet.max_row + 1):
            for column_number in range(1, sheet.max_column + 1):
                cell = sheet.cell(row_number, column_number)
                is_flagged = (row_number, column_number) in expected_flags
                color = (cell.font.color.rgb if cell.font.color is not None and
                         cell.font.color.type == "rgb" else None)
                fill = (cell.fill.fgColor.rgb if cell.fill.patternType == "solid" and
                        cell.fill.fgColor.type == "rgb" else None)
                actual_flag = (cell.font.bold is True and color == "FFDC2626")
                expected_fill = expected_flags.get((row_number, column_number), False)
                if (actual_flag != is_flagged or (fill == "FFFFE699") != expected_fill
                        or (not is_flagged and (cell.font.bold is True or color == "FFDC2626"))
                        or (not expected_fill and cell.fill.patternType is not None)):
                    raise PermissionError(
                        f"Excel highlights do not match approved flags at R{row_number}:C{column_number}")
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise PermissionError(f"Excel export cannot be independently verified: {exc}") from exc
    finally:
        if 'workbook' in locals():
            workbook.close()


def verified_workbook_evidence(xlsx: bytes, final_csv: bytes, *, original_csv: bytes,
                              approved_flags: list[dict], approved_rows: list[dict],
                              approved_columns: list[dict], audit_tables: dict) -> dict:
    """Bind Gemma's decision to actual read-back data and historical styling.

    The complete actual workbook is checked by Python. The semantic digest is
    stable when verification metadata/timestamps are subsequently appended.
    """
    import json
    from openpyxl import load_workbook
    from .reviewer_audit_workbook import verify_audit_sheets, _annotation_audit_targets
    verify_final_excel(xlsx, final_csv, original_csv=original_csv,
                       approved_flags=approved_flags, approved_rows=approved_rows,
                       approved_columns=approved_columns)
    verify_audit_sheets(xlsx, audit_tables)
    raw, final = _records(original_csv), _records(final_csv)
    rm,cm = _exported_positions(raw,final,approved_rows,approved_columns)
    wb = load_workbook(io.BytesIO(xlsx), data_only=False)
    try:
        sheet = wb['Cleaned dataset']
        def style(cell):
            color = cell.font.color
            return {'bold':bool(cell.font.bold),
                    'red':bool(color is not None and color.type=='rgb' and color.rgb=='FFDC2626'),
                    'yellow':bool(cell.fill.patternType=='solid' and cell.fill.fgColor.type=='rgb' and cell.fill.fgColor.rgb=='FFFFE699')}
        annotations = []
        for f in approved_flags:
            r,c=f['row'],f['column']
            location = [rm[r],cm[c]] if r in rm and c in cm else None
            cell = sheet.cell(*location) if location else None
            annotations.append({'source':[r,c], 'final':location,
                'observed_style':style(cell) if cell else None,
                'observed_value':cell.value if cell else None,
                'expected_yellow':f.get('fill',True)})
        audit_annotations = []
        # Audit trail and source-to-final proofs are stable across metadata updates.
        for name in ('Audit trail','Source-to-final checks'):
            for r,c,fill in _annotation_audit_targets(name,audit_tables[name]):
                cell=wb[name].cell(r,c)
                audit_annotations.append({'sheet':name,'row':r,'coordinate':wb[name].cell(r,4).value,
                                          'source_value':cell.value,'observed_style':style(cell)})
        content = {'actual_cells': [[(cell.value,cell.data_type,style(cell)) for cell in row] for row in sheet.iter_rows()],
                   'audit_trail':list(wb['Audit trail'].values),
                   'source_to_final':list(wb['Source-to-final checks'].values),
                   'overview':list(wb['Review overview'].values),
                   'interactions':list(wb['Interaction log'].values),
                   'audit_annotations':audit_annotations}
        digest=hashlib.sha256(json.dumps(content,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()
        return {'workbook_evidence_sha256':digest,'annotations':annotations,'audit_annotations':audit_annotations,
                'cells_verified':len(final)*len(final[0]), 'audit_verified':True}
    finally:
        wb.close()


def prepare_verified_workbook(*, filename, approved_flags, audit_arguments):
    """CPU worker: construct the actual candidate and verify it before model review."""
    from .styled_export import build_styled_excel_download
    from .reviewer_audit_workbook import audit_tables, embed_audit_sheets
    args = audit_arguments
    # Rebuild only from immutable approved inputs. Never repair by altering data.
    for attempt in range(2):
        try:
            name,base = build_styled_excel_download(args['final_csv'],original_csv=args['original_csv'],
                original_filename=filename,approved_flags=approved_flags,
                approved_row_operations=args['approved_rows'],approved_column_deletions=args['approved_columns'])
            tables = audit_tables(**args)
            provisional=embed_audit_sheets(base,tables)
            proof=verified_workbook_evidence(provisional,args['final_csv'],original_csv=args['original_csv'],
                approved_flags=approved_flags,approved_rows=args['approved_rows'],approved_columns=args['approved_columns'],audit_tables=tables)
            return {'name':name,'base':base,'proof':proof}
        except (PermissionError,ValueError,OSError):
            if attempt:
                raise


def seal_verified_workbook(*, base, approved_flags, audit_arguments, reviewed_proof):
    """CPU worker: attach final model result, reopen and verify delivered bytes."""
    from .reviewer_audit_workbook import audit_tables, embed_audit_sheets
    args=audit_arguments
    result=args.get('verified',{}).get('gemma_final_review') or {}
    if result.get('verified') is not True or result.get('workbook_evidence_sha256')!=reviewed_proof:
        raise PermissionError('No matching Gemma workbook review')
    if (result.get('source_sha256') != hashlib.sha256(args['original_csv']).hexdigest() or
            result.get('final_sha256') != hashlib.sha256(args['final_csv']).hexdigest()):
        raise PermissionError('Gemma final review is bound to different data')
    for attempt in range(2):
        try:
            tables=audit_tables(**args)
            data=embed_audit_sheets(base,tables)
            actual=verified_workbook_evidence(data,args['final_csv'],original_csv=args['original_csv'],
                approved_flags=approved_flags,approved_rows=args['approved_rows'],approved_columns=args['approved_columns'],audit_tables=tables)
            if actual['workbook_evidence_sha256']!=reviewed_proof:
                raise PermissionError('Final workbook differs from Gemma-reviewed workbook evidence')
            return {'verified_excel':data,'workbook_evidence_sha256':reviewed_proof,'artifact_sha256':hashlib.sha256(data).hexdigest()}
        except (PermissionError,ValueError,OSError):
            if attempt:
                raise
