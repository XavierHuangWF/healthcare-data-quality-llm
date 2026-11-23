"""End-to-end exported data and style must match *all* named approvals."""
import csv
import hashlib
import io
from pathlib import Path

import pytest

from healthcare_data_quality.cleaning.final_export_integrity import (
    verify_final_csv, verify_final_excel,
)
from healthcare_data_quality.cleaning.styled_export import build_styled_excel_download


def _csv(*rows):
    out = io.StringIO(newline="")
    csv.writer(out).writerows(rows)
    return out.getvalue().encode()


def _case(tmp_path):
    original = _csv(["Taxonomy", "Count", "Drop"], ["Bacteria;A", "1", "x"],
                    ["Bacteria;B", "2", "y"], ["Bacteria;C", "3", "z"])
    path = tmp_path / "source.csv"
    path.write_bytes(original)
    return path, original


def _event(original, *, name="R", fid="f1", op="replace_cells", before="1", after="12"):
    return {
        "dataset_id": "dataset_1", "finding_id": fid, "operation": op,
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "reviewer": name, "modified_positions": [{"csv_row": 2, "column": 2}],
        "approved_cell_changes": [{"row": 2, "column": 2, "before": before, "after": after}],
        "changes": 1,
    }


def _verify(path, original, staged, final=None, *, history=None, statuses=None,
            rows=None, columns=None):
    return verify_final_csv(
        path, staged, final if final is not None else staged,
        history=history or [], review_statuses=statuses or {},
        finding_titles={"f1": "Count correction", "f2": "Flag taxonomy", "f3": "Remove row"},
        approved_rows=rows or [], approved_columns=columns or [],
    )


def test_named_cell_edit_is_replayed_and_final_output_matches(tmp_path):
    path, original = _case(tmp_path)
    staged = original.replace(b",1,x", b",12,x")
    event = _event(original)
    event["staged_sha256"] = hashlib.sha256(staged).hexdigest()
    result = _verify(path, original, staged, history=[event], statuses={"f1": "Applied and verified"})
    assert result["rows"] == 3 and result["columns"] == 3
    assert path.read_bytes() == original


@pytest.mark.parametrize("failure", ["wrong_final", "wrong_before", "tampered_stage", "wrong_source", "unapproved_status"])
def test_unapproved_or_misrepresented_final_data_blocks_download(tmp_path, failure):
    path, original = _case(tmp_path)
    staged = original.replace(b",1,x", b",12,x")
    event = _event(original)
    event["staged_sha256"] = hashlib.sha256(staged).hexdigest()
    final = staged
    statuses = {"f1": "Applied and verified"}
    if failure == "wrong_final":
        final = original
    elif failure == "wrong_before":
        event["approved_cell_changes"][0]["before"] = "999"
    elif failure == "tampered_stage":
        staged = staged.replace(b",2,y", b",99,y")
    elif failure == "wrong_source":
        path.write_bytes(original.replace(b",3,z", b",4,z"))
    elif failure == "unapproved_status":
        statuses = {"f3": "Applied and verified"}
    with pytest.raises(PermissionError):
        _verify(path, original, staged, final, history=[event], statuses=statuses)


def test_flag_only_review_is_not_a_data_repair_and_excel_matches(tmp_path):
    path, original = _case(tmp_path)
    flag = {"row": 2, "column": 1, "before": "Bacteria;A", "style": "bold_red"}
    event = {"dataset_id": "dataset_1", "finding_id": "f2", "operation": "flag_cells",
             "source_sha256": hashlib.sha256(original).hexdigest(),
             "staged_sha256": hashlib.sha256(original).hexdigest(), "reviewer": "R",
             "modified_positions": [], "changes": 1, "flags": [flag]}
    report = _verify(path, original, original, history=[event], statuses={"f2": "Applied and verified"})
    assert report["flag_only_findings"] == 1 and report["flagged_cells"] == 1
    name, xlsx = build_styled_excel_download(
        original, original_csv=original, original_filename="source.csv",
        approved_flags=[flag], approved_row_operations=[], approved_column_deletions=[])
    verify_final_excel(xlsx, original, original_csv=original, approved_flags=[flag],
                       approved_rows=[], approved_columns=[])
    assert path.read_bytes() == original


def test_approved_flags_on_deleted_rows_are_audited_and_do_not_block_export(tmp_path):
    path, original = _case(tmp_path)
    flag = {"row": 2, "column": 1, "before": "Bacteria;A", "style": "bold_red"}
    source_hash = hashlib.sha256(original).hexdigest()
    event_flag = {"dataset_id": "dataset_1", "finding_id": "f2", "operation": "flag_cells",
                  "source_sha256": source_hash, "staged_sha256": source_hash,
                  "reviewer": "R", "modified_positions": [], "changes": 1, "flags": [flag]}
    event_drop = {"dataset_id": "dataset_1", "finding_id": "f3", "operation": "drop_rows",
                  "source_sha256": source_hash, "staged_sha256": source_hash,
                  "reviewer": "R", "modified_positions": [{"row": 2}], "changes": 1}
    rows = [{"operation": "drop_rows", "source_sha256": source_hash, "reviewer": "R",
             "finding_id": "f3", "changes": [{"row": 2, "before": "Bacteria;A", "after": ""}]}]
    final = _csv(["Taxonomy", "Count", "Drop"], ["Bacteria;B", "2", "y"], ["Bacteria;C", "3", "z"])
    result = _verify(path, original, original, final,
                     history=[event_flag, event_drop],
                     statuses={"f2": "Applied and verified", "f3": "Approved row changes for final export"},
                     rows=rows)
    assert result["flagged_cells"] == 0
    assert result["superseded_flags"] == [(2, 1)]
    _, xlsx = build_styled_excel_download(final, original_csv=original,
        original_filename="source.csv", approved_flags=[flag],
        approved_row_operations=rows, approved_column_deletions=[])
    verify_final_excel(xlsx, final, original_csv=original, approved_flags=[flag],
                       approved_rows=rows, approved_columns=[])
    from openpyxl import load_workbook
    ws = load_workbook(io.BytesIO(xlsx)).active
    assert ws["A2"].value == "Bacteria;B"
    assert ws["A2"].font.bold is not True  # Never migrate deleted row's style


def test_revised_annotation_is_removed_from_ledger_so_export_can_proceed(tmp_path):
    path, original = _case(tmp_path)
    source_hash = hashlib.sha256(original).hexdigest()
    drop = {"dataset_id": "dataset_1", "finding_id": "f3", "operation": "drop_rows",
            "source_sha256": source_hash, "staged_sha256": source_hash,
            "reviewer": "R", "modified_positions": [{"row": 2}], "changes": 1}
    rows = [{"operation": "drop_rows", "source_sha256": source_hash, "reviewer": "R",
             "finding_id": "f3", "changes": [{"row": 2, "before": "Bacteria;A", "after": ""}]}]
    final = _csv(["Taxonomy", "Count", "Drop"], ["Bacteria;B", "2", "y"], ["Bacteria;C", "3", "z"])
    result = _verify(path, original, original, final, history=[drop],
                     statuses={"f3": "Approved row changes for final export", "f2": "Reviewed — no change"},
                     rows=rows)
    assert result["removed_rows"] == 1 and result["flagged_cells"] == 0


def test_workbook_without_approved_red_bold_format_is_rejected(tmp_path):
    path, original = _case(tmp_path)
    flag = {"row": 2, "column": 1, "before": "Bacteria;A", "style": "bold_red"}
    _, clean_no_styles = build_styled_excel_download(
        original, original_csv=original, original_filename="source.csv",
        approved_flags=[], approved_row_operations=[], approved_column_deletions=[])
    with pytest.raises(PermissionError, match="highlight"):
        verify_final_excel(clean_no_styles, original, original_csv=original,
                           approved_flags=[flag], approved_rows=[], approved_columns=[])


def test_review_label_does_not_describe_annotation_as_correction():
    root = Path(__file__).resolve().parents[1]
    content = (root / "app.py").read_text()
    assert "Reviewed — flagged (values unchanged)" in content
    assert "verify_final_csv(" in content and "verify_final_excel(" in content


def test_highlight_on_later_replaced_value_requires_new_review(tmp_path):
    path, original = _case(tmp_path)
    staged = original.replace(b"Bacteria;A,1,x", b"Bacteria;Amended,1,x")
    source_hash = hashlib.sha256(original).hexdigest()
    flag = {"row": 2, "column": 1, "before": "Bacteria;A", "style": "bold_red"}
    flag_event = {"dataset_id": "dataset_1", "finding_id": "f2", "operation": "flag_cells",
                  "source_sha256": source_hash, "staged_sha256": source_hash,
                  "reviewer": "R", "modified_positions": [], "changes": 1, "flags": [flag]}
    edit_event = {"dataset_id": "dataset_1", "finding_id": "f1", "operation": "replace_cells",
                  "source_sha256": source_hash, "staged_sha256": hashlib.sha256(staged).hexdigest(),
                  "reviewer": "R", "modified_positions": [{"csv_row": 2, "column": 1}], "changes": 1,
                  "approved_cell_changes": [{"row": 2, "column": 1, "before": "Bacteria;A",
                                              "after": "Bacteria;Amended"}]}
    with pytest.raises(PermissionError, match="highlight"):
        _verify(path, original, staged, history=[flag_event, edit_event],
                statuses={"f1": "Applied and verified", "f2": "Applied and verified"})


def test_approved_cell_replacement_hidden_by_later_row_deletion_is_audited(tmp_path):
    path, original = _case(tmp_path)
    staged = original.replace(b",1,x", b",12,x")
    source_hash = hashlib.sha256(original).hexdigest()
    edit = _event(original)
    edit["staged_sha256"] = hashlib.sha256(staged).hexdigest()
    drop = {"dataset_id": "dataset_1", "finding_id": "f3", "operation": "drop_rows",
            "source_sha256": source_hash, "staged_sha256": hashlib.sha256(staged).hexdigest(),
            "reviewer": "R", "modified_positions": [{"row": 2}], "changes": 1}
    rows = [{"operation": "drop_rows", "source_sha256": source_hash, "reviewer": "R",
             "finding_id": "f3", "changes": [{"row": 2, "before": "Bacteria;A", "after": ""}]}]
    final = _csv(["Taxonomy", "Count", "Drop"], ["Bacteria;B", "2", "y"], ["Bacteria;C", "3", "z"])
    result = _verify(path, original, staged, final, history=[edit, drop],
                     statuses={"f1": "Applied and verified", "f3": "Approved row changes for final export"},
                     rows=rows)
    assert result["superseded_cell_edits"] == [(2, 2)]
    assert result["rows"] == 2


def test_excel_never_executes_untrusted_csv_formula_as_a_formula():
    from openpyxl import load_workbook
    original = _csv(["Taxon", "Count"], ['=HYPERLINK("https://example.invalid")', "2"])
    _, workbook_bytes = build_styled_excel_download(
        original, original_csv=original, original_filename="data.csv",
        approved_flags=[], approved_row_operations=[], approved_column_deletions=[])
    book = load_workbook(io.BytesIO(workbook_bytes), data_only=False)
    assert book.active["A2"].value == '=HYPERLINK("https://example.invalid")'
    assert book.active["A2"].data_type == "s"
    verify_final_excel(workbook_bytes, original, original_csv=original,
                       approved_flags=[], approved_rows=[], approved_columns=[])


def test_four_taxonomy_flags_superseded_by_named_row_deletions_still_export(tmp_path):
    """Regression: Finding 5 original rows 3,5,15,20 later deleted."""
    from healthcare_data_quality.cleaning.in_memory_cleaning import apply_approved_row_operations
    from openpyxl import load_workbook

    raw = _csv(["Taxonomy", "Count"], *[
        [f"Bacteria;Taxon {i}", str(i)] for i in range(2, 27)
    ])
    path = tmp_path / "dataset.csv"
    path.write_bytes(raw)
    source_rows = list(csv.reader(io.StringIO(raw.decode())))
    source_hash = hashlib.sha256(raw).hexdigest()
    original_flag_rows = [3, 5, 15, 20]
    flags = [{"row": row, "column": 1, "before": source_rows[row-1][0],
              "style": "bold_red"} for row in original_flag_rows]
    deleted_rows = [2, 3, 5, 7, 15, 20, 24]
    deletion_changes = [
        {"row": row, "before": source_rows[row-1][0], "after": ""}
        for row in deleted_rows
    ]
    approval = {"operation": "drop_rows", "source_sha256": source_hash,
                "reviewer": "R", "finding_id": "delete_missing",
                "changes": deletion_changes}
    ledger = [
        {"dataset_id": "dataset_1", "finding_id": "flag_taxonomy", "reviewer": "R",
         "operation": "flag_cells", "flags": flags, "changes": 4,
         "source_sha256": source_hash, "staged_sha256": source_hash,
         "modified_positions": []},
        {"dataset_id": "dataset_1", "finding_id": "delete_missing", "reviewer": "R",
         "operation": "drop_rows", "changes": len(deleted_rows),
         "modified_positions": [{"row": row} for row in deleted_rows],
         "source_sha256": source_hash, "staged_sha256": source_hash},
    ]
    final = apply_approved_row_operations(raw, raw, [approval])
    audit = verify_final_csv(
        path, raw, final, history=ledger,
        review_statuses={"flag_taxonomy": "Applied and verified",
                         "delete_missing": "Approved row changes for final export"},
        finding_titles={"flag_taxonomy": "Flag taxonomy",
                        "delete_missing": "Remove missing-value rows"},
        approved_rows=[approval], approved_columns=[],
    )
    assert audit["superseded_flags"] == [(3, 1), (5, 1), (15, 1), (20, 1)]
    assert audit["flagged_cells"] == 0
    _, xlsx = build_styled_excel_download(
        final, original_csv=raw, original_filename="dataset.csv", approved_flags=flags,
        approved_row_operations=[approval], approved_column_deletions=[],
    )
    verify_final_excel(xlsx, final, original_csv=raw, approved_flags=flags,
                       approved_rows=[approval], approved_columns=[])
    sheet = load_workbook(io.BytesIO(xlsx)).active
    assert all(sheet.cell(r, 1).font.bold is not True for r in range(2, sheet.max_row + 1))
    assert path.read_bytes() == raw


def test_deleted_and_surviving_annotations_preserve_correct_export_coordinates(tmp_path):
    from healthcare_data_quality.cleaning.in_memory_cleaning import apply_approved_row_operations
    from openpyxl import load_workbook

    path, raw = _case(tmp_path)
    source_hash = hashlib.sha256(raw).hexdigest()
    flags = [{"row": 2, "column": 1, "before": "Bacteria;A", "style": "bold_red"},
             {"row": 3, "column": 1, "before": "Bacteria;B", "style": "bold_red"}]
    approval = {"operation": "drop_rows", "source_sha256": source_hash,
                "reviewer": "R", "finding_id": "f3",
                "changes": [{"row": 2, "before": "Bacteria;A", "after": ""}]}
    history = [
        {"dataset_id": "dataset_1", "finding_id": "f2", "reviewer": "R",
         "operation": "flag_cells", "flags": flags, "changes": 2,
         "source_sha256": source_hash, "staged_sha256": source_hash,
         "modified_positions": []},
        {"dataset_id": "dataset_1", "finding_id": "f3", "reviewer": "R",
         "operation": "drop_rows", "changes": 1,
         "modified_positions": [{"row": 2}],
         "source_sha256": source_hash, "staged_sha256": source_hash},
    ]
    final = apply_approved_row_operations(raw, raw, [approval])
    audit = _verify(path, raw, raw, final, history=history,
                    statuses={"f2": "Applied and verified",
                              "f3": "Approved row changes for final export"}, rows=[approval])
    assert audit["superseded_flags"] == [(2, 1)]
    assert audit["flagged_cells"] == 1
    _, xlsx = build_styled_excel_download(final, original_csv=raw,
        original_filename="source.csv", approved_flags=flags,
        approved_row_operations=[approval], approved_column_deletions=[])
    verify_final_excel(xlsx, final, original_csv=raw, approved_flags=flags,
                       approved_rows=[approval], approved_columns=[])
    sheet = load_workbook(io.BytesIO(xlsx)).active
    assert sheet["A2"].value == "Bacteria;B"
    assert sheet["A2"].font.bold is True
    assert sheet["A2"].font.color.rgb == "FFDC2626"
    assert sheet["A3"].value == "Bacteria;C"
    assert sheet["A3"].font.bold is not True
