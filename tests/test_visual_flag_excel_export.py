"""Flags are source-bound styling; CSV stays unchanged, Excel retains formatting."""
from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
from openpyxl import load_workbook

from healthcare_data_quality.cleaning.visual_flags import requested_visual_flag
from healthcare_data_quality.cleaning.styled_export import build_styled_excel_download


def _csv(*rows):
    output = io.StringIO(newline="")
    csv.writer(output).writerows(rows)
    return output.getvalue().encode()


def _workbook(data):
    return load_workbook(io.BytesIO(data))["Cleaned dataset"]


def test_deictic_and_direct_taxonomy_flag_sentences(tmp_path):
    raw = _csv(["Taxon", "A"], ["Bacteria;A", "10"], ["Bacteria;B", "20"],
               ["Bacteria;C", "30"], ["Bacteria;D", "40"])
    source = tmp_path / "source.csv"
    source.write_bytes(raw)
    finding = {"id": "f5", "dataset_id": "dataset_1", "evidence":
               [{"CSV row": n} for n in (2, 3, 4, 5)]}
    for instruction in (
        "flag these four rows make the taxonomy bold and red",
        "highlight all of them", "make these taxonomy names bold red",
        "make the font of these four rows column 1 bold and red", "highlight", "flag",
    ):
        proposal = requested_visual_flag(instruction, finding, source)
        assert [(item["row"], item["column"]) for item in proposal["flags"]] == [(r, 1) for r in (2,3,4,5)]
    for instruction in ("highlight these rows", "mark all rows", "highline these four rows"):
        proposal = requested_visual_flag(instruction, finding, source)
        assert [(item["row"], item["column"]) for item in proposal["flags"]] == [(r,c) for r in (2,3,4,5) for c in (1,2)]
    for instruction, rows in (("flag rows 2, 4 and 5", (2,4,5)), ("highlight row 2, row 3 and row 5", (2,3,5))):
        assert [(item["row"], item["column"]) for item in requested_visual_flag(instruction, finding, source)["flags"]] == [(r,c) for r in rows for c in (1,2)]
    assert source.read_bytes() == raw


def test_styled_xlsx_preserves_flagged_text_and_numeric_data_without_changing_csv():
    raw = _csv(["Taxon", "Count", "Precise ID"], ["Taxon A", "10", "00123"],
               ["Taxon B", "4.5", "NA"])
    flags = [{"row": 2, "column": 1, "before": "Taxon A", "style": "bold_red"}]
    name, xlsx = build_styled_excel_download(
        raw, original_csv=raw, original_filename="dirty.csv", approved_flags=flags,
        approved_row_operations=[], approved_column_deletions=[])
    assert name == "cleaned_highlighted_dirty.xlsx"
    ws = _workbook(xlsx)
    assert ws["A2"].value == "Taxon A"
    assert ws["A2"].font.bold is True
    assert ws["A2"].font.color.rgb == "FFDC2626"
    assert ws["A3"].font.bold is not True
    assert ws["B2"].value == 10
    assert ws["B3"].value == 4.5
    assert ws["C2"].value == "00123"
    assert raw == _csv(["Taxon", "Count", "Precise ID"], ["Taxon A", "10", "00123"],
                       ["Taxon B", "4.5", "NA"])


def test_deletions_remap_approved_original_coordinates():
    source = _csv(["Taxon", "Drop", "Keep"], ["A", "1", "10"],
                  ["B", "2", "20"], ["C", "3", "30"], ["D", "4", "40"])
    final = _csv(["Taxon", "Keep"], ["B", "20"], ["C", "30"], ["D", "40"])
    flags = [{"row": 3, "column": 1, "before": "B", "style": "bold_red"},
             {"row": 2, "column": 1, "before": "A", "style": "bold_red"},
             {"row": 5, "column": 1, "before": "D", "style": "bold_red"}]
    _, file = build_styled_excel_download(
        final, original_csv=source, original_filename="raw.csv", approved_flags=flags,
        approved_row_operations=[{"operation": "drop_rows", "changes": [{"row": 2}]}],
        approved_column_deletions=[{"position": 2}])
    ws = _workbook(file)
    assert ws["A2"].value == "B" and ws["A2"].font.bold
    assert ws["A4"].value == "D" and ws["A4"].font.bold
    assert ws["A3"].value == "C" and not ws["A3"].font.bold
    assert ws["B2"].value == 20


def test_mismatched_source_flag_and_structure_are_rejected():
    raw = _csv(["Taxon", "Count"], ["Taxon A", "10"])
    args = dict(cleaned_csv=raw, original_csv=raw, original_filename="dirty.csv",
                approved_row_operations=[], approved_column_deletions=[])
    with pytest.raises(PermissionError, match="visual flag"):
        build_styled_excel_download(**args, approved_flags=[
            {"row": 2, "column": 1, "before": "WRONG", "style": "bold_red"}])
    with pytest.raises(PermissionError, match="dimensions"):
        build_styled_excel_download(**{**args, "cleaned_csv": _csv(["Taxon", "Count"])},
                                    approved_flags=[])


def test_approved_history_drives_export_styles_and_revision_removal():
    from healthcare_data_quality.cleaning.styled_export import approved_flags_for_export
    import hashlib

    raw = _csv(["Taxon", "Count"], ["Bacteria;A", "2"], ["Bacteria;B", "3"])
    sha = hashlib.sha256(raw).hexdigest()
    flag = {"row": 2, "column": 1, "before": "Bacteria;A", "style": "bold_red"}
    event = {"operation": "flag_cells", "dataset_id": "dataset_1",
             "source_sha256": sha, "reviewer": "Reviewer", "flags": [flag]}
    assert approved_flags_for_export([event], "dataset_1", raw) == [flag]
    assert approved_flags_for_export([], "dataset_1", raw) == []  # revised action revoked
    with pytest.raises(PermissionError):
        approved_flags_for_export([{**event, "reviewer": ""}], "dataset_1", raw)
    with pytest.raises(PermissionError):
        approved_flags_for_export([event], "dataset_1", raw + b"\n")


def test_exported_workbook_flag_is_visibly_highlighted_with_bold_red_text():
    raw = _csv(["Taxon", "Count"], ["Bacteria;A", "2"], ["Bacteria;B", "3"])
    flag = {"row": 2, "column": 1, "before": "Bacteria;A", "style": "bold_red"}
    _, data = build_styled_excel_download(raw, original_csv=raw,
        original_filename="sample.csv", approved_flags=[flag],
        approved_row_operations=[], approved_column_deletions=[])
    workbook = _workbook(data)
    assert workbook["A2"].font.bold is True
    assert workbook["A2"].font.color.rgb == "FFDC2626"
    assert workbook["A2"].fill.fgColor.rgb == "FFFFE699"
    assert workbook["A3"].fill.patternType is None


def test_final_review_has_single_download_button_with_excel_default():
    # Protect the user-facing contract without running the full Streamlit app.
    import ast
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "app.py").read_text(encoding="utf-8"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name == "render_cleaning_workspace")
    source_lines = (root / "app.py").read_text(encoding="utf-8").splitlines()
    chunk = "\n".join(source_lines[fn.lineno - 1:fn.end_lineno])
    anchor = chunk.index('    # One export control.')
    tail = chunk[anchor:]
    assert tail.count('st.download_button(') == 2  # one final dataset, one reviewer audit
    assert '"Complete review package (.zip) — original, cleaned, Excel and audit"' in tail
    assert '"Excel (.xlsx) — data and review audit"' in tail
    assert '"CSV (.csv) — cleaned values only"' in tail
    assert 'want_excel = str(file_format).startswith("Excel")' in tail
    assert 'embed_audit_sheets(' in tail


def test_download_reports_flags_deleted_during_cleaning():
    from healthcare_data_quality.cleaning.styled_export import surviving_flag_count
    raw = _csv(["Taxon", "Count"], ["A", "1"], ["B", "2"] )
    cleaned = _csv(["Taxon", "Count"], ["B", "2"])
    count = surviving_flag_count(cleaned, original_csv=raw,
        approved_flags=[{"row": 2, "column": 1, "before": "A", "style": "bold_red"}],
        approved_row_operations=[{"operation": "drop_rows", "changes": [{"row": 2}]}],
        approved_column_deletions=[])
    assert count == 0
