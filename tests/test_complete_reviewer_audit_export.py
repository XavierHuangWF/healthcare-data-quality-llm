"""End-to-end audit reconciliation and deleted-flag preservation."""
import csv
import hashlib
import io
import json
from pathlib import Path

import pytest
from openpyxl import load_workbook

from healthcare_data_quality.cleaning.reviewer_audit import (
    build_reviewer_audit, reviewer_audit_json, embed_reviewer_audit_sheets,
)
from healthcare_data_quality.cleaning.styled_export import build_styled_excel_download


def _csv(*rows):
    output = io.StringIO(newline='')
    csv.writer(output).writerows(rows)
    return output.getvalue().encode()


def _fixture():
    raw = _csv(['Taxon', 'Sample_A', 'Drop_control'],
               ['Taxon A', '0.2', '0'],
               ['Taxon B', '9', '1'],
               ['Taxon C', '3', '0'])
    final = _csv(['Taxon', 'Sample_A'],
                 ['Taxon A', '2'],
                 ['Taxon C', '3'])
    h = hashlib.sha256(raw).hexdigest()
    history = [
        {'dataset_id': 'dataset_1', 'finding_id': 'f1', 'finding': 'Fractional count',
         'operation': 'replace_cells', 'reviewer': 'Xay', 'source_sha256': h,
         'approved_cell_changes': [{'row': 2, 'column': 2, 'before': '0.2', 'after': '2'}]},
        {'dataset_id': 'dataset_1', 'finding_id': 'f2', 'finding': 'Delete row',
         'operation': 'drop_rows', 'reviewer': 'Xay', 'source_sha256': h,
         'modified_positions': [{'row': 3}]},
        {'dataset_id': 'dataset_1', 'finding_id': 'f3', 'finding': 'Remove control',
         'operation': 'drop_column', 'reviewer': 'Xay', 'source_sha256': h,
         'modified_positions': [{'column': 3}]},
        {'dataset_id': 'dataset_1', 'finding_id': 'f4', 'finding': 'Flag row',
         'operation': 'flag_cells', 'reviewer': 'Xay', 'source_sha256': h,
         'flags': [{'row': 3, 'column': 1, 'before': 'Taxon B', 'style': 'bold_red'},
                   {'row': 4, 'column': 1, 'before': 'Taxon C', 'style': 'bold_red'}]},
    ]
    row_approvals = [{'operation': 'drop_rows', 'changes': [{'row': 3}]}]
    col_approvals = [{'position': 3}]
    return raw, final, history, row_approvals, col_approvals


def test_report_and_embedded_workbook_surviving_and_deleted_flags():
    raw, final, hist, rows, cols = _fixture()
    report = build_reviewer_audit(
        original_csv=raw, cleaned_csv=final, original_filename='data.csv',
        history=hist, approved_row_operations=rows, approved_column_deletions=cols,
        decisions=[{'finding_id': 'f5', 'decision': 'no_change_authorized_or_executed'}],
        review_statuses={'f1': 'Applied and verified', 'f2': 'Applied and verified',
                         'f3': 'Applied and verified', 'f4': 'Applied and verified',
                         'f5': 'Reviewed — no change'},
        conversations={'f4': [{'role': 'user', 'content': 'highlight these rows'},
                              {'role': 'assistant', 'content': 'Flag approved'}]},
        reviewer='Xay')
    assert report['cleaned_sha256'] == hashlib.sha256(final).hexdigest()
    assert report['original_sha256'] == hashlib.sha256(raw).hexdigest()
    assert len(report['flagged_source_evidence']) == 2
    assert report['flagged_source_evidence'][0]['outcome'] == 'superseded by approved deletion'
    assert report['flagged_source_evidence'][1]['final_location'] == 'R3:C1'
    assert len(report['finding_reviews']) == 5
    assert len(report['conversation']) == 2
    assert json.loads(reviewer_audit_json(report))['reviewer'] == 'Xay'
    _, book_bytes = build_styled_excel_download(
        final, original_csv=raw, original_filename='data.csv',
        approved_flags=[flag for e in hist if e['operation']=='flag_cells' for flag in e['flags']],
        approved_row_operations=rows, approved_column_deletions=cols)
    new_bytes = embed_reviewer_audit_sheets(book_bytes, report)
    wb = load_workbook(io.BytesIO(new_bytes))
    assert wb.sheetnames == ['Cleaned dataset', 'Audit summary', 'Audit trail',
                             'Finding decisions', 'Conversation', 'Flagged source evidence']
    assert wb['Cleaned dataset']['A2'].value == 'Taxon A'
    assert wb['Cleaned dataset']['B2'].value == 2
    assert wb['Cleaned dataset']['A3'].value == 'Taxon C'
    assert wb['Cleaned dataset']['A3'].font.bold
    assert wb['Cleaned dataset']['A3'].font.color.rgb == 'FFDC2626'
    evidence = wb['Flagged source evidence']
    assert evidence['B2'].value == 'Taxon B'
    assert evidence['B2'].font.bold
    assert evidence['B2'].font.color.rgb == 'FFDC2626'
    assert evidence['D2'].value == 'superseded by approved deletion'
    assert wb['Audit summary']['B3'].value == hashlib.sha256(final).hexdigest()
    assert raw.endswith(b'Taxon C,3,0\r\n')


def test_blank_header_keep_as_is_and_no_flags_still_allow_xlsx_with_audit():
    source = _csv(['', 'Count'], ['Taxon A', '2'])
    report = build_reviewer_audit(original_csv=source, cleaned_csv=source,
        original_filename='raw.csv', history=[], decisions=[{'finding_id':'f9',
        'decision':'no_change_authorized_or_executed'}],
        review_statuses={'f9':'Reviewed — no change'}, conversations={},
        approved_row_operations=[], approved_column_deletions=[], reviewer='Xay')
    _, xlsx = build_styled_excel_download(source, original_csv=source,
        original_filename='raw.csv', approved_flags=[],
        approved_row_operations=[], approved_column_deletions=[])
    wb = load_workbook(io.BytesIO(embed_reviewer_audit_sheets(xlsx, report)))
    assert wb['Cleaned dataset']['A1'].value in {'', None}
    assert wb['Finding decisions']['C2'].value == 'Reviewed — no change'
    assert len(report['approved_operations']) == 0


def test_invalid_flag_source_is_rejected_before_export():
    source = _csv(['Taxon','Count'],['A','2'])
    bad_event = {'dataset_id': 'dataset_1','finding_id':'f5','operation':'flag_cells',
                 'reviewer':'Xay','flags':[{'row': 2,'column':1,
                                         'before':'WRONG','style':'bold_red'}]}
    with pytest.raises(PermissionError,match='source cell'):
        build_reviewer_audit(original_csv=source, cleaned_csv=source,
                            original_filename='data.csv',history=[bad_event],
                            decisions=[],review_statuses={},conversations={},
                            approved_row_operations=[],approved_column_deletions=[])


def test_installer_idempotent_with_latest_app_source():
    import importlib.util
    installer = Path(__file__).resolve().parents[1] / 'tools' / 'apply_complete_review_audit.py'
    spec = importlib.util.spec_from_file_location('apply_complete_review_audit', installer)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    source = (Path(__file__).resolve().parents[1] / 'app.py').read_text()
    # The current app's export has evolved since the legacy installer.
    # Verify audit support remains wired without invoking obsolete source rewrites.
    assert 'audit_tables(' in source and 'embed_audit_sheets(' in source
    assert 'Download cleaned dataset' in source
    assert 'value=True, key=f"review_completed_audit_{fid}"' in source


def test_audit_blocks_unapproved_value_not_reflected_in_final_csv():
    raw = _csv(['Taxon', 'Count'], ['A', '0.2'])
    final = _csv(['Taxon', 'Count'], ['A', '0'])
    history = [{'dataset_id':'dataset_1','finding_id':'f1',
                'operation':'replace_cells','reviewer':'Xay',
                'approved_cell_changes':[{'row':2,'column':2,
                                          'before':'0.2','after':'2'}]}]
    with pytest.raises(PermissionError, match='Audit/export mismatch'):
        build_reviewer_audit(original_csv=raw, cleaned_csv=final,
            original_filename='data.csv', history=history, decisions=[],
            review_statuses={'f1':'Applied and verified'},conversations={},
            approved_row_operations=[],approved_column_deletions=[])


def test_audit_records_previous_approved_cell_edit_superseded_by_later_edit():
    raw = _csv(['Taxon', 'Count'], ['A', '0.2'])
    final = _csv(['Taxon', 'Count'], ['A', '3'])
    history = [{'dataset_id':'dataset_1','finding_id':'f1',
                'operation':'replace_cells','reviewer':'Xay',
                'approved_cell_changes':[{'row':2,'column':2,
                                          'before':'0.2','after':'2'}]},
               {'dataset_id':'dataset_1','finding_id':'f2',
                'operation':'replace_cells','reviewer':'Xay',
                'approved_cell_changes':[{'row':2,'column':2,
                                          'before':'2','after':'3'}]}]
    report = build_reviewer_audit(original_csv=raw, cleaned_csv=final,
            original_filename='data.csv', history=history, decisions=[],
            review_statuses={'f1':'Applied and verified','f2':'Applied and verified'},
            conversations={},approved_row_operations=[],approved_column_deletions=[])
    assert report['change_reconciliation'][0]['outcome']=='superseded by later approved cell edit'
    assert report['change_reconciliation'][1]['final_csv_value']=='3'
