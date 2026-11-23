"""Regression: direct compound columns, embedded named review history and final CSV reconciliation."""
import csv
import io
from pathlib import Path
from openpyxl import Workbook, load_workbook
import pytest

from healthcare_data_quality.cleaning.direct_actions import propose_explicit_column_deletions
from healthcare_data_quality.cleaning.reviewer_audit_workbook import (
    audit_tables, embed_audit_sheets, verify_audit_sheets,
)
from healthcare_data_quality.cleaning.final_export_integrity import verify_final_excel
from healthcare_data_quality.services.recommendation_options import source_backed_choices


def _data(tmp_path):
    source = tmp_path / 'source.csv'
    source.write_text(',A,B,C\nTaxon 1,1,2,3\nTaxon 2,4,5,6\n', encoding='utf-8')
    finding = {'id': 'source:1', 'dataset_id': 'dataset_1', 'check_key': 'find_invalid_counts',
               'title': 'Invalid counts', 'actionable': True,
               'evidence': [{'CSV row': 2, 'Column': 2, 'Original value': '1'}]}
    return source, finding


def test_explicit_multi_column_deletion_is_one_action_group(tmp_path):
    source, finding = _data(tmp_path)
    for text in ['delete columns 2 and 4', 'deleted column 2, 4', 'drop entire columns 2 & 4']:
        result = propose_explicit_column_deletions(text, finding, source)
        assert len(result['operations']) == 2
        assert [x['changes'][0]['position'] for x in result['operations']] == [2, 4]
        assert result['explicit_columns'] == frozenset({2, 4})
        assert all(x['changes'][0]['after'] == '' for x in result['operations'])
        assert result['clarification_options'] == []
    assert propose_explicit_column_deletions('clear cells 2 and 4', finding, source) is None
    assert propose_explicit_column_deletions('do not delete columns 2 and 4', finding, source) is None
    with pytest.raises(ValueError):
        propose_explicit_column_deletions('delete columns 1 and 4', finding, source)


def _base_xlsx(source):
    wb = Workbook(); sheet = wb.active; sheet.title = 'Cleaned dataset'
    for row in csv.reader(io.StringIO(source.read_text())):
        sheet.append([None if text == '' else text for text in row])
    out = io.BytesIO(); wb.save(out); return out.getvalue()


def test_embedded_audit_is_verifiable_and_preserves_named_decisions(tmp_path):
    source, finding = _data(tmp_path)
    orig = source.read_bytes()
    row_ledger = [{'finding_id': 'source:1', 'operation': 'drop_rows', 'reviewer': 'Reviewer X',
                   'changes': [{'row': 3, 'before': 'Taxon 2', 'after': ''}]}]
    col_ledger = [{'finding_id': 'source:2', 'position': 3, 'before': 'B', 'reviewer': 'Reviewer X'}]
    final = b',A,C\nTaxon 1,1,3\n'
    history = [
        {'dataset_id': 'dataset_1', 'finding_id': 'source:1', 'operation': 'drop_rows',
         'reviewer': 'Reviewer X', 'modified_positions': [{'row': 3}], 'changes': 1},
        {'dataset_id': 'dataset_1', 'finding_id': 'source:2', 'operation': 'drop_column',
         'reviewer': 'Reviewer X', 'modified_positions': [{'column': 3}], 'changes': 1},
    ]
    tables = audit_tables(original_csv=orig, final_csv=final, history=history,
                          review_statuses={'source:1': 'Approved row changes for final export',
                                           'source:2': 'Approved column deletion for final export',
                                           'source:3': 'Reviewed — no change'},
                          findings={'source:1': 'Missing rows', 'source:2': 'Controls',
                                    'source:3': 'Blank first header'},
                          approved_rows=row_ledger, approved_columns=col_ledger,
                          discussions={'source:3': [{'role': 'user', 'content': 'keep it as is'},
                                                    {'role': 'assistant', 'content': 'Kept unchanged'}]},
                          reviewer='Reviewer X', verified={'rows': 1})
    assert any(row[1] == 'drop_rows' and row[6] == 'Deleted from final dataset'
               for row in tables['Audit trail'])
    assert any(row[1] == 'drop_column' for row in tables['Audit trail'])
    assert any(row[1] == 'no_change' for row in tables['Audit trail'])
    # The base fixture workbook need only contain final CSV, not original data.
    wb = Workbook(); sheet = wb.active; sheet.title = 'Cleaned dataset'
    for row in csv.reader(io.StringIO(final.decode())): sheet.append(row)
    buff = io.BytesIO(); wb.save(buff)
    attached = embed_audit_sheets(buff.getvalue(), tables)
    verify_audit_sheets(attached, tables)
    assert b'PK' == attached[:2]
    tampered = bytearray(attached)
    # A wrong expected record must not be silently accepted.
    invalid = {**tables, 'Audit trail': [list(row) for row in tables['Audit trail']]}
    invalid['Audit trail'][1][2] = 'Different reviewer'
    with pytest.raises(PermissionError, match='Audit sheet'):
        verify_audit_sheets(attached, invalid)


def test_memory_choice_only_from_exact_reverified_history(tmp_path, monkeypatch):
    source, finding = _data(tmp_path)
    from healthcare_data_quality.jobs import cleaning_jobs
    exact = {'match': 'exact', 'issue_relation': 'same', 'source_match': True,
             'record': {'operations': [{'operation': 'replace_cells', 'changes': [
                 {'row': 2, 'column': 2, 'before': '1', 'after': '2'}]}]}}
    monkeypatch.setattr(cleaning_jobs, 'ranked_reviewed_solutions', lambda *args, **kwargs: [exact])
    opts = source_backed_choices(finding, source, {
        'suggestion': 'Replace with source-verified rule',
        'memory_assessment': {'decision': 'reused', 'reason': 'source match'}})
    assert opts and opts[0]['label'] == 'Reuse source-verified approved method'
    assert opts[0]['rows'][0]['After'] == '2'
    assert 'Prior approved method' in opts[0]['memory_note']
    # A rejected prior method cannot override the model's fresh assessment.
    assert source_backed_choices(finding, source, {
        'suggestion': 'Other plan', 'memory_assessment': {'decision': 'rejected'}}) == []
    wrong = {'match': 'exact', 'issue_relation': 'same', 'source_match': True,
             'record': {'operations': [{'operation': 'replace_cells', 'changes': [
                 {'row': 2, 'column': 2, 'before': '22', 'after': '2'}]}]}}
    monkeypatch.setattr(cleaning_jobs, 'ranked_reviewed_solutions', lambda *args, **kwargs: [wrong])
    assert source_backed_choices(finding, source, {'suggestion': 'Other plan'}) == []


def test_actual_user_column_7_and_11_approved_as_one_compound_batch():
    from healthcare_data_quality.cleaning.finding_cleaning import expand_uniform_evidence_plan
    from healthcare_data_quality.cleaning.compound_approval import evaluate_compound
    source = Path(__file__).resolve().parents[1] / 'data' / 'dirtyL7PVDforXav.csv'
    finding = {'id':'dataset_1:qc:1', 'dataset_id':'dataset_1', 'source':'QC',
               'title':'2 cells contain fractional count values',
               'check_key':'find_invalid_counts', 'actionable': True,
               'evidence':[{'CSV row':11, 'Column':7, 'Original value':'0.2'},
                           {'CSV row':11, 'Column':11, 'Original value':'0.5'}],
               'total_evidence_records':2}
    proposal = propose_explicit_column_deletions('delete columns 7 and 11', finding, source)
    verified = [expand_uniform_evidence_plan(plan, finding, source,
                full_finding_accepted=False,
                user_requested_columns=proposal['explicit_columns'])[0]
                for plan in proposal['operations']]
    reviewed = evaluate_compound(verified, finding, source, source.read_bytes(), [], [],
                                 reviewer='Reviewer X',
                                 requested_columns=proposal['explicit_columns'])
    assert reviewed['export_columns'] == 104
    assert [p['position'] for p in reviewed['columns_after']] == [7, 11]
    assert len(reviewed['events']) == 2  # one atomic named Yes approves both
