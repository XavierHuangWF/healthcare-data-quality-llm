"""Cross-finding UX, taxonomy reasoning and audited source-to-final contract."""
from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
from openpyxl import load_workbook

from healthcare_data_quality.cleaning.direct_actions import (
    draft_format_correction, taxonomy_convention_evidence, same_taxonomy_identity,
)
from healthcare_data_quality.cleaning.review_reconciliation import independently_reconcile
from healthcare_data_quality.cleaning.reviewer_audit_workbook import (
    audit_tables, embed_audit_sheets, verify_audit_sheets,
)
from healthcare_data_quality.services.recommendation_options import source_backed_choices
from healthcare_data_quality.ui.finding_choice_integrity import (
    scoped_review_choices, scoped_clarification_choices,
)
from healthcare_data_quality.ui.clear_review_actions import final_action_label

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'dirtyL7PVDforXav.csv'


def _source_rows(source=SOURCE):
    with source.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.reader(handle))


def finding6(source=SOURCE):
    rows = _source_rows(source)
    return {'id': 'dataset_1:qc:6', 'dataset_id': 'dataset_1',
            'check_key': 'inspect_taxonomy_structure', 'source': 'QC',
            'title': 'One inconsistent taxonomy separator',
            'evidence': [{'CSV row': 7, 'Original classification': rows[6][0],
                          'Problem': 'Uses | instead of ;'}]}


def test_real_csv_taxonomy_fix_addresses_two_formats_without_changing_biology():
    source_before = SOURCE.read_bytes()
    stats = taxonomy_convention_evidence(SOURCE)
    assert stats['data_rows'] == 158
    assert stats['semicolon_rows'] == 157
    assert stats['majority_style'] == 'unprefixed'
    finding = finding6()
    direct = draft_format_correction(finding, SOURCE)
    assert direct is not None
    change = direct['changes'][0]
    assert change['row'] == 7 and change['column'] == 1
    assert '|' not in change['after']
    assert 'k__' not in change['after']
    assert change['after'].startswith('Bacteria; Actinobacteria; Actinobacteria; ')
    assert change['after'].endswith('Varibaculum; timonense (SGB000000)')
    assert same_taxonomy_identity(change['before'], change['after'])
    assert len(change['after'].split(';')) == 7
    assert SOURCE.read_bytes() == source_before
    verified = source_backed_choices(finding, SOURCE, {'suggestion': 'Improve separators and naming'})
    scoped = scoped_review_choices(finding, SOURCE, verified)
    assert len(scoped) == 1
    assert scoped[0]['rows'][0]['Before'] == _source_rows()[6][0]
    assert scoped[0]['rows'][0]['After'] == change['after']
    assert 'naming' in scoped[0]['label']
    assert 'not independently certified' in scoped[0]['impact']


def test_taxonomy_refuses_bad_prefix_or_missing_rank_instead_of_guessing(tmp_path):
    source = tmp_path / 'bad.csv'
    source.write_text('taxon,v\nk__Bacteria|x__Wrong|c__Class|o__Order|f__Family|g__Genus|s__Spec,1\n'
                      'Bacteria; X; Class; Order; Family; Genus; Spec,2\n', encoding='utf-8')
    finding = {'dataset_id': 'dataset_1', 'check_key': 'inspect_taxonomy_structure',
               'evidence': [{'CSV row': 2,
                             'Original classification': _source_rows(source)[1][0],
                             'Problem': 'Uses | instead of ;'}]}
    assert draft_format_correction(finding, source) is None


def test_no_foreign_clarification_in_taxonomy_and_final_buttons_are_precise():
    finding = finding6()
    choices = scoped_clarification_choices(finding, [
        'Use absolute values', 'Enter specific replacement values',
        'Use the standard naming from other rows', 'Standardize taxonomy separator',
    ])
    assert choices == ['Use the standard naming from other rows', 'Standardize taxonomy separator']
    assert final_action_label({'operation': 'drop_column', 'changes': [{'position': 106}]}) == 'Delete column 106'
    assert final_action_label({'operation': 'drop_rows', 'changes': [{'row': 2}, {'row': 7}]}) == 'Delete rows 2, 7'
    assert final_action_label({'operation': 'replace_cells', 'changes': [{'row': 7, 'column': 1}]}, finding).startswith('Standardize separator and naming')
    assert 'st.form_submit_button("Yes — Approve & Execute"' not in (ROOT / 'app.py').read_text()
    app = (ROOT / 'app.py').read_text()
    assert 'st-key-flag_apply_verified_' in app
    assert 'Download complete verified review ZIP' in app


def test_original_and_final_reconciliation_requires_wholly_authorized_changes(tmp_path):
    orig = b',A,B,C\nTaxon A,1,2,3\nTaxon B,4,5,6\n'
    final = b',A,C\nTaxon A,9,3\n'
    events = [
        {'dataset_id': 'dataset_1', 'finding_id': 'f1', 'reviewer': 'R',
         'operation': 'replace_cells', 'approved_cell_changes': [
             {'row': 2, 'column': 2, 'before': '1', 'after': '9'}], 'approval_group': 'grp1'},
        {'dataset_id': 'dataset_1', 'finding_id': 'f2', 'reviewer': 'R',
         'operation': 'drop_rows', 'modified_positions': [{'row': 3}]},
        {'dataset_id': 'dataset_1', 'finding_id': 'f3', 'reviewer': 'R',
         'operation': 'drop_column', 'modified_positions': [{'column': 3}]},
    ]
    row_ledger = [{'finding_id': 'f2', 'operation': 'drop_rows', 'reviewer': 'R',
                   'changes': [{'row': 3, 'before': 'Taxon B', 'after': ''}]}]
    col_ledger = [{'finding_id': 'f3', 'position': 3, 'before': 'B', 'reviewer': 'R'}]
    titles = {'f1': 'Finding 1 · Correct count', 'f2': 'Finding 2 · Delete row',
              'f3': 'Finding 3 · Remove column', 'f4': 'Finding 4 · Keep header'}
    statuses = {'f1': 'Applied and verified', 'f2': 'Applied and verified',
                'f3': 'Applied and verified', 'f4': 'Reviewed — no change'}
    result = independently_reconcile(original_csv=orig, final_csv=final,
                                    history=events, review_statuses=statuses,
                                    findings=titles, approved_rows=row_ledger,
                                    approved_columns=col_ledger)
    assert result['surviving_changed_cells'] == 1
    assert result['deleted_rows'] == 1
    assert result['deleted_columns'] == 1
    assert result['final_sha256']
    assert len(result['summary']) == 4
    assert any('"Taxon B"' in x[5] and '"6"' in x[5]
               for x in result['details'] if x[2] == 'drop_rows')
    with pytest.raises(PermissionError, match='Unapproved CSV value change'):
        independently_reconcile(original_csv=orig,
                                final_csv=b',A,C\nTaxon A,99,3\n',
                                history=events[:2], review_statuses=statuses,
                                findings=titles, approved_rows=row_ledger,
                                approved_columns=col_ledger)
    tables = audit_tables(original_csv=orig, final_csv=final, history=events,
                          review_statuses=statuses, findings=titles,
                          approved_rows=row_ledger, approved_columns=col_ledger,
                          reviewer='R', verified={'pass': True}, discussions={
                              'f1': [{'role':'user', 'content':'set to 9'},
                                     {'role':'assistant', 'content':'Preview ready'}]})
    assert 'Review overview' in tables
    assert 'Source-to-final checks' in tables
    assert 'Interaction log' in tables
    assert tables['Interaction log'][1][4] == 1
    from openpyxl import Workbook
    wb = Workbook(); wb.active.title = 'Cleaned dataset'
    for row in csv.reader(io.StringIO(final.decode())):
        wb.active.append(row)
    buf = io.BytesIO(); wb.save(buf)
    attachment = embed_audit_sheets(buf.getvalue(), tables)
    verify_audit_sheets(attachment, tables)
    actual = load_workbook(io.BytesIO(attachment))
    assert actual.sheetnames == ['Cleaned dataset', 'Review overview', 'Audit trail',
                                 'Source-to-final checks', 'Interaction log', 'Verification']
    assert actual['Source-to-final checks']['A2'].value.startswith('Finding')
    assert actual['Verification']['B5'].value  # approved final digest
    actual.close()
