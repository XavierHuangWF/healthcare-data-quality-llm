"""Regression: a fully superseded finding must have a Gemma-reviewed close option."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from healthcare_data_quality.cleaning.in_memory_cleaning import _records
from healthcare_data_quality.cleaning.previous_approval_resolution import (
    source_verified_prior_row_resolution,
)
from healthcare_data_quality.services import model_plan_review

SOURCE = Path(__file__).resolve().parents[1] / 'data' / 'dirtyL7PVDforXav.csv'


def fixture_source():
    data = _records(SOURCE.read_bytes())
    return data, hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def finding(rows=(3, 5, 15, 20), title='4 rows have an incorrect number of taxonomy levels'):
    return {'id': 'dataset_1:qc:5', 'dataset_id': 'dataset_1',
            'check_key': 'inspect_taxonomy_structure', 'title': title,
            'evidence': [{'CSV row': r, 'Original classification (verbatim)':
                          fixture_source()[0][r-1][0]} for r in rows]}


def approvals(rows=(3, 5, 15, 20)):
    data, sha = fixture_source()
    return [{'operation': 'drop_rows', 'source_sha256': sha,
             'finding_id': 'dataset_1:qc:2', 'reviewer': 'Reviewer A',
             'changes': [{'row': r, 'before': data[r-1][0], 'after': 'Delete row'}
                         for r in rows]}]


def test_all_source_rows_previously_removed_can_be_reviewed_without_redeletion():
    result = source_verified_prior_row_resolution(finding(), SOURCE, approvals())
    assert result and result['rows'] == [3, 5, 15, 20]
    assert result['earlier_finding_ids'] == ['dataset_1:qc:2']
    assert result['source_sha256'] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()


@pytest.mark.parametrize('entries', [
    [], approvals((3, 5, 15)),
    [{**approvals()[0], 'source_sha256': 'bad'}],
    [{**approvals()[0], 'reviewer': ''}],
    [{**approvals()[0], 'changes': [{**approvals()[0]['changes'][0], 'before': 'bogus'}]}],
])
def test_incomplete_or_unattributed_approvals_cannot_close(entries):
    assert source_verified_prior_row_resolution(finding(), SOURCE, entries) is None


def test_evidence_examples_not_full_coverage_do_not_pass():
    assert source_verified_prior_row_resolution(
        finding((3, 5), title='4 rows have an incorrect number of taxonomy levels'),
        SOURCE, approvals()) is None
    assert source_verified_prior_row_resolution(
        {**finding(), 'evidence': []}, SOURCE, approvals()) is None


def test_no_cross_file_approval_or_stale_proof(tmp_path):
    other = tmp_path / 'different.csv'
    other.write_bytes(SOURCE.read_bytes() + b'\n')
    assert source_verified_prior_row_resolution(finding(), other, approvals()) is None
    old = source_verified_prior_row_resolution(finding(), SOURCE, approvals())
    new = source_verified_prior_row_resolution(finding(), SOURCE, approvals((3, 5, 15, 20, 24)))
    # The ledger may contain other approved deletions: this finding's exact
    # source-evidenced scope is unchanged and cannot acquire a fifth row.
    assert old['fingerprint'] == new['fingerprint']


def test_gemma_model_assesses_the_no_edit_closure_before_showing_button(monkeypatch):
    calls = []
    def fake_gemma(prompt, **kwargs):
        calls.append(prompt)
        return json.dumps({'assessment': 'supported',
                           'finding_scope_matches': True,
                           'contains_unrequested_change': False,
                           'reason': 'Prior named approvals already removed all four rows'})
    monkeypatch.setattr(model_plan_review, 'query_local_model', fake_gemma)
    result = model_plan_review.assess_review_option(
        finding(), SOURCE,
        {'label': 'Mark reviewed — resolved by earlier deletions',
         'operations': [], 'decision': 'resolved_by_prior_approvals'},
        user_request='close this finding',
        history=[{'operation':'drop_rows', 'positions':[{'row':x} for x in [3,5,15,20]]}],
    )
    assert result['can_offer'] and len(calls) == 1
    assert 'resolved_by_prior_approvals' in calls[0]
    assert 'close this finding' in calls[0]


def test_app_has_single_explicit_close_action_and_terminal_export_gate():
    src = (Path(__file__).resolve().parents[1] / 'app.py').read_text()
    assert 'if _render_prior_approval_closure(fid, finding, source):' in src
    assert "st.button(str(previous['label'])" in src
    assert "st.session_state.cleaning_review_statuses[fid] = 'Resolved by earlier approved edit'" in src
    assert 'resolved_by_prior_approvals' in src

@pytest.mark.parametrize('decision, rows, label, valid', [
    ('mark_reviewed_resolved', [3,5,15,20], 'Mark reviewed — resolved by earlier approvals', True),
    ('mark_reviewed_resolved', [3,5,15], 'Mark reviewed — resolved by earlier approvals', False),
    ('needs_discussion', [3,5,15,20], 'Mark reviewed — resolved by earlier approvals', False),
    ('mark_reviewed_resolved', [3,5,15,20], 'Delete rows 3, 5, 15 and 20', False),
])
def test_gemma_generates_exact_no_edit_button(monkeypatch,decision,rows,label,valid):
    from healthcare_data_quality.services import resolved_review_button as generator
    def fake(prompt,**kwargs):
        assert 'ZERO data edits' in prompt
        return json.dumps({'decision': decision,'target_rows':rows,'label':label,
                           'reason':'Already approved deletions cover all rows'})
    monkeypatch.setattr(generator,'query_local_model',fake)
    proof=source_verified_prior_row_resolution(finding(),SOURCE,approvals())
    if valid:
        result=generator.generate_prior_approval_close_button(finding(),proof,
                                                               reviewer_request='close finding')
        assert result['label']==label and result['rows']==[3,5,15,20]
    else:
        with pytest.raises((PermissionError,ValueError)):
            generator.generate_prior_approval_close_button(finding(),proof)

@pytest.mark.parametrize('plans, expected', [
    ([{'operation':'drop_rows','changes':[{'row':3},{'row':5}]}],True),
    ([{'operation':'replace_cells','changes':[{'row':15,'column':1}]}],True),
    ([{'operation':'drop_rows','changes':[{'row':3},{'row':6}]}],False),
    ([{'operation':'drop_column','changes':[{'position':3}]}],False),
    ([{'operation':'merge_rows','changes':[{'row':3,'source_row':5}]}],False),
    ([],False),
])
def test_only_exact_overlapping_rows_are_safely_retired(plans,expected):
    from healthcare_data_quality.cleaning.previous_approval_resolution import (
        entirely_superseded_operations,
    )
    proof=source_verified_prior_row_resolution(finding(),SOURCE,approvals())
    assert entirely_superseded_operations(plans,proof) is expected


def test_final_audit_includes_close_as_zero_edit_and_checks_prior_deletions():
    import csv
    from io import StringIO
    from healthcare_data_quality.cleaning.reviewer_audit import build_reviewer_audit
    data, sha = fixture_source()
    removed = (3,5,15,20)
    output = StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(row for i,row in enumerate(data,1) if i not in removed)
    final_csv = output.getvalue().encode('utf8')
    earlier = approvals(removed)
    history = [{'dataset_id':'dataset_1', 'finding_id':'dataset_1:qc:2',
                'operation':'drop_rows','reviewer':'Reviewer A','source_sha256':sha,
                'modified_positions':[{'row':r} for r in removed]}]
    decision = {'finding_id':'dataset_1:qc:5','decision':'resolved_by_prior_approvals',
                'source_sha256':sha,'reviewer':'Reviewer A',
                'original_csv_rows':list(removed),
                'earlier_approved_findings':['dataset_1:qc:2'], 'new_edits':0}
    kwargs = dict(original_csv=SOURCE.read_bytes(),cleaned_csv=final_csv,
                  original_filename='source.csv',history=history,decisions=[decision],
                  review_statuses={'dataset_1:qc:2':'Approved row changes for final export',
                                   'dataset_1:qc:5':'Resolved by earlier approved edit'},
                  conversations={},approved_row_operations=earlier,
                  approved_column_deletions=[],reviewer='Reviewer A',
                  finding_titles={'dataset_1:qc:5':'4 rows have incorrect number of levels'})
    audit=build_reviewer_audit(**kwargs)
    closure = [r for r in audit['change_reconciliation']
               if r['operation']=='review_resolved_by_prior_approvals']
    assert len(closure)==1 and closure[0]['approved_after']=='No additional changes'
    assert '3, 5, 15, 20' in closure[0]['original_location']
    # If an earlier approval is revoked without reopening the closure,
    # the export must FAIL instead of silently declaring the finding done.
    with pytest.raises(PermissionError):
        build_reviewer_audit(**{**kwargs,
            'approved_row_operations':[{'operation':'drop_rows','source_sha256':sha,
                'finding_id':'dataset_1:qc:2','reviewer':'Reviewer A',
                'changes':[{'row':3,'before':data[2][0]}]}]})
    with pytest.raises(PermissionError,match='source-bound'):
        build_reviewer_audit(**{**kwargs,'decisions':[{**decision,'source_sha256':'invalid'}]})

@pytest.mark.parametrize('text,expected',[
    ('ignore it',True),('completed',True),('close this finding',True),
    ('please mark this finding as reviewed',True),
    ('no further changes needed',True),
    ('ignore this and delete row 25',False),
    ('close this finding but also rename column 1',False),
    ('completed? delete 3',False),('',False),
])
def test_explicit_close_commands_are_not_accidentally_compound(text,expected):
    from healthcare_data_quality.cleaning.previous_approval_resolution import (
        requests_close_resolved_finding,
    )
    assert requests_close_resolved_finding(text) is expected


def test_reviewed_prior_deletion_status_releases_gate_only_when_all_reviews_done():
    import ast
    source = (Path(__file__).resolve().parents[1] / 'app.py').read_text()
    module = ast.parse(source)
    fn = next(node for node in module.body if isinstance(node,ast.FunctionDef)
              and node.name=='_cleaning_export_gate')
    env = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]),'app.py','exec'),env)
    gate=env['_cleaning_export_gate']
    catalog=[{'id':'qc2'},{'id':'qc5'}]
    closed={'qc2':'Approved row changes for final export',
            'qc5':'Resolved by earlier approved edit'}
    assert gate(catalog,closed,{}, {}, {}, {}, investigation_running=False)==(True,2,2)
    assert gate(catalog,{**closed,'qc5':'Discussion updated'}, {}, {}, {}, {},
                investigation_running=False)==(False,1,2)
    assert gate(catalog,closed,{'qc5':{'operation':'drop_rows'}}, {}, {}, {},
                investigation_running=False)==(False,1,2)
    assert gate(catalog,closed,{}, {}, {}, {},investigation_running=True)==(False,2,2)
