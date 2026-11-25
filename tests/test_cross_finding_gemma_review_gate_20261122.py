"""Regression matrix for reviewer intent, model gating and cross-finding edits."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from healthcare_data_quality.services import model_plan_review as gateway
from healthcare_data_quality.cleaning.explicit_columns import explicit_column_deletion
from healthcare_data_quality.cleaning.intent_scope import (
    explicitly_requested_columns, reject_unrequested_extra_columns,
)
from healthcare_data_quality.cleaning.proposal_conflicts import (
    filter_conflicting_choices, proposal_conflicts,
)
from healthcare_data_quality.cleaning.cleaning_workspace import validate_header_plan
from healthcare_data_quality.ui.finding_wait_gate import is_locked
from healthcare_data_quality.cleaning.direct_actions import (
    draft_format_correction, same_taxonomy_identity,
)

CSV = Path(__file__).resolve().parents[1] / 'data' / 'dirtyL7PVDforXav.csv'


def finding(key='compare_repeated_headers'):
    return {'id': 'dataset_1:qc:8', 'dataset_id': 'dataset_1', 'check_key': key,
            'title': 'Columns 105 and 106 have identical values', 'actionable': True,
            'evidence': [{'Column A': 105, 'Column B': 106, 'Repeated header': 'ABC99'}]}


def deletion(*positions):
    with CSV.open(encoding='utf-8-sig', newline='') as f:
        header = next(csv.reader(f))
    return [{'operation': 'drop_column', 'dataset_id': 'dataset_1',
             'changes': [{'position': p, 'before': header[p-1], 'after': ''}]}
            for p in positions]


@pytest.mark.parametrize('sentence, positions', [
    ('delete 105', {105}), ('delete column 105', {105}),
    ('Delete entire Column 105', {105}),
    ('delete columns 105 and 106', {105, 106}),
    ('drop column 106', {106}),
])
def test_exact_reviewer_column_scope(sentence, positions):
    original = finding()
    assert explicitly_requested_columns(sentence, original) == positions
    plan = explicit_column_deletion(sentence, original, CSV)
    assert plan is not None
    assert {o['changes'][0]['position'] for o in plan['operations']} == positions
    reject_unrequested_extra_columns(sentence, original, plan['operations'])


def test_finding8_single_column_never_inherits_unrequested_pair():
    with pytest.raises(PermissionError, match='do not match'):
        reject_unrequested_extra_columns('delete 105', finding(), deletion(105, 106))
    with pytest.raises(PermissionError, match='do not match'):
        reject_unrequested_extra_columns('delete 105', finding(), deletion(106))
    assert explicit_column_deletion('delete 105', finding('inspect_headers'), CSV) is None


def test_gemma_scope_assessment_is_recorded_and_source_bound(monkeypatch):
    option = {'label': 'Delete only column 105', 'operations': deletion(105),
              'rows': [{'Location': 'Column 105', 'Before': 'ABC99', 'After': 'Delete column'}]}
    calls = []
    def stub(prompt, **kwargs):
        calls.append((prompt, kwargs))
        assert 'delete column 105' in prompt.lower()
        assert 'source_sha256' in prompt
        return json.dumps({'assessment':'supported', 'finding_scope_matches':True,
                           'contains_unrequested_change':False, 'reason':'One evidenced column'})
    monkeypatch.setattr(gateway, 'query_local_model', stub)
    proof = gateway.assess_review_option(finding(), CSV, option, user_request='delete column 105')
    assert proof['can_offer'] and len(calls) == 1
    assert gateway.accepted_assessed_choices(finding(), CSV, [option], [proof])
    assert not gateway.accepted_assessed_choices(finding(), CSV, [option], [])
    assert not gateway.accepted_assessed_choices({**finding(), 'id':'other'}, CSV, [option], [proof])
    with TemporaryDirectory() as folder:
        modified = Path(folder) / 'different.csv'
        modified.write_bytes(CSV.read_bytes() + b'\n')
        assert not gateway.accepted_assessed_choices(finding(), modified, [option], [proof])


@pytest.mark.parametrize('assessment, matches, extra', [
    ('uncertain', True, False), ('reject', True, False),
    ('supported', False, False), ('supported', True, True),
])
def test_gemma_model_can_veto_bad_or_excessive_options(monkeypatch,assessment,matches,extra):
    monkeypatch.setattr(gateway, 'query_local_model', lambda *a,**kw: json.dumps({
        'assessment':assessment,'finding_scope_matches':matches,
        'contains_unrequested_change':extra,'reason':'Semantic scope review'}))
    option = {'label':'Delete column 105', 'operations': deletion(105)}
    result = gateway.assess_review_option(finding(), CSV, option)
    assert not result['can_offer']
    assert gateway.accepted_assessed_choices(finding(), CSV, [option], [result]) == []


def test_malformed_gemma_review_fails_closed(monkeypatch):
    monkeypatch.setattr(gateway, 'query_local_model', lambda *a, **kw: '{"assessment":"supported"}')
    with pytest.raises(ValueError, match='omitted required'):
        gateway.assess_review_option(finding(), CSV, {'label':'Delete', 'operations':deletion(105)})


def test_delete_row_prevents_double_deletion_and_cell_edit():
    original = CSV.read_bytes()
    removed = [{'operation':'drop_rows', 'changes':[{'row':7, 'before':'original'}]}]
    assert proposal_conflicts([{'operation':'drop_rows', 'changes':[{'row':7}]}], CSV, original,
                              approved_rows=removed) == ['Row 7 was already approved for deletion']
    collisions = proposal_conflicts([{'operation':'replace_cells', 'changes':[
        {'row':7, 'column':1, 'before':'anything','after':'changed'}]}], CSV, original,
        approved_rows=removed)
    assert 'removed' in collisions[0]


def test_delete_column_prevents_duplicate_edit_or_second_drop():
    original = CSV.read_bytes()
    removed = [{'position':105}]
    candidates = [{'label':'Delete105','operations':deletion(105)},
                  {'label':'Delete106','operations':deletion(106)}]
    filtered, errors = filter_conflicting_choices(
        candidates, state={'cleaning_approved_columns_by_dataset': {'dataset_1': removed}},
        finding=finding(), source=CSV)
    assert len(filtered) == 1 and filtered[0]['label'] == 'Delete106'
    assert any('105' in error for error in errors)
    collisions = proposal_conflicts([{'operation':'replace_cells','changes':[
        {'row':4,'column':105}]}], CSV, original, approved_columns=removed)
    assert collisions and 'removed' in collisions[0]


def test_existing_staged_cell_is_not_offered_for_a_second_edit():
    with CSV.open(encoding='utf-8-sig',newline='') as f:
        rows = list(csv.reader(f))
    rows[10][6] = '2'  # physical R11:C7; source had 0.2
    from io import StringIO
    sio = StringIO(newline='')
    writer=csv.writer(sio);writer.writerows(rows)
    conflicts=proposal_conflicts([{'operation':'replace_cells','changes':[
        {'row':11,'column':7,'after':'5'}]}], CSV, sio.getvalue().encode())
    assert conflicts and 'already changed' in conflicts[0]


def test_blank_header_rename_validation_accepts_displayed_slash():
    original = CSV
    proposal = {'operation':'rename_headers','dataset_id':'dataset_1',
                'changes':[{'position':1,'before':'','after':'Taxon / feature'}]}
    checked = validate_header_plan(proposal, [original])
    assert checked['changes'][0]['after']=='Taxon / feature'
    with pytest.raises(ValueError):
        validate_header_plan({'operation':'rename_headers','dataset_id':'dataset_1',
            'changes':[{'position':1,'before':'','after':'ABC99'}]}, [original])


def test_finding6_combines_both_formatting_in_one_causally_safe_edit():
    with CSV.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    before=rows[6][0]
    correct = draft_format_correction({
        'check_key':'inspect_taxonomy_structure', 'dataset_id':'dataset_1',
        'evidence':[{'CSV row':7, 'Original classification':before,
                     'Problem':'Uses | instead of ;'}]
    }, CSV)
    assert correct is not None and len(correct['changes'])==1
    after=correct['changes'][0]['after']
    assert '|' not in after and 'k__' not in after and len(after.split(';'))==7
    assert same_taxonomy_identity(before,after) and 'SGB000000' in after


def test_failed_agent_b_request_is_discussable_instead_of_permanent_lock():
    assert not is_locked(finding_id='broken',snapshot={'entries':{'broken':{'status':'failed'}}},
                         completed=False,active_conversation=False)
    assert is_locked(finding_id='pending',snapshot={'entries':{'pending':{'status':'running'}}},
                     completed=False,active_conversation=False)


def test_agent_b_worker_assesses_every_offered_option_before_ready(tmp_path):
    from healthcare_data_quality.jobs.cleaning_jobs import CleaningRecommendationJobs
    from healthcare_data_quality.services.model_plan_review import option_fingerprint
    finding8 = {**finding(), 'source':'QC', 'method':'Compare repeated headers'}
    proposed = {'label':'Remove duplicated column 106', 'operations':deletion(106)}
    calls=[]
    def recommender(_item):
        return {'suggestion':'Remove the second identical column',
                'decision_needed':'Retain a single duplicate header',
                'choices':[proposed]}
    def assessor(f, path, option, **kwargs):
        calls.append((f['id'], option['operations']))
        return {'fingerprint':option_fingerprint(f,path,option),
                'can_offer': True, 'reason':'Checked by fake local Gemma'}
    jobs=CleaningRecommendationJobs(tmp_path, recommender=recommender,
                                    choice_assessor=assessor)
    key=jobs.start('test-model-review', [finding8],
                   source_paths_by_finding={finding8['id']:CSV})
    jobs._workers[key].join(4)
    entry=jobs.snapshot(key)['entries'][finding8['id']]
    assert entry['status']=='ready',entry
    assert len(calls)==1 and calls[0][0]==finding8['id']
    assert len(entry['gemma_option_reviews'])==1
    assert entry['gemma_option_reviews'][0]['can_offer']


def test_local_gemma_review_failure_is_reported_not_silently_endorsed(tmp_path):
    from healthcare_data_quality.jobs.cleaning_jobs import CleaningRecommendationJobs
    finding8={**finding(),'source':'QC','method':'Review duplicated header'}
    def recommender(_item):
        return {'suggestion':'Remove duplicated column', 'choices':[
            {'label':'Delete 106','operations':deletion(106)}]}
    def model_unavailable(*args,**kwargs):
        raise ConnectionError('local model unavailable')
    jobs=CleaningRecommendationJobs(tmp_path,recommender=recommender,
                                    choice_assessor=model_unavailable)
    key=jobs.start('test-failed-model-review',[finding8],
                   source_paths_by_finding={finding8['id']:CSV})
    jobs._workers[key].join(4)
    entry=jobs.snapshot(key)['entries'][finding8['id']]
    assert entry['status']=='ready'
    assert entry['gemma_option_reviews'][0]['decision'] == 'unavailable'
    assert not entry['gemma_option_reviews'][0]['can_offer']
    assert 'ConnectionError' in entry['gemma_option_reviews'][0]['reason']


def test_blank_control_options_are_both_reviewable_with_exact_before_after():
    from healthcare_data_quality.services.recommendation_options import source_backed_choices
    item={'id':'gemma:check1','dataset_id':'dataset_1','check_key':'gemma_check',
          'source':'Gemma check', 'title':'Investigate negative controls Rep_Blank, P7_Blank, P6_Blank',
          'actionable':False, 'evidence':[]}
    opts=source_backed_choices(item,CSV,{'suggestion':'Control columns are negative controls'})
    assert len(opts)==2
    assert opts[0]['decision']=='no_change' and opts[0]['rows'][0]['After'].startswith('Keep')
    assert {op['changes'][0]['position'] for op in opts[1]['operations']} == {2,3,4}
    assert all(set(row)>={'Before','After','Location'} for option in opts for row in option['rows'])
