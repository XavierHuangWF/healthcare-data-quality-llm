"""Regression matrix: local Gemma assessment, retry, and safe failure learning."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from healthcare_data_quality.learning.model_failure_patterns import record_failure, recent_failure_hints
from healthcare_data_quality.services.model_plan_review import accepted_assessed_choices, option_fingerprint
from healthcare_data_quality.jobs.cleaning_jobs import CleaningRecommendationJobs

SOURCE = Path(__file__).resolve().parents[1] / 'data' / 'dirtyL7PVDforXav.csv'


def _finding():
    return {'id': 'dataset_1:qc:8', 'dataset_id': 'dataset_1',
            'check_key': 'compare_repeated_headers', 'source':'QC',
            'title': 'Duplicate columns', 'evidence': [{'Column A':105,'Column B':106,
            'Repeated header':'ABC99'}]}


def _option(column):
    return {'label':f'Delete {column}', 'operations': [{'operation':'drop_column',
            'dataset_id':'dataset_1','changes':[{'position': column,
            'before':'ABC99','after':''}]}]}


def test_failure_learning_saves_only_anonymous_categories(tmp_path):
    path = tmp_path / 'problems.jsonl'
    record_failure(_finding(), 'model_option_review',
                   ValueError('Patient: extremely sensitive data in column 105'),path=path)
    written = path.read_text()
    assert 'Patient' not in written and 'sensitive' not in written
    assert '105' not in written
    report = recent_failure_hints(_finding(),path=path)
    assert len(report)==1 and report[0]['stage']=='model_option_review'
    assert report[0]['recent_count']==1


@pytest.mark.parametrize('allowed', [False,True])
def test_no_option_shown_without_positive_model_assessment(allowed):
    finding = _finding()
    opt = _option(105)
    note = {'fingerprint':option_fingerprint(finding,SOURCE,opt),
            'can_offer':allowed, 'reason':'reviewed against source evidence'}
    result = accepted_assessed_choices(finding,SOURCE,[opt],[note])
    assert len(result) == int(allowed)
    assert (not result or result[0]['gemma_review_reason'] == 'reviewed against source evidence')


def test_model_retries_same_option_without_widening_scope(tmp_path):
    original = _finding()
    calls=[]
    def suggest(_):
        return {'suggestion':'Delete one repeated column','choices':[_option(105)]}
    def assess(finding, source, option, **_):
        calls.append(option['operations'][0]['changes'][0]['position'])
        if len(calls)==1:
            raise ConnectionError('transient endpoint reset')
        return {'fingerprint':option_fingerprint(finding,source,option),
                'can_offer':True, 'reason':'Gemma compared the exact source'}
    jobs=CleaningRecommendationJobs(tmp_path, recommender=suggest, choice_assessor=assess)
    key=jobs.start('retry-source-exact-column', [original],source_paths_by_finding={original['id']:SOURCE})
    jobs._workers[key].join(5)
    entry=jobs.snapshot(key)['entries'][original['id']]
    assert entry['status']=='ready',entry
    assert calls==[105,105]
    assert entry['gemma_option_reviews'][0]['can_offer']


def test_unavailable_model_fails_closed_without_unreviewed_button(tmp_path):
    original=_finding()
    def suggest(_):
        return {'suggestion':'Delete repeated column', 'choices':[_option(105)]}
    failures=[]
    def assess(finding, source, option, **_):
        failures.append(option)
        raise ConnectionError('offline')
    jobs=CleaningRecommendationJobs(tmp_path,recommender=suggest,choice_assessor=assess)
    key=jobs.start('retry-always-unavailable',[original],source_paths_by_finding={original['id']:SOURCE})
    jobs._workers[key].join(5)
    entry=jobs.snapshot(key)['entries'][original['id']]
    assert entry['status']=='failed' and len(failures)==2
    assert not entry.get('gemma_option_reviews')


def test_scope_binding_disallows_reusing_model_review_for_alternative_column():
    finding=_finding()
    option_a=_option(105)
    review={'fingerprint':option_fingerprint(finding,SOURCE,option_a),
            'can_offer':True, 'reason':'A'}
    assert not accepted_assessed_choices(finding,SOURCE,[_option(106)],[review])
