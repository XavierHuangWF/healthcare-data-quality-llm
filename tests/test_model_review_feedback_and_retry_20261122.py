"""Regression matrix: local Gemma assessment, retry, and safe failure learning."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from healthcare_data_quality.learning.model_failure_patterns import record_failure, recent_failure_hints
from healthcare_data_quality.services.model_plan_review import (
    accepted_assessed_choices, option_fingerprint, reviewer_selectable_choices,
)
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


def test_unavailable_model_keeps_source_exact_human_option(tmp_path):
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
    assert entry['status']=='ready' and len(failures)==2
    assert entry['gemma_option_reviews'][0]['decision'] == 'unavailable'
    assert entry['gemma_option_reviews'][0]['can_offer'] is False
    from healthcare_data_quality.services.recommendation_options import source_backed_choices
    from healthcare_data_quality.ui.finding_choice_integrity import scoped_review_choices
    candidates = scoped_review_choices(original, SOURCE, source_backed_choices(
        original, SOURCE, entry))
    selectable = reviewer_selectable_choices(
        original, SOURCE, candidates, entry['gemma_option_reviews'])
    assert selectable and selectable[0]['gemma_review_status'] == 'not_assessed'


def test_scope_binding_disallows_reusing_model_review_for_alternative_column():
    finding=_finding()
    option_a=_option(105)
    review={'fingerprint':option_fingerprint(finding,SOURCE,option_a),
            'can_offer':True, 'reason':'A'}
    assert not accepted_assessed_choices(finding,SOURCE,[_option(106)],[review])


def _fractional_finding():
    return {
        'id': 'dataset_1:qc:1', 'dataset_id': 'dataset_1',
        'check_key': 'find_invalid_counts',
        'title': '2 cells contain fractional count values',
        'evidence': [{'CSV row': 11, 'Column': 7, 'Original value': '0.2'},
                     {'CSV row': 11, 'Column': 11, 'Original value': '0.5'}],
        'total_evidence_records': 2,
    }


def test_actual_fractional_na_choice_remains_visible_when_gemma_disagrees(monkeypatch):
    from healthcare_data_quality.services import recommendation_options as ro
    from healthcare_data_quality.ui.finding_choice_integrity import scoped_review_choices
    from healthcare_data_quality.ui.human_choice_replay import prioritize_verified_human_choices
    finding = _fractional_finding()
    # Stabilize the test independently of memory files uploaded with the project.
    monkeypatch.setattr(ro, '_matching_approved_memory_choice', lambda *args: None)
    monkeypatch.setattr('healthcare_data_quality.ui.human_choice_replay.verified_approved_choices',
                        lambda *args, **kwargs: [])
    proposal = {'suggestion': ("Set 2 cells to NA — row 11, column 7: '0.2' -> 'NA'; "
                               "row 11, column 11: '0.5' -> 'NA'")}
    options = scoped_review_choices(
        finding, SOURCE, prioritize_verified_human_choices(
            finding, SOURCE, ro.source_backed_choices(finding, SOURCE, proposal)))
    assert len(options) == 1
    assert [(r['Before'], r['After']) for r in options[0]['rows']] == [
        ('0.2', 'NA'), ('0.5', 'NA'),
    ]
    refused = {
        'fingerprint': option_fingerprint(finding, SOURCE, options[0]),
        'can_offer': False, 'scope_matches': True, 'has_extra_changes': False,
        'decision': 'uncertain', 'reason': 'Biological meaning of fractional counts unclear',
    }
    alternatives = reviewer_selectable_choices(finding, SOURCE, options, [refused])
    assert len(alternatives) == 1
    assert alternatives[0]['gemma_review_status'] == 'disputed'
    assert 'Biological meaning' in alternatives[0]['gemma_review_reason']
    assert reviewer_selectable_choices(finding, SOURCE, options, [])
    assert not reviewer_selectable_choices(finding, SOURCE, options, [
        {**refused, 'scope_matches': False}])
    assert not reviewer_selectable_choices(finding, SOURCE, options, [
        {**refused, 'has_extra_changes': True}])


def test_fractional_review_renders_named_yes_even_when_gemma_uncertain(monkeypatch):
    from test_cleaning_review_cards import StubUI, _app
    from healthcare_data_quality.services.recommendation_options import source_backed_choices
    finding = _fractional_finding()
    suggestion = {
        'suggestion': "Set 2 cells to NA — row 11, column 7: '0.2' -> 'NA'; row 11, column 11: '0.5' -> 'NA'",
    }
    from healthcare_data_quality.services import recommendation_options as ro
    monkeypatch.setattr(ro, '_matching_approved_memory_choice', lambda *args: None)
    monkeypatch.setattr('healthcare_data_quality.ui.human_choice_replay.verified_approved_choices',
                        lambda *args, **kwargs: [])
    option = source_backed_choices(finding, SOURCE, suggestion)[0]
    suggestion['gemma_option_reviews'] = [{
        'fingerprint': option_fingerprint(finding, SOURCE, option),
        'can_offer': False, 'scope_matches': True, 'has_extra_changes': False,
        'decision': 'reject', 'reason': 'The interpretation needs reviewer validation',
    }]
    class Buttons(StubUI):
        def __init__(self):
            super().__init__()
            self.rendered = []
        def button(self, label, *, key=None, **kwargs):
            self.rendered.append((str(label), key, kwargs.get('disabled')))
            return False
    st = Buttons()
    app = _app(monkeypatch, st)
    st.session_state.cleaning_approved_reviewer = 'Researcher'
    st.session_state.cleaning_recommendations = {finding['id']: suggestion}
    st.session_state.cleaning_plans_by_finding = {}
    st.session_state.cleaning_batch_queue_by_finding = {}
    st.session_state.cleaning_pending_gemma_b = {}
    st.session_state.cleaning_discussion_mode_by_finding = {}
    st.session_state.cleaning_history = []
    st.session_state.cleaning_review_statuses = {}
    app._render_live_agent_b_finding(
        finding['id'], finding, SOURCE, [], [SOURCE], None, 'dataset_1')
    yes = [item for item in st.rendered if item[1] and item[1].startswith('review_yes_')]
    assert len(yes) == 1
    assert 'NA' in yes[0][0] and yes[0][2] is False
    assert any('did not endorse' in str(x) for x in st.comments)
    assert [(item['Before'], item['After']) for record in st.comments
            if isinstance(record, tuple) and record[0] == 'table'
            for item in record[1].to_dict('records')] == [('0.2', 'NA'), ('0.5', 'NA')]


def test_failed_recommendation_has_no_manual_reassessment_or_unreviewed_execution(monkeypatch):
    from test_cleaning_review_cards import StubUI, _app
    from healthcare_data_quality.services import recommendation_options as ro
    finding = _fractional_finding()
    monkeypatch.setattr(ro, '_matching_approved_memory_choice', lambda *args: None)
    monkeypatch.setattr('healthcare_data_quality.ui.human_choice_replay.verified_approved_choices',
                        lambda *args, **kwargs: [])
    class FailedQueue:
        def snapshot(self, _):
            return {'entries': {finding['id']: {
                'status': 'failed', 'error': 'local model unavailable',
                'worker_role': 'agent_b'}}}
    monkeypatch.setattr('healthcare_data_quality.jobs.cleaning_jobs.recommendation_manager',
                        lambda: FailedQueue())
    class Buttons(StubUI):
        def __init__(self):
            super().__init__()
            self.rendered = []
        def button(self, label, *, key=None, **kwargs):
            self.rendered.append((str(label), str(key)))
            return False
    st = Buttons()
    app = _app(monkeypatch, st)
    # Simulate exhausted automatic recovery; no manual model retry or execution control.
    monkeypatch.setattr(app, "_queue_exact_option_recovery", lambda *a, **k: False)
    monkeypatch.setattr(app, "_queue_automatic_option_draft", lambda *a, **k: False)
    st.session_state.cleaning_recommendations = {}
    st.session_state.cleaning_plans_by_finding = {}
    st.session_state.cleaning_batch_queue_by_finding = {}
    st.session_state.cleaning_pending_gemma_b = {}
    st.session_state.cleaning_discussion_mode_by_finding = {}
    st.session_state.cleaning_history = []
    st.session_state.cleaning_review_statuses = {}
    app._render_live_agent_b_finding(
        finding['id'], finding, SOURCE, [], [SOURCE], 'job', 'dataset_1')
    assert not any(key.startswith('review_yes_') for _, key in st.rendered)
    assert not any(label.startswith('Reassess: ') or label == 'Find alternatives' for label, _ in st.rendered)
    assert any('No — Discuss Another Idea' == label for label, _ in st.rendered)


def test_hard_scope_rejection_has_no_manual_reassessment_or_execution(monkeypatch):
    """Merge guard: a model-detected foreign-scope edit must never gain Yes."""
    from test_cleaning_review_cards import StubUI, _app
    from healthcare_data_quality.services import recommendation_options as ro
    finding = _fractional_finding()
    monkeypatch.setattr(ro, '_matching_approved_memory_choice', lambda *args: None)
    monkeypatch.setattr('healthcare_data_quality.ui.human_choice_replay.verified_approved_choices',
                        lambda *args, **kwargs: [])
    suggestion = {
        'suggestion': "Set 2 cells to NA — row 11, column 7: '0.2' -> 'NA'; "
                      "row 11, column 11: '0.5' -> 'NA'",
    }
    proposed = ro.source_backed_choices(finding, SOURCE, suggestion)[0]
    suggestion['gemma_option_reviews'] = [{
        'fingerprint': option_fingerprint(finding, SOURCE, proposed),
        'can_offer': False, 'scope_matches': False,
        'has_extra_changes': True, 'decision': 'rejected',
        'reason': 'Model proposal contains changes outside this finding',
    }]

    class Buttons(StubUI):
        def __init__(self):
            super().__init__()
            self.rendered = []
        def button(self, label, *, key=None, **kwargs):
            self.rendered.append((str(label), str(key)))
            return False

    st = Buttons()
    app = _app(monkeypatch, st)
    # Simulate exhausted automatic recovery; no manual model retry or execution control.
    monkeypatch.setattr(app, "_queue_exact_option_recovery", lambda *a, **k: False)
    monkeypatch.setattr(app, "_queue_automatic_option_draft", lambda *a, **k: False)
    st.session_state.cleaning_approved_reviewer = 'Researcher'
    st.session_state.cleaning_recommendations = {finding['id']: suggestion}
    st.session_state.cleaning_plans_by_finding = {}
    st.session_state.cleaning_batch_queue_by_finding = {}
    st.session_state.cleaning_pending_gemma_b = {}
    st.session_state.cleaning_discussion_mode_by_finding = {}
    st.session_state.cleaning_history = []
    st.session_state.cleaning_review_statuses = {}
    app._render_live_agent_b_finding(
        finding['id'], finding, SOURCE, [], [SOURCE], None, 'dataset_1')

    assert not any(key.startswith('review_yes_') for _, key in st.rendered)
    assert not any(label.startswith('Reassess: ') or label == 'Find alternatives' for label, _ in st.rendered)


@pytest.mark.parametrize('selected_index', [0, 1])
def test_direct_action_titles_pair_each_preview_with_only_its_own_approval(monkeypatch, selected_index):
    from test_cleaning_review_cards import StubUI, _app
    from healthcare_data_quality.services import recommendation_options as ro
    from healthcare_data_quality.ui.finding_choice_integrity import scoped_review_choices
    from healthcare_data_quality.ui.human_choice_replay import prioritize_verified_human_choices

    finding = _fractional_finding()
    fid = finding['id']
    suggestion = {'suggestion': 'Mark fractional counts as NA'}
    remembered = scoped_review_choices(finding, SOURCE, [{
        'label': 'Delete rows 11',
        'operations': [{'operation': 'drop_rows', 'changes': [{'row': 11}]}],
        'provenance': 'Previously approved for this exact source and finding',
    }])
    monkeypatch.setattr(ro, '_matching_approved_memory_choice', lambda *a: None)
    monkeypatch.setattr('healthcare_data_quality.ui.human_choice_replay.verified_approved_choices',
                        lambda *a, **k: remembered)
    options = scoped_review_choices(finding, SOURCE, prioritize_verified_human_choices(
        finding, SOURCE, ro.source_backed_choices(finding, SOURCE, suggestion)))
    assert len(options) == 2
    suggestion['gemma_option_reviews'] = [
        {'fingerprint': option_fingerprint(finding, SOURCE, o), 'can_offer': True,
         'reason': 'Exact scope reviewed'} for o in options]

    class UI(StubUI):
        def __init__(self):
            super().__init__()
            self.events = []
        def dataframe(self, frame, *args, **kwargs):
            self.events.append(('preview', frame.to_dict('records')))
        def markdown(self, label, **kwargs):
            self.events.append(('heading', label))
        def button(self, label, *, key=None, **kwargs):
            self.events.append(('button', label, key, kwargs.get('disabled')))
            return key in self.buttons_on and not kwargs.get('disabled')
    ui = UI()
    app = _app(monkeypatch, ui)
    ui.session_state.update(cleaning_approved_reviewer='', cleaning_recommendations={fid: suggestion},
        cleaning_plans_by_finding={}, cleaning_batch_queue_by_finding={}, cleaning_pending_gemma_b={},
        cleaning_discussion_mode_by_finding={}, cleaning_history=[], cleaning_review_statuses={})
    def render():
        app._render_live_agent_b_finding(fid, finding, SOURCE, [], [SOURCE], None, 'dataset_1')
    original = SOURCE.read_bytes()
    render()
    controls = [e for e in ui.events if e[0] == 'button']
    assert [e[1] for e in controls] == ['Delete row 11', 'Mark 2 fractional counts as NA', 'No — Discuss Another Idea']
    assert all(e[3] for e in controls[:2])  # no named reviewer, no approval
    sequence = [e for e in ui.events if e[0] in {'preview', 'button'}]
    assert [e[0] for e in sequence] == ['preview', 'button', 'preview', 'button', 'button']
    assert sequence[0][1][0]['After'] == 'Delete row'
    assert [(r['Before'],r['After']) for r in sequence[2][1]] == [('0.2','NA'),('0.5','NA')]
    assert not any(e[0] == 'heading' and any(o['label'] in e[1] for o in options) for e in ui.events)
    assert not ui.session_state.cleaning_history
    ui.session_state.cleaning_approved_reviewer = 'Researcher'
    selected = []
    def execute(state, current, source, option):
        selected.append(option['operations'])
        return {}
    monkeypatch.setattr('healthcare_data_quality.ui.one_click_choice.execute_verified_choice', execute)
    monkeypatch.setattr(app, '_rerun_cleaning_fragment', lambda *a: None)
    ui.buttons_on.add(controls[selected_index][2])
    render()
    assert selected == [options[selected_index]['operations']]
    assert SOURCE.read_bytes() == original
