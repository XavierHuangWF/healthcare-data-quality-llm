"""3 logical agents, two background cards, progressive and authorized chat."""
from __future__ import annotations

from threading import Event
from types import SimpleNamespace

import pytest

from healthcare_data_quality.jobs.investigation_jobs import InvestigationJobs
from healthcare_data_quality.ui.exploration_banner import render_recommendation_banner_html
from test_cleaning_review_cards import StubUI, _app, _catalog


def test_investigation_publishes_verified_qc_before_gemma_tool_phase_finishes(tmp_path):
    qc_ready, continue_tools = Event(), Event()
    arrived = []

    def fake_run(path, *, additional_datasets, progress_callback, baseline_ready_callback):
        progress_callback('Running automatic baseline checks...')
        baseline_ready_callback({'status': 'investigating', 'dataset_baselines': {
            'dataset_1': {'checks': {}}}})
        assert continue_tools.wait(3)
        return {'status': 'complete', 'dataset_baselines': {'dataset_1': {'checks': {}}}}

    manager = InvestigationJobs()
    source = tmp_path / 'input.csv'
    source.write_text('sample,count\nA,1\n')
    manager.start('run-one', [source], on_baseline=lambda x: (arrived.append('B'), qc_ready.set()),
                  on_finished=lambda x: arrived.append('exploration'), runner=fake_run)
    assert qc_ready.wait(2)
    partial = manager.snapshot('run-one')
    assert partial['status'] == 'running' and partial['phase'] == 2
    assert partial['partial_audit']['cleaning_recommendation_run_id'] == 'run-one'
    assert 'audit' in partial and partial['audit'] is None
    assert arrived == ['B']   # recommendations have already been dispatched
    continue_tools.set()
    import time
    for _ in range(50):
        state = manager.snapshot('run-one')
        if state['status'] == 'completed':
            break
        time.sleep(0.01)
    assert state['status'] == 'completed'
    assert state['audit']['cleaning_recommendation_run_id'] == 'run-one'
    assert arrived == ['B', 'exploration']


def test_failed_phase2_keeps_qc_snapshot_for_review(tmp_path):
    done = Event()
    def bad_runner(path, *, additional_datasets, progress_callback, baseline_ready_callback):
        baseline_ready_callback({'dataset_baselines': {'dataset_1': {}}})
        done.set()
        raise RuntimeError('registered tool failure')
    source = tmp_path / 'a.csv'; source.write_text('k,v\na,1\n')
    manager = InvestigationJobs()
    manager.start('failed-run', [source], runner=bad_runner,
                  on_baseline=lambda partial: None, on_finished=lambda result: None)
    assert done.wait(2)
    import time
    for _ in range(40):
        state = manager.snapshot('failed-run')
        if state['status'] == 'failed':
            break
        time.sleep(0.01)
    assert state['status'] == 'failed' and state['partial_audit'] is not None
    assert state['audit'] is None


def test_agent_c_waits_for_its_own_agent_b_recommendation(monkeypatch, tmp_path):
    st = StubUI()
    csv = tmp_path / 'data.csv'; csv.write_text('x,y\na,3\n')
    app = _app(monkeypatch, st)
    monkeypatch.setattr('healthcare_data_quality.cleaning.finding_cleaning.build_cleaning_findings',
                        lambda *args, **kwargs: _catalog()[:2])
    monkeypatch.setattr('healthcare_data_quality.cleaning.finding_cleaning.allowed_positions',
                        lambda *args: (set(), set()))
    entries = {'dataset_1:qc:1': {'status': 'ready', 'suggestion': 'Verify before replacing',
                                  'decision_needed': 'Confirm protocol'},
               'dataset_1:qc:2': {'status': 'running'}}
    class Manager:
        def snapshot(self, key): return {'entries': entries}
    monkeypatch.setattr('healthcare_data_quality.jobs.cleaning_jobs.recommendation_manager',lambda:Manager())
    seen_chats = []
    def capture_chat(_placeholder, *, key=None, **kwargs):
        seen_chats.append(key)
        return None
    st.chat_input = capture_chat
    app.render_cleaning_workspace([csv], [SimpleNamespace(name='data.csv')],
                                  {'cleaning_recommendation_run_id': 'test-run'}, 'dataset-key', [])
    # Agent C is eligible but its discussion form must not appear by default.
    assert not seen_chats
    st.session_state.cleaning_discussion_mode_by_finding['dataset_1:qc:1'] = 'alternative'
    app.render_cleaning_workspace([csv], [SimpleNamespace(name='data.csv')],
                                  {'cleaning_recommendation_run_id': 'test-run'}, 'dataset-key', [])
    assert any(key.startswith('review_message_dataset_1:qc:1_') for key in seen_chats)
    assert not any(key.startswith('review_message_dataset_1:qc:2_') for key in seen_chats)
    assert set(st.session_state.cleaning_recommendations) == {'dataset_1:qc:1'}
    with pytest.raises(ValueError, match='Choose an Agent B suggestion or start a direct Agent C discussion'):
        app._queue_agent_c_for_finding(_catalog()[1], csv, [SimpleNamespace(name='data.csv')],
                                       [csv], 'Can I replace it?')


def test_background_recommendations_card_is_agent_b():
    html = render_recommendation_banner_html({'entries': {
        'one': {'status': 'ready'}, 'two': {'status': 'running'}}})
    assert 'Agent B — preparing suggestions' in html
    assert '1 of 2 suggestions ready' in html
    assert 'class="card is-active"' in html


def test_sampled_original_evidence_is_prompt_only_not_stored_in_queue(tmp_path):
    from healthcare_data_quality.jobs.cleaning_jobs import CleaningRecommendationJobs
    seen = []
    def fake(item):
        seen.append(item)
        return {'suggestion': 'Confirm source policy', 'decision_needed': 'Who recorded the values?'}
    jobs = CleaningRecommendationJobs(tmp_path, recommender=fake)
    secret = 'LOCAL-PRIVATE-CELL-987654'
    catalog = [{'id': 'f1', 'dataset_id': 'dataset_1', 'source': 'QC',
                'title': 'Some missing count entries', 'method': 'Count missing cells',
                'evidence': [{'Original value': secret}], 'actionable': True}]
    key = jobs.start('private-run', catalog, run_id='r42')
    jobs._workers[key].join(3)
    assert jobs.snapshot(key)['entries']['f1']['status'] == 'ready'
    assert seen[0]['evidence'][0]['Original value'] == secret
    assert seen[0]['evidence_is_a_sample'] is True
    assert seen[0]['total_evidence_records'] == 1
    assert secret not in (tmp_path / f'{key}.json').read_text()


def test_baseline_callback_failure_does_not_abort_registered_tools(tmp_path):
    from healthcare_data_quality.jobs.investigation_jobs import InvestigationJobs
    from time import sleep
    def fake_run(path, *, additional_datasets, progress_callback, baseline_ready_callback):
        baseline_ready_callback({'dataset_baselines': {'dataset_1': {}}})
        return {'status': 'complete', 'dataset_baselines': {'dataset_1': {}}}
    manager = InvestigationJobs()
    csv = tmp_path / 'file.csv'; csv.write_text('a,b\n1,2\n')
    manager.start('notify-error', [csv], runner=fake_run,
                  on_baseline=lambda partial: (_ for _ in ()).throw(RuntimeError('queue unavailable')),
                  on_finished=lambda full: None)
    for i in range(50):
        state = manager.snapshot('notify-error')
        if state['status'] != 'running': break
        sleep(0.01)
    assert state['status'] == 'completed'
    assert 'queue unavailable' in state['callback_warning']
    assert state['audit'] is not None
