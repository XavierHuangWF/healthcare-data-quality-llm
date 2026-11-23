"""Real lane arbitration and queue-claim tests without a live model endpoint."""
import threading
import time
from collections import Counter

from healthcare_data_quality.jobs.cleaning_jobs import CleaningRecommendationJobs
from healthcare_data_quality.services import llm_client
from healthcare_data_quality.ui.exploration_banner import (
    render_agent_c_banner_html, render_recommendation_banner_html,
)


def _findings(count=4):
    return [{"id": f"dataset_1:qc:{i}", "dataset_id": "dataset_1",
             "title": f"Finding {i}", "source": "QC", "method": "Test",
             "actionable": True, "evidence": [{"row": i}]}
            for i in range(1, count + 1)]


def _join_workers(manager, key):
    for _ in range(100):
        with manager._lock:
            threads = [t for t in (manager._workers.get(key),
                                    manager._assist_workers.get(key)) if t]
        for thread in threads:
            thread.join(.05)
        snapshot = manager.snapshot(key)
        if all(e['status'] in {'ready', 'failed'} for e in snapshot['entries'].values()):
            return snapshot
    raise AssertionError('Jobs did not complete')


def test_two_workers_share_vllm_without_duplicate_claims(tmp_path, monkeypatch):
    monkeypatch.setenv('HDQ_LLM_MAX_ACTIVE', '3')
    monkeypatch.setenv('HDQ_AGENT_C_ASSIST_B', '1')
    gate = llm_client._RequestGate()
    monkeypatch.setattr(llm_client, '_GATE', gate)
    entered_b, entered_c = threading.Event(), threading.Event()
    allow_finish = threading.Event()
    calls = []
    lock = threading.Lock()

    def recommend(finding):
        role = llm_client._REQUEST_PRIORITY.get()
        with gate.admit(role):
            with lock:
                calls.append((finding['id'], role))
            if role == 'recommendation':
                entered_b.set()
            if role == 'idle_assist':
                entered_c.set()
            assert allow_finish.wait(3)
        return {'suggestion': 'Review verified evidence', 'decision_needed': 'Select a method'}

    manager = CleaningRecommendationJobs(tmp_path, recommender=recommend,
                                          assist_enabled=True)
    key = manager.start('source', _findings(4), run_id='run1')
    try:
        assert entered_b.wait(2), 'Primary B lane should work'
        assert entered_c.wait(2), 'Idle C lane should help B'
        snapshot = manager.snapshot(key)
        active = [x for x in snapshot['entries'].values() if x['status'] == 'running']
        assert len(active) == 2
        assert {x['worker_role'] for x in active} == {'agent_b', 'agent_c_idle_assist'}
        assert 'Agent C — assisting Agent B' in render_agent_c_banner_html(
            assisting_findings=1, idle=True)
        assert 'Agent C assisting on 1' in render_recommendation_banner_html(snapshot)
    finally:
        allow_finish.set()
    final = _join_workers(manager, key)
    assert all(x['status'] == 'ready' for x in final['entries'].values())
    assert Counter(fid for fid, _ in calls) == Counter(f['id'] for f in _findings(4))
    assert any(x.get('assisted_by_agent_c') for x in final['entries'].values())
    assert all(x.get('assisted_by_agent_c') is not None for x in final['entries'].values())
    # Rerendering/requeueing never repeats an already finished finding.
    manager.start('source', _findings(4), run_id='run1')
    assert Counter(fid for fid, _ in calls) == Counter(f['id'] for f in _findings(4))


def test_idle_assistance_does_not_launch_when_chat_active(tmp_path, monkeypatch):
    monkeypatch.setenv('HDQ_LLM_MAX_ACTIVE', '3')
    gate = llm_client._RequestGate()
    monkeypatch.setattr(llm_client, '_GATE', gate)
    chat_started = threading.Event()
    release_chat = threading.Event()

    def chat():
        with gate.admit('interactive'):
            chat_started.set()
            assert release_chat.wait(4)

    chat_thread = threading.Thread(target=chat, daemon=True)
    chat_thread.start()
    assert chat_started.wait(1)
    calls = []
    manager = CleaningRecommendationJobs(
        tmp_path, recommender=lambda f: (calls.append(f['id']) or {'suggestion': 'Check source'}),
        assist_enabled=True)
    try:
        key = manager.start('csv', _findings(4), run_id='run2')
        assert key not in manager._assist_workers
        result = _join_workers(manager, key)
        assert all(x['status'] == 'ready' for x in result['entries'].values())
        assert len(calls) == 4
    finally:
        release_chat.set()
        chat_thread.join(2)


def test_capacity_two_and_explicit_disable_keep_sequential_workers(tmp_path, monkeypatch):
    for capacity, enabled in [('2', '1'), ('3', '0')]:
        monkeypatch.setenv('HDQ_LLM_MAX_ACTIVE', capacity)
        monkeypatch.setenv('HDQ_AGENT_C_ASSIST_B', enabled)
        folder = tmp_path / (capacity + enabled)
        manager = CleaningRecommendationJobs(
            folder, recommender=lambda f: {'suggestion': 'Review'},
            assist_enabled=True)
        key = manager.start('csv', _findings(3), run_id='run')
        assert key not in manager._assist_workers
        assert all(x['status'] == 'ready' for x in _join_workers(manager, key)['entries'].values())


def test_interactive_chat_is_next_after_borrowed_inference(tmp_path, monkeypatch):
    monkeypatch.setenv('HDQ_LLM_MAX_ACTIVE', '3')
    gate = llm_client._RequestGate()
    monkeypatch.setattr(llm_client, '_GATE', gate)
    entered = threading.Event()
    release = threading.Event()
    chat_entered = threading.Event()
    events = []

    def borrowed():
        with gate.admit('idle_assist'):
            entered.set()
            assert release.wait(3)
        events.append('borrowed_end')

    def chat():
        with gate.admit('interactive'):
            events.append('chat')
            chat_entered.set()

    t1 = threading.Thread(target=borrowed)
    t1.start()
    assert entered.wait(1)
    t2 = threading.Thread(target=chat)
    t2.start()
    for _ in range(100):
        with gate._condition:
            if gate._waiting_interactive:
                assert not gate._can_admit('idle_assist', 3)
                break
        time.sleep(.005)
    else:
        raise AssertionError('Chat failed to queue')
    assert not chat_entered.is_set()
    release.set()
    assert chat_entered.wait(2)
    for thread in (t1, t2):
        thread.join(2)
    assert events[-1] == 'chat' or 'chat' in events
    assert gate._active == 0


def test_mixed_qc_and_gemma_check_entries_are_claimed_once(tmp_path, monkeypatch):
    monkeypatch.setenv('HDQ_LLM_MAX_ACTIVE', '3')
    monkeypatch.setenv('HDQ_AGENT_C_ASSIST_B', '1')
    seen = []
    lock = threading.Lock()

    def recommend(finding):
        with lock:
            seen.append(finding['id'])
        return {'suggestion': 'Source-verifiable approach', 'decision_needed': 'Review'}

    findings = _findings(3) + [{
        'id': 'gemma:1:control', 'source': 'Gemma check', 'title': 'Control columns',
        'method': 'Inspect controls', 'actionable': True, 'evidence': [],
    }]
    manager = CleaningRecommendationJobs(tmp_path, recommender=recommend,
                                          assist_enabled=True)
    key = manager.start('source', findings, run_id='mixed-run')
    state = _join_workers(manager, key)
    assert set(state['entries']) == {f['id'] for f in findings}
    assert all(e['status'] == 'ready' for e in state['entries'].values())
    assert Counter(seen) == Counter(f['id'] for f in findings)
