"""Regression tests for distinct investigation, recommendation, chat lanes."""
from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from healthcare_data_quality.llm_client import _RequestGate, _max_active_requests, model_request_priority


@pytest.mark.parametrize('value', ['1', '2', '3'])
def test_allowed_concurrency(monkeypatch, value):
    monkeypatch.setenv('HDQ_LLM_MAX_ACTIVE', value)
    assert _max_active_requests() == int(value)


@pytest.mark.parametrize('value', ['0', '4', '-1', 'two', '1.5'])
def test_invalid_concurrency_fails(monkeypatch, value):
    monkeypatch.setenv('HDQ_LLM_MAX_ACTIVE', value)
    with pytest.raises(ValueError, match='1, 2, or 3'):
        _max_active_requests()


def test_three_distinct_roles_enter_simultaneously(monkeypatch):
    monkeypatch.setenv('HDQ_LLM_MAX_ACTIVE', '3')
    gate = _RequestGate()
    release = threading.Event()
    entered = {x: threading.Event() for x in ('investigation', 'recommendation', 'interactive')}
    errors = []
    def run(role):
        try:
            with gate.admit(role):
                entered[role].set()
                assert release.wait(3)
        except Exception as exc:
            errors.append(exc)
    threads = [threading.Thread(target=run, args=(role,)) for role in entered]
    try:
        for t in threads:
            t.start()
        assert all(event.wait(2) for event in entered.values())
        assert gate._active == 3
        assert gate._active_by_role == dict.fromkeys(entered, 1)
    finally:
        release.set()
        for t in threads:
            t.join(3)
    assert not errors
    assert gate._active == 0


def test_third_background_request_cannot_steal_interactive_slot(monkeypatch):
    monkeypatch.setenv('HDQ_LLM_MAX_ACTIVE', '3')
    gate = _RequestGate()
    release_a = threading.Event()
    start_a, start_a2, done_a2, start_c = [threading.Event() for _ in range(4)]
    threads = [
        threading.Thread(target=lambda: _hold(gate, 'investigation', start_a, release_a)),
        threading.Thread(target=lambda: _record(gate, 'investigation', start_a2, done_a2)),
        threading.Thread(target=lambda: _record(gate, 'interactive', start_c, None)),
    ]
    try:
        threads[0].start()
        assert start_a.wait(2)
        threads[1].start()
        threads[2].start()
        assert start_c.wait(2)
        assert not start_a2.wait(0.05), 'second Agent A request should queue'
    finally:
        release_a.set()
        for t in threads:
            t.join(3)
    assert start_a2.is_set() and done_a2.is_set()


def _hold(gate, role, entered, release):
    with gate.admit(role):
        entered.set()
        assert release.wait(3)


def _record(gate, role, entered, done):
    with gate.admit(role):
        entered.set()
    if done:
        done.set()


def test_two_slots_still_share_one_background_lane(monkeypatch):
    monkeypatch.setenv('HDQ_LLM_MAX_ACTIVE', '2')
    gate = _RequestGate()
    a_started, a_release, b_started, c_started = [threading.Event() for _ in range(4)]
    threads = [
        threading.Thread(target=lambda: _hold(gate, 'investigation', a_started, a_release)),
        threading.Thread(target=lambda: _record(gate, 'recommendation', b_started, None)),
        threading.Thread(target=lambda: _record(gate, 'interactive', c_started, None)),
    ]
    try:
        threads[0].start()
        assert a_started.wait(2)
        threads[1].start()
        threads[2].start()
        assert c_started.wait(2)
        assert not b_started.wait(0.05)
    finally:
        a_release.set()
        for t in threads:
            t.join(3)
    assert b_started.is_set()


def test_compatibility_alias_and_context():
    gate = _RequestGate()
    assert gate._role('background') == 'investigation'
    with model_request_priority('investigation'):
        pass
    with model_request_priority('recommendation'):
        pass
    with pytest.raises(ValueError):
        with model_request_priority('unknown'):
            pass


def test_launcher_rejects_context_below_minimum_and_accepts_three(tmp_path):
    root = Path(__file__).resolve().parents[1]
    launcher = root / 'scripts' / 'serve_gemma4.sh'
    mock_vllm = tmp_path / 'venvs' / 'healthcare-vllm' / 'bin' / 'vllm'
    mock_vllm.parent.mkdir(parents=True)
    mock_vllm.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    mock_vllm.chmod(0o755)
    env = {**os.environ, 'HOME': str(tmp_path), 'HDQ_MAX_MODEL_LEN': '12288',
           'HDQ_MAX_NUM_SEQS': '3'}
    good = subprocess.run(['bash', str(launcher)], env=env,
                          capture_output=True, text=True, timeout=5)
    assert good.returncode == 0, good.stderr
    assert '--max-model-len\n12288' in good.stdout
    assert '--max-num-seqs\n3' in good.stdout
    for var, value in [('HDQ_MAX_MODEL_LEN', '8192'), ('HDQ_MAX_NUM_SEQS', '4')]:
        bad_env = {**env, var: value}
        bad = subprocess.run(['bash', str(launcher)], env=bad_env,
                             capture_output=True, text=True, timeout=5)
        assert bad.returncode == 2


def test_metrics_gauge_parser():
    from scripts.test_gemma_three_way_concurrency import _running_requests_from_metrics
    assert _running_requests_from_metrics(
        '# HELP vllm:num_requests_running ...\n'
        'vllm:num_requests_running{model_name="gemma4-12b"} 3.0\n') == 3
    assert _running_requests_from_metrics('unrelated_metric 99\n') is None
