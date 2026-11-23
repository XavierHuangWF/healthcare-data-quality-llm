"""Tests for the status panel and non-destructive app.py installer."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import sys

PROJECT = Path(__file__).resolve().parents[1]
TOOLS = PROJECT / 'tools' / 'apply_agent_b_wait_progress.py'
WAIT = PROJECT / 'src' / 'healthcare_data_quality' / 'ui' / 'recommendation_wait.py'


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_actual_elapsed_timestamp_and_real_counts():
    waiting = _load(WAIT, 'hdq_wait_test_module')
    snap = {'created_at': '2026-11-21T20:00:00+00:00', 'entries': {
        'a': {'status': 'ready'}, 'b': {'status': 'running'},
        'c': {'status': 'pending'}, 'd': {'status': 'failed'},
    }}
    state = waiting.recommendation_wait_state(snap)
    assert state['started_epoch_ms'] == int(datetime(2026, 11, 21, 20, tzinfo=timezone.utc).timestamp()*1000)
    assert (state['ready'], state['running'], state['pending'], state['failed']) == (1, 1, 1, 1)
    assert state['processed'] == 2
    html = waiting.recommendation_wait_html(snap)
    assert '1 ready · 1 generating · 1 queued · 1 failed' in html
    assert 'role="progressbar"' in html
    assert 'aria-valuenow="2"' in html
    assert 'window.setInterval(updateClock, 1000)' in html
    assert 'no ETA' in html
    assert '4 of 4 processed' not in html


def test_no_false_running_or_fake_clock_without_job():
    waiting = _load(WAIT, 'hdq_wait_no_job')
    html = waiting.recommendation_wait_html(None)
    assert 'Preparing the recommendation queue' in html
    assert 'No recommendation tasks received yet' in html
    assert 'const start = null;' in html
    assert 'ELAPSED' in html
    assert waiting.recommendation_wait_state({'entries': {'x': {'status': 'failed'}}})['status'] == 'failed'


def test_safe_installer_patches_old_and_new_app_without_touching_navigation():
    installer = _load(TOOLS, 'hdq_wait_installer')
    for notice in (
        'Waiting for the first Agent B suggestion. Cleaning and Gemma discussion are unavailable until then.',
        "Waiting for Agent B's first verified suggestion. Cleaning will open automatically when it's ready.",
    ):
        app = '''def ui(st, audit, cache_key):\n    st.button("Clean data with LM model")\n    if True:\n        st.info(''' + repr(notice) + ''')\n    st.button("Manage skills & memory")\n'''
        # Use repr() so embedded apostrophes are escaped correctly.
        revised = installer.integrate(app)
        compile(revised, 'app.py', 'exec')
        assert revised.count(installer.PREFIX) == 1
        assert 'st.button("Clean data with LM model")' in revised
        assert 'st.button("Manage skills & memory")' in revised
        assert 'st.fragment(run_every="3s")(_show_hdq_wait_status)()' in revised
        assert installer.integrate(revised) == revised


def test_installer_refuses_ambiguous_app():
    installer = _load(TOOLS, 'hdq_wait_installer_ambiguous')
    try:
        installer.integrate('st.info("Waiting for Agent B")\nst.info("Waiting for Agent B")\n')
    except RuntimeError as exc:
        assert 'app.py was not modified' in str(exc)
    else:
        raise AssertionError('Should refuse multiple matches')
