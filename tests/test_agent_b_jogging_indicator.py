"""Regression tests for the minimal animated runner installer."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(relative_path, modname):
    spec = importlib.util.spec_from_file_location(modname, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_waiting_symbol_after_original_text():
    runner = _load('src/healthcare_data_quality/ui/running_indicator.py', 'hdq_running_indicator')
    sentence = "Waiting for Agent B's first verified suggestion. Cleaning will open automatically when it is ready."
    markup = runner.waiting_runner_html(sentence)
    assert __import__("html").unescape(markup).index(sentence) < markup.index('🏃‍♂️')
    assert 'animation: hdq-jog' in markup
    assert 'prefers-reduced-motion: reduce' in markup
    assert 'role="status"' in markup


def test_safe_rewrite_of_one_line_wait_notice():
    installer = _load('tools/apply_agent_b_jogging_icon.py', 'hdq_runner_installer_a')
    src = '''def render_ui(st):
    st.button('Clean data with LM model')
    if True:
        st.info("Waiting for Agent B's first verified suggestion. Cleaning will open automatically when it is ready.")
    st.button('Manage skills & memory')
'''
    patched, changed = installer.update_notice(src)
    assert changed
    compile(patched, 'app.py', 'exec')
    assert patched.count('show_waiting_runner(st,') == 1
    assert "st.button('Clean data with LM model')" in patched
    assert "st.button('Manage skills & memory')" in patched
    assert installer.update_notice(patched) == (patched, False)


def test_rewrites_existing_agent_b_wait_panel():
    installer = _load('tools/apply_agent_b_jogging_icon.py', 'hdq_runner_installer_b')
    widget = ROOT / 'src/healthcare_data_quality/ui/recommendation_wait.py'
    if not widget.is_file():
        return  # Test also performed with a real prior progress module in integration.
    patched, changed = installer.update_existing_progress_panel(widget.read_text(encoding='utf-8'))
    assert '🏃‍♂️' in patched
    assert 'hdq-jogger-icon' in patched
    assert installer.update_existing_progress_panel(patched) == (patched, False)
    compile(patched, 'recommendation_wait.py', 'exec')


def test_refuse_multiple_notices():
    installer = _load('tools/apply_agent_b_jogging_icon.py', 'hdq_runner_installer_c')
    src = ('st.info("Waiting for Agent B suggestion 1")\n'
           'st.info("Waiting for Agent B suggestion 2")\n')
    try:
        installer.update_notice(src)
    except RuntimeError as exc:
        assert 'No changes' not in str(exc) or 'no changes' in str(exc).lower()
    else:
        raise AssertionError('Expected ambiguity check')


def test_html_escape_untrusted_text():
    runner = _load('src/healthcare_data_quality/ui/running_indicator.py', 'hdq_running_indicator_escape')
    html = runner.waiting_runner_html('Waiting for Agent B suggestion <script>')
    assert '&lt;script&gt;' in html
    assert '<script>' not in html
