"""Regression coverage for source-file paths after reorganizing agent modules."""
from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]


def _app(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamlit", ModuleType("streamlit"))
    spec = importlib.util.spec_from_file_location("hdq_source_paths_test", ROOT / "app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return app


def test_cache_hash_finds_all_relocated_agent_files(monkeypatch, tmp_path):
    app = _app(monkeypatch)
    base = Path(app.__file__).resolve().parent / 'src' / 'healthcare_data_quality' / 'agents'
    expected = {
        'exploratory_agent.py', 'exploratory_dedup.py',
        'exploratory_combined.py', 'exploratory_sandbox.py',
        'exploratory_runner.py',
    }
    assert expected.issubset({p.name for p in base.iterdir()})
    assert all((base / name).is_file() for name in expected)
    before = app.exploration_cache_key('baseline-key')
    monkeypatch.chdir(tmp_path)
    assert app.exploration_cache_key('baseline-key') == before
    assert app.exploration_cache_key('different-key') != before
    monkeypatch.setenv('HDQ_MODEL_REVISION', 'alternate-revision')
    assert app.exploration_cache_key('baseline-key') != before


def test_cache_key_fails_clearly_on_missing_source(monkeypatch):
    app = _app(monkeypatch)
    original = app.importlib.util.find_spec

    def find_spec(name):
        if name == app._EXPLORATION_SOURCE_MODULES[0]:
            return None
        return original(name)

    monkeypatch.setattr(app.importlib.util, 'find_spec', find_spec)
    try:
        app.exploration_cache_key('baseline-key')
    except FileNotFoundError as error:
        assert app._EXPLORATION_SOURCE_MODULES[0] in str(error)
    else:
        raise AssertionError('A missing source module must fail explicitly')


def test_cache_key_refuses_source_outside_project(monkeypatch, tmp_path):
    app = _app(monkeypatch)
    other = tmp_path / 'exploratory_agent.py'
    other.write_text('pass\n')
    original = app.importlib.util.find_spec

    def find_spec(name):
        if name == app._EXPLORATION_SOURCE_MODULES[0]:
            return SimpleNamespace(origin=str(other))
        return original(name)

    monkeypatch.setattr(app.importlib.util, 'find_spec', find_spec)
    try:
        app.exploration_cache_key('baseline-key')
    except FileNotFoundError as error:
        assert 'outside this project' in str(error)
    else:
        raise AssertionError('Source must be within the project tree')
