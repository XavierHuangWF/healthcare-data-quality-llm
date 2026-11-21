"""The evidence UI shows actual installed detection code on demand."""
import inspect
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


from healthcare_data_quality.ui.evidence_view import DETECTION_IMPLEMENTATIONS, deterministic_detection_code

# Keep this display test independent of an earlier, since-removed test helper.
class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _app(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamlit", ModuleType("streamlit"))
    path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("hdq_detection_evidence_app", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _csv(tmp_path):
    path = tmp_path / "evidence.csv"
    path.write_text("Taxon,ABC99,ABC99\nT1,2,2\nT2,3,3\n", encoding="utf-8")
    return path


def _render_spy(app, monkeypatch):
    calls = []

    class UI:
        def __getattr__(self, name):
            def call(*args, **kwargs):
                calls.append((name, args))
                return _Context() if name in {"expander", "container"} else None
            return call

    monkeypatch.setattr(app, "st", UI())
    return calls

from healthcare_data_quality.investigation.ingestion import inspect_headers
from healthcare_data_quality.ui.finding_presentation import header_findings


def test_every_qc_category_has_actual_project_source_not_placeholder():
    expected = {
        'find_invalid_counts', 'find_duplicate_taxa', 'find_duplicate_ids',
        'find_duplicate_conflicts', 'inspect_taxonomy_structure',
        'inspect_taxonomy_naming', 'compare_repeated_headers', 'inspect_headers',
    }
    assert set(DETECTION_IMPLEMENTATIONS) == expected
    for key in expected:
        code = deterministic_detection_code(key)
        assert code, key
        for record, func in zip(code, DETECTION_IMPLEMENTATIONS[key]):
            assert record['function'] == func.__name__
            assert record['path'].startswith('src/healthcare_data_quality/')
            assert record['code'] == inspect.getsource(func)
            assert record['code'].lstrip().startswith('def ')


def test_arbitrary_check_keys_cannot_read_source_files():
    assert deterministic_detection_code('') == []
    assert deterministic_detection_code('../../../secret.txt') == []
    assert deterministic_detection_code('__import__("os").system("date")') == []


def test_evidence_view_shows_real_comparison_code_without_duplicate_agent_sections(tmp_path, monkeypatch):
    app = _app(monkeypatch)
    calls = _render_spy(app, monkeypatch)
    path = _csv(tmp_path)
    audit = {'dataset_baselines': {'dataset_1': {'checks': {}}}}
    app.render_finding_evidence(audit, [path], [path.name], [inspect_headers(path)], [])
    rendered_code = [args[0] for name, args in calls if name == 'code']
    assert any('def header_findings(' in code and 'counts["matching"]' in code for code in rendered_code)
    assert any('def inspect_headers(' in code for code in rendered_code)
    titles = [args[0] for name, args in calls if name == 'expander']
    assert 'View Python detection code (advanced)' in titles
    assert sum('identical values' in label for label in titles) == 1
    assert 'Gemma investigation evidence' not in ' '.join(str(args) for _, args in calls)


def test_count_validation_code_is_shown_and_matches_the_installed_function(tmp_path, monkeypatch):
    app = _app(monkeypatch)
    calls = _render_spy(app, monkeypatch)
    path = tmp_path / 'counts.csv'
    path.write_text('Taxon,Sample\nk__A,1.5\nk__B,\n', encoding='utf-8')
    baseline = {'checks': {'find_invalid_counts': {
        'status': 'completed',
        'observations': [
            {'row': 2, 'column': 2, 'sample_id': 'Sample', 'taxon': 'k__A',
             'original_value': '1.5', 'issue_type': 'fractional'},
            {'row': 3, 'column': 2, 'sample_id': 'Sample', 'taxon': 'k__B',
             'original_value': '', 'issue_type': 'missing'},
        ]}}}
    audit = {'dataset_baselines': {'dataset_1': baseline}}
    app.render_finding_evidence(audit, [path], [path.name], [inspect_headers(path)], [])
    rendered_code = [args[0] for name, args in calls if name == 'code']
    expected = deterministic_detection_code('find_invalid_counts')[0]['code']
    assert expected in rendered_code
    assert 'number != number.to_integral_value()' in expected
    explanations = ' '.join(str(args) for name, args in calls if name == 'write')
    assert 'integral value' in explanations
    assert 'zero is not missing' in explanations
