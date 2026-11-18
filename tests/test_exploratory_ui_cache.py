"""UI file/cache contract checks without requiring Streamlit in CI."""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def _app(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamlit", ModuleType("streamlit"))
    spec = importlib.util.spec_from_file_location("hdq_exploratory_app_test", ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_exploratory_code_does_not_invalidate_old_baseline_cache(monkeypatch):
    app = _app(monkeypatch)
    source = ROOT / "src" / "healthcare_data_quality"
    relevant = sorted(p for p in source.rglob("*.py") if not p.name.startswith("exploratory_"))
    relevant.extend([ROOT / "scripts" / "run_workflow.py", ROOT / "scripts" / "serve_gemma4.sh"])
    relevant.extend(sorted((ROOT / "data" / "decision_memory").glob("*.jsonl")))
    digest = hashlib.sha256(f"analysis-v{app.CACHE_SCHEMA_VERSION}".encode())
    digest.update(os.environ.get("HDQ_MODEL_REVISION", "").encode("utf-8"))
    for path in relevant:
        if path.is_file():
            digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    assert app.analysis_fingerprint() == digest.hexdigest()


def test_exploratory_cache_is_private_and_can_restore_results(tmp_path, monkeypatch):
    app = _app(monkeypatch)
    monkeypatch.setattr(app, "EXPLORATION_CACHE_DIR", tmp_path / "private")
    key = "a" * 64
    runs = [{"proposal": {"hypothesis": "Test"}, "result": {"summary": "Checked", "findings": []}}]
    app.save_explorations(key, runs)
    paths = app.saved_explorations(key)
    assert len(paths) == 1
    assert app.load_explorations(paths[0], key) == runs
    assert (paths[0].stat().st_mode & 0o077) == 0
    assert (paths[0].parent.stat().st_mode & 0o077) == 0


def test_baseline_and_exploration_share_one_data_quality_tab(monkeypatch):
    _app(monkeypatch)
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert '["Data quality findings", "Supporting evidence", "Clean data with Gemma"]' in source
    assert 'render_finding_evidence(' in source
    assert 'render_cleaning_workspace(paths, uploaded, audit, cache_key)' in source
    assert '"Exploratory discoveries", "Supporting evidence"' not in source
    assert 'render_exploratory_findings(paths, cache_key, audit)' in source
    assert 'render_baseline(dataset_id, baseline, source_path=paths[label_index])' in source


def test_unified_run_opt_in_starts_nonblocking_exploration(monkeypatch):
    _app(monkeypatch)
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'run_requested = st.button("Run full investigation", type="primary")' in source
    assert '"Include Gemma\'s automatic exploratory analyses (isolated read-only Python)"' in source
    assert 'if run_requested:' in source
    assert 'start_live_exploration(full, cache_key, paths, names)' in source
    assert 'investigation_manager().start(run_id, paths, runner=run_background,' in source
    assert 'sync_live_exploration(audit, cache_key)' in source
    assert 'run_exploratory_series(' in source  # legacy/manual helper preserved
    assert 'def combined_audit_export(' in source
    assert 'healthcare_data_quality_combined_investigation.json' in source


def test_full_run_progress_publishes_real_background_events(monkeypatch):
    _app(monkeypatch)
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    worker = (ROOT / "exploration_jobs.py").read_text(encoding="utf-8")
    registered_worker = (ROOT / "investigation_jobs.py").read_text(encoding="utf-8")
    assert 'progress_callback=progress,' in registered_worker
    assert 'baseline_ready_callback=publish_baseline' in registered_worker
    assert 'st.fragment(run_every="3s")(refresh_explorations)' in source
    assert 'st.fragment(run_every="3s")(refresh_cleaning)' in source
    assert 'with model_request_priority("investigation"):' in worker
    assert "components.html(live_timer_html(self.started_epoch_ms)" in source
    assert "estimated percentage or time remaining" in source


def test_combined_export_preserves_baseline_and_contains_verified_runs(monkeypatch):
    app = _app(monkeypatch)
    monkeypatch.setattr(app, 'exploration_cache_key', lambda key: 'exploration-key')
    app.st.session_state = {'exploration_key': 'exploration-key',
                            'exploration_runs': [{'result': {'findings': [{'title': 'X'}]}}],
                            'exploration_stop': {'decision': 'stop'},
                            'exploration_error': None}
    baseline = {'baseline': {'checks': [1]}}
    report = app.combined_audit_export(baseline, 'key')
    assert baseline == {'baseline': {'checks': [1]}}
    assert report['exploratory_investigations'][0]['result']['findings'][0]['title'] == 'X'
    assert report['exploratory_stop_decision']['decision'] == 'stop'


def test_zero_new_findings_are_a_valid_neutral_ui_outcome(monkeypatch):
    _app(monkeypatch)
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "No additional candidate issues were identified in the Gemma checks completed." in source
    assert "The dataset is not certified clean" not in source  # Never claim certification.
    assert "Try another exploratory investigation" not in source
    assert "Investigation run status (advanced)" in source
    assert 'gemma_stop.get("stop_source") == "proposal_retry_exhausted"' in source
    assert "Gemma recommends stopping further exploration for now" not in source


def test_zero_finding_investigations_do_not_appear_as_issue_cards(monkeypatch):
    app = _app(monkeypatch)
    zero = {"proposal": {"hypothesis": "Check blank controls"},
            "result": {"summary": "No abnormal values", "findings": []}}
    detected = {"proposal": {"hypothesis": "Investigate batch shift"},
                "result": {"summary": "Evidence found", "findings": [{"title": "Batch shift"}]}}
    assert app.exploratory_runs_with_findings([zero]) == []
    # Preserve the original round index when an earlier investigation found nothing.
    assert app.exploratory_runs_with_findings([zero, detected]) == [(2, detected)]
    assert app.exploratory_runs_with_findings([detected, zero]) == [(1, detected)]


def test_zero_finding_rounds_still_export_with_verified_audit(monkeypatch):
    app = _app(monkeypatch)
    monkeypatch.setattr(app, "exploration_cache_key", lambda key: "matching-cache")
    zero = {"proposal": {"hypothesis": "Check blank controls"},
            "result": {"summary": "No abnormal values", "findings": []},
            "verification": {"sandbox_runs": 2}}
    app.st.session_state = {
        "exploration_key": "matching-cache", "exploration_runs": [zero],
        "exploration_stop": {"decision": "stop"}, "exploration_error": None,
    }
    report = app.combined_audit_export({"baseline": {}}, "dataset-key")
    assert report["exploratory_investigations"] == [zero]
    assert report["exploratory_stop_decision"]["decision"] == "stop"


def test_zero_finding_cards_are_filtered_in_render_loop(monkeypatch):
    # Guard the actual renderer against regressions to enumerate(runs) directly.
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "visible_runs = exploratory_runs_with_findings(runs)" in source
    assert "for index, run in visible_runs:" in source


def test_completed_exploration_finalizes_same_checkpoint_without_duplicate_file(tmp_path, monkeypatch):
    app = _app(monkeypatch)
    monkeypatch.setattr(app, "EXPLORATION_CACHE_DIR", tmp_path / "explorations")
    key = "b" * 64
    runs = [{"proposal": {"hypothesis": "Check controls"},
             "result": {"summary": "Nothing new", "findings": []}}]
    checkpoint = app.save_explorations(key, runs)
    final_stop = {"decision": "stop", "reason": "No additional test", "remaining_questions": []}
    updated = app.save_explorations(key, runs, stop=final_stop,
                                    replace_checkpoint=checkpoint)
    assert updated == checkpoint
    assert len(app.saved_explorations(key)) == 1
    assert app.load_explorations(checkpoint, key, include_stop=True) == (runs, final_stop)


def test_checkpoint_finalization_cannot_overwrite_other_cache_file(tmp_path, monkeypatch):
    import pytest
    app = _app(monkeypatch)
    monkeypatch.setattr(app, "EXPLORATION_CACHE_DIR", tmp_path / "explorations")
    current = "a" * 64
    other = "b" * 64
    checkpoint = app.save_explorations(other, [])
    with pytest.raises(ValueError, match="outside"):
        app.save_explorations(current, [], replace_checkpoint=checkpoint)
    assert checkpoint.exists()


def test_zero_finding_rounds_remain_visible_as_completed_checks(monkeypatch):
    app = _app(monkeypatch)
    blank = {
        "proposal": {"hypothesis": "Inspect Rep_Blank, P7_Blank, P6_Blank"},
        "result": {"summary": "Checked 'Blank' control columns for non-zero counts.", "findings": []},
        "verification": {"sandbox_runs": 2},
    }
    novel = {
        "proposal": {"hypothesis": "Investigate unexpected group relationship"},
        "result": {"summary": "Found new evidence", "findings": [{"title": "A candidate issue"}]},
    }
    assert app.exploratory_runs_with_findings([blank, novel]) == [(2, novel)]
    assert app.exploratory_runs_without_findings([blank, novel]) == [(1, blank)]
    assert app.exploratory_runs_without_findings([novel, blank]) == [(2, blank)]
    assert app.exploratory_runs_without_findings([novel]) == []


def test_completed_checks_are_separated_from_new_issue_cards(monkeypatch):
    _app(monkeypatch)
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "checked_runs = exploratory_runs_without_findings(runs)" in source
    assert "for index, run in visible_runs:" in source
    assert "for index, run in checked_runs:" in source
    assert "Completed Gemma checks (no additional findings)" in source
    assert "— no additional issue" in source
    assert "if runs and findings_count == 0:" in source


def test_render_keeps_blank_control_check_without_inventing_a_finding(monkeypatch):
    from contextlib import nullcontext

    app = _app(monkeypatch)
    check = {
        "proposal": {
            "hypothesis": "Blank controls may contain counts",
            "method": "Scan three blank columns",
            "assumptions": "Blank columns are negative controls",
            "code": "print('checked')",
        },
        "result": {
            "summary": "Checked 'Blank' control columns for non-zero counts.",
            "findings": [],
        },
        "verification": {"sandbox_runs": 2},
    }
    calls = []
    app.st.session_state = {
        "exploration_runs": [check], "exploration_stop": None,
        "exploration_error": None,
    }
    monkeypatch.setattr(app, "prepare_exploration_state", lambda _: None)
    for name in ("subheader", "caption", "info", "markdown", "write", "code", "json"):
        monkeypatch.setattr(app.st, name, lambda *args, _name=name, **kwargs: calls.append((_name, args)), raising=False)
    monkeypatch.setattr(app.st, "expander", lambda title, **_: (calls.append(("expander", (title,))) or nullcontext()), raising=False)
    monkeypatch.setattr(app.st, "button", lambda *args, **kwargs: False, raising=False)
    app.render_exploratory_findings([], "dummy-key", {})
    assert any(name == "caption" and "No additional candidate issues" in str(args[0]) for name, args in calls)
    assert any(name == "expander" and "Blank' control columns" in args[0] for name, args in calls)
    assert any(name == "markdown" and "Completed Gemma checks" in args[0] for name, args in calls)
    assert not any(name == "markdown" and "Candidate finding:" in args[0] for name, args in calls)


def test_investigation_status_distinguishes_finished_from_clean(monkeypatch):
    app = _app(monkeypatch)
    bad = {"dataset_baselines": {"dataset_1": {"checks": {
        "find_invalid_counts": {"status": "completed", "observations": [
            {"issue_type": "negative", "row": 16, "column": 6,
             "sample_id": "AbC10", "taxon": "X", "original_value": "-679"}
        ]}
    }}}}
    stop = {"decision": "stop", "reason": "No new hypotheses", "remaining_questions": []}
    review = app.investigation_outcome(
        bad, header_problems=False, gemma_runs=[],
        gemma_stop=stop, gemma_error=None, exploration_enabled=True,
    )
    assert review["state"] == "review"
    assert "issues require review" in review["headline"]
    assert "not been certified clean" in review["details"]

    okay = {"dataset_baselines": {"dataset_1": {"checks": {
        "find_invalid_counts": {"status": "completed", "observations": []}
    }}}}
    neutral = app.investigation_outcome(
        okay, header_problems=False, gemma_runs=[],
        gemma_stop=stop, gemma_error=None, exploration_enabled=True,
    )
    assert neutral["state"] == "limited_clear"
    assert "no issues detected in completed checks" in neutral["headline"]
    assert "does not establish" in neutral["details"]


def test_investigation_status_makes_incomplete_work_explicit(monkeypatch):
    app = _app(monkeypatch)
    data = {"dataset_baselines": {"dataset_1": {"checks": {
        "find_invalid_counts": {"status": "completed", "observations": []}
    }}}}
    paused = app.investigation_outcome(
        data, header_problems=False, gemma_runs=[{"result": {"findings": []}}],
        gemma_stop={"stop_source": "proposal_retry_exhausted", "decision": "stop",
                    "reason": "Repeated proposals"},
        gemma_error=None, exploration_enabled=True,
    )
    assert paused["state"] == "partial"
    assert "could not propose another distinct test" in paused["details"]
    interrupted = app.investigation_outcome(
        data, header_problems=False, gemma_runs=[], gemma_stop=None,
        gemma_error="model timeout", exploration_enabled=True,
    )
    assert interrupted["state"] == "partial"
    assert "interrupted" in interrupted["details"]
    not_run = app.investigation_outcome(
        data, header_problems=False, gemma_runs=[], gemma_stop=None,
        gemma_error=None, exploration_enabled=True,
    )
    assert not_run["state"] == "partial"
    assert "not available" in not_run["details"]


def test_investigation_status_does_not_miss_header_errors(monkeypatch):
    app = _app(monkeypatch)
    audit = {"dataset_baselines": {"dataset_1": {"checks": {
        "find_invalid_counts": {"status": "completed", "observations": []}
    }}}}
    outcome = app.investigation_outcome(
        audit, header_problems=True, gemma_runs=[], gemma_stop=None,
        gemma_error=None, exploration_enabled=False,
    )
    assert outcome["state"] == "review"


def test_manual_retry_button_removed_from_regular_results(monkeypatch):
    _app(monkeypatch)
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'key="continue_unified_exploration"' not in source
    assert "Try another exploratory investigation" not in source
    assert "Investigation run finished — issues require review" in source
    assert "Investigation run finished — no issues detected in completed checks" in source


def test_backend_budget_exhaustion_is_partial_even_without_found_issues(monkeypatch):
    app = _app(monkeypatch)
    audit = {"status": "incomplete_budget_exhausted",
             "dataset_baselines": {"dataset_1": {"checks": {
                 "find_invalid_counts": {"status": "completed", "observations": []}
             }}}}
    outcome = app.investigation_outcome(
        audit, header_problems=False, gemma_runs=[],
        gemma_stop={"decision": "stop", "reason": "No further test"},
        gemma_error=None, exploration_enabled=True,
    )
    assert outcome["state"] == "partial"
    assert "execution budget" in outcome["details"]
