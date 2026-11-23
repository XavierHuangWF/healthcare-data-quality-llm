"""No-change reviewer authority, source-pattern suggestions, truthful final audits."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from healthcare_data_quality.cleaning.reviewer_intents import requests_no_change
from healthcare_data_quality.cleaning.final_review_trace import final_review_rows
from healthcare_data_quality.services.source_patterns import summarize_source_patterns
from healthcare_data_quality.services.recommendation_options import source_backed_choices


def _csv(path: Path, header: list[str], data: list[list[str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)
    return path


@pytest.mark.parametrize("message", [
    "keep it as is", "Keep as is.", "leave it unchanged", "keep header blank",
    "keep the header blank", "no changes", "do not rename the header",
    "please leave column 1 header blank",
])
def test_no_change_reviewer_commands_are_terminal_and_unambiguous(message):
    assert requests_no_change(message)


@pytest.mark.parametrize("message", [
    "keep it as is and delete row 3", "keep the first row, delete the second",
    "keep this but replace 2 with 5", "don't keep it as is", "rename it",
])
def test_compound_actions_cannot_be_silently_closed_as_no_change(message):
    assert not requests_no_change(message)


def test_recommendations_compare_full_normal_style_and_offer_keep_blank(tmp_path):
    p = _csv(tmp_path / "source.csv", ["", "Count", "Duplicate"], [
        ["A;B;C;D;E;F;G", "0", "0"],
        ["A;B;C;D;E;F;G", "1", "1"],
        ["A;B;C;D;E", "", "0"],
    ])
    profile = summarize_source_patterns(p)
    assert profile["data_rows_examined"] == 3
    assert profile["column_count"] == 3
    assert profile["empty_header_columns"] == 1
    assert profile["first_column_majority_taxonomy"] is True
    assert profile["normal_taxonomy_separator"] == "semicolon"
    assert profile["count_empty_cells"] == 1
    assert profile["count_zero_cells"] == 3
    assert "source.csv" not in str(profile)
    assert "A;B;C" not in str(profile)  # no raw record content in LLM context
    finding = {"id": "dataset_1:header:1", "check_key": "inspect_headers",
               "dataset_id": "dataset_1", "evidence": [
                   {"Column": 1, "Header name": "(blank)"}]}
    choices = source_backed_choices(finding, p, {"suggestion": "Rename column 1."})
    assert choices[0]["decision"] == "no_change"
    assert choices[0]["rows"][0]["After"] == "(blank — unchanged)"
    assert choices[1]["operations"][0]["changes"] == [{
        "position": 1, "before": "", "after": "Taxon / feature"}]
    assert all("Before" in r and "After" in r for o in choices for r in o["rows"])


def test_uninformative_first_column_never_receives_invented_header(tmp_path):
    p = _csv(tmp_path / "opaque.csv", ["", "Value"], [["001", "1"], ["002", "0"]])
    finding = {"check_key": "inspect_headers", "dataset_id": "dataset_1",
               "evidence": [{"Column": 1, "Header name": "(blank)"}]}
    options = source_backed_choices(finding, p, {"suggestion": "Rename header."})
    assert len(options) == 1 and options[0]["decision"] == "no_change"


def test_completed_edit_superseded_by_approved_deletion_shows_actual_final_state(tmp_path):
    source = _csv(tmp_path / "source.csv", ["", "Count"], [
        ["Taxon;A", "0.2"], ["Taxon;B", "-9"], ["Taxon;C", "7"],
    ])
    original = source.read_bytes()
    staged = original.replace(b"0.2", b"2").replace(b"-9", b"9")
    plans = [{"operation": "replace_cells", "changes": [
        {"row": 2, "column": 2, "before": "0.2", "after": "2"},
        {"row": 3, "column": 2, "before": "-9", "after": "9"},
    ]}]
    approved_rows = [{"operation": "drop_rows", "changes": [
        {"row": 2, "before": "Taxon;A", "after": ""}],
        "source_sha256": hashlib.sha256(original).hexdigest(), "reviewer": "R"}]
    rows = final_review_rows(plans, source, staged, approved_rows, [])
    assert rows[0]["After (final)"] == "Deleted by later approved action"
    assert rows[1]["After (final)"] == "9"
    assert source.read_bytes() == original


def test_header_review_and_deleted_columns_trace_final_positions(tmp_path):
    source = _csv(tmp_path / "s.csv", ["", "Removed", "Retained"], [["A;B;C", "0", "1"]])
    plans = [{"operation": "rename_headers", "changes": [
        {"position": 1, "before": "", "after": "Taxon / feature"}]},
        {"operation": "drop_column", "changes": [
            {"position": 2, "before": "Removed", "after": ""}]}]
    original = source.read_bytes()
    staged = original.replace(b",Removed,Retained", b"Taxon / feature,Removed,Retained")
    columns = [{"position": 2, "before": "Removed", "reviewer": "R",
                "source_sha256": hashlib.sha256(original).hexdigest()}]
    actual = final_review_rows(plans, source, staged, [], columns)
    assert actual[0]["After (final)"] == "Taxon / feature"
    assert actual[1]["After (final)"] == "Deleted column"


def test_review_ui_routes_keep_unchanged_without_llm_and_uses_final_file_audit():
    root = Path(__file__).resolve().parents[1]
    app = (root / "app.py").read_text(encoding="utf-8")
    assert 'if requests_no_change(question):' in app
    assert '"Reviewed — no change"' in app
    assert '"reviewer": reviewer, "user_request": question' in app
    assert 'from healthcare_data_quality.cleaning.final_review_trace import final_review_rows' in app


def test_direct_keep_as_is_finishes_review_without_any_gemma_call(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from test_cleaning_review_cards import StubUI, _app
    from healthcare_data_quality.learning import interaction_memory
    ui = StubUI()
    app = _app(monkeypatch, ui)
    source = _csv(tmp_path / "source.csv", ["", "Count"], [["A;B;C;D;E;F;G", "1"]])
    finding = {"id": "dataset_1:header", "dataset_id": "dataset_1",
               "title": "Column 1 has no header", "check_key": "inspect_headers",
               "evidence": [{"Column": 1, "Header name": "(blank)"}]}
    ui.session_state.cleaning_approved_reviewer = "ReviewerABC"
    ui.session_state.cleaning_recommendations = {}
    ui.session_state.cleaning_discussion_mode_by_finding = {finding["id"]: "alternative"}
    ui.session_state.cleaning_pending_gemma_b = {}
    ui.session_state.cleaning_history = []
    saved = []
    monkeypatch.setattr(interaction_memory, "save_reviewed_solution",
                        lambda *a, **kw: saved.append(kw))
    monkeypatch.setattr(app, "ask_gemma_about_cleaning",
                        lambda *a, **kw: pytest.fail("No-change review called Gemma"))
    app._queue_agent_c_for_finding(finding, source, [SimpleNamespace(name="source.csv")],
                                  [source], "keep it as is")
    assert ui.session_state.cleaning_review_statuses[finding["id"]] == "Reviewed — no change"
    assert ui.session_state.cleaning_decisions[-1]["reviewer"] == "ReviewerABC"
    assert ui.session_state.cleaning_messages_by_finding[finding["id"]][-1]["role"] == "assistant"
    assert saved and saved[0]["decision"] == "reviewed_no_change"
    assert source.read_bytes().startswith(b",Count")
    eligible, finished, count = app._cleaning_export_gate(
        [finding], ui.session_state.cleaning_review_statuses, {}, {}, {}, {},
        investigation_running=False)
    assert eligible and finished == count == 1


def test_earlier_reviewer_keep_message_is_reconciled_without_retyping(monkeypatch, tmp_path):
    from test_cleaning_review_cards import StubUI, _app
    from healthcare_data_quality.learning import interaction_memory
    ui = StubUI()
    app = _app(monkeypatch, ui)
    source = _csv(tmp_path / "s.csv", ["", "Count"], [["A;B", "0"]])
    finding = {"id": "dataset_1:header", "dataset_id": "dataset_1",
               "title": "Column 1 has no header", "check_key": "inspect_headers", "evidence": []}
    fid = finding["id"]
    ui.session_state.cleaning_approved_reviewer = "ReviewerABC"
    ui.session_state.cleaning_review_statuses = {fid: "Discussion open"}
    ui.session_state.cleaning_messages_by_finding = {fid: [
        {"role": "user", "content": "keep it as is"},
        {"role": "assistant", "content": "I will keep the header as is."}]}
    ui.session_state.cleaning_history = []
    monkeypatch.setattr(interaction_memory, "save_reviewed_solution", lambda *a, **k: {})
    assert app._recover_completed_no_change_reply(finding, source)
    assert ui.session_state.cleaning_review_statuses[fid] == "Reviewed — no change"
    assert len(ui.session_state.cleaning_messages_by_finding[fid]) == 2  # no duplicated chat
    assert not app._recover_completed_no_change_reply(finding, source)


def test_blank_taxonomy_header_can_be_recommended_unchanged_without_repair(monkeypatch):
    import json
    from healthcare_data_quality.ui.cleaning_review import ask_gemma_for_review_options
    finding = {"id": "dataset_1:header:1", "source": "QC", "actionable": True,
               "title": "Column 1 has no header", "check_key": "inspect_headers",
               "evidence": [{"Column": 1, "Header name": "(blank)"}],
               "source_pattern_summary": {"first_column_majority_taxonomy": True}}
    monkeypatch.setattr("healthcare_data_quality.ui.cleaning_review.query_local_model",
        lambda *a, **k: json.dumps({"recommendations": [{
            "finding_id": finding["id"], "action_type": "no_change",
            "action": "Keep the intentionally blank taxonomy header unchanged.",
            "rationale": "The column is taxonomic labels and does not require a header to be meaningful."
        }]}))
    assert ask_gemma_for_review_options([finding])[finding["id"]]["suggestion"].startswith("Keep")


def test_agent_b_matches_prior_no_change_using_full_finding_evidence(monkeypatch, tmp_path):
    import time
    from healthcare_data_quality.jobs import cleaning_jobs
    source = _csv(tmp_path / "source.csv", ["", "Count"], [["A;B;C;D;E;F;G", "1"]])
    finding = {"id": "dataset_1:header:1", "dataset_id": "dataset_1", "source": "QC",
               "title": "Blank header", "check_key": "inspect_headers", "evidence": [
                   {"Column": 1, "Header name": "(blank)", "source_only": "exact"}],
               "actionable": True}
    looked_up = []
    def fake_preferred(candidate, path):
        looked_up.append(candidate)
        assert candidate["evidence"][0]["source_only"] == "exact"
        return {"match": "exact", "record": {"decision": "reviewed_no_change"}}
    monkeypatch.setattr(cleaning_jobs, "preferred_reviewed_solution", fake_preferred)
    monkeypatch.setattr(cleaning_jobs, "reviewed_method_context", lambda *a, **kw: [])
    monkeypatch.setattr(cleaning_jobs, "_learned_skills_for_finding", lambda *a: [])
    worker = cleaning_jobs.CleaningRecommendationJobs(tmp_path / "jobs", recommender=lambda payload: {
        "suggestion": "Keep the first taxonomy header blank for this dataset.",
        "decision_needed": "The existing column is taxonomy and the reviewer preferred to preserve its header.",
        "choices": [],
    })
    key = worker.start("test-pattern-mem-key", [finding], run_id="new-test-run",
                       source_paths_by_finding={finding["id"]: source})
    for _ in range(300):
        state = worker.snapshot(key)
        if state["entries"][finding["id"]]["status"] in {"ready", "failed"}:
            break
        time.sleep(.01)
    entry = state["entries"][finding["id"]]
    assert entry["status"] == "ready", entry
    assert entry["exact_prior_no_change"] is True
    assert entry["source_pattern_summary"]["data_rows_examined"] == 1
    assert "source_only" not in repr(state)  # never persist source evidence in job JSON
    assert looked_up


def test_final_export_rejects_header_change_contradicting_nochange_review():
    from healthcare_data_quality.cleaning.final_review_trace import verify_unchanged_headers
    before = [["", "Count"], ["Taxon", "1"]]
    finding = {"id": "f9", "check_key": "inspect_headers",
               "evidence": [{"Column": 1, "Header name": "(blank)"}]}
    status = {"f9": "Reviewed — no change"}
    verify_unchanged_headers(before, before, status, {"f9": finding}, {1: 1, 2: 2})
    changed = [["WrongHeader", "Count"], ["Taxon", "1"]]
    with pytest.raises(PermissionError, match="reviewed to remain blank"):
        verify_unchanged_headers(before, changed, status, {"f9": finding}, {1: 1, 2: 2})
    # If the source column was separately approved for deletion, nothing
    # survives to retain its old header; no stray replacement is inferred.
    verify_unchanged_headers(before, [["Count"], ["1"]], status,
                             {"f9": finding}, {2: 1})
