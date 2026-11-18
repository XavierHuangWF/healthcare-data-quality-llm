"""Cumulative, idempotent, source-verified staging and one-session reviewer."""
import csv
from io import StringIO
from types import SimpleNamespace

import pytest

from healthcare_data_quality.in_memory_cleaning import (
    apply_verified_in_memory, assess_verified_in_memory, review_completion, sha256_bytes,
)
from healthcare_data_quality.scoped_r_execution import verify_r_against_current_staged_values
from test_two_click_r_flow import FINDING, ReviewUI, setup_review


def plan(original, changes, *, finding_id="finding-b", operation="replace_cells"):
    return {"operation": operation, "dataset_id": "dataset_1", "finding_id": finding_id,
            "source_sha256": sha256_bytes(original), "changes": changes}


def test_identical_prior_cleaning_is_idempotent_no_new_edit_or_csv(tmp_path):
    original = b"taxon,a,b\nx,0.2,0.5\n"
    first = plan(original, [{"row": 2, "column": 2, "before": "0.2", "after": "NA"}])
    second = plan(original, [
        {"row": 2, "column": 2, "before": "0.2", "after": "NA"},
        {"row": 2, "column": 3, "before": "0.5", "after": "NA"},
    ])
    staged1, _ = apply_verified_in_memory(original, original, first, reviewer="x")
    assessment = assess_verified_in_memory(original, staged1, second)
    assert len(assessment["already_resolved"]) == 1
    assert len(assessment["ready_to_apply"]) == 1
    staged2, record2 = apply_verified_in_memory(original, staged1, second, reviewer="x")
    assert record2["changes_applied"] == 1
    assert record2["already_resolved"] == 1
    assert list(csv.reader(StringIO(staged2.decode()))) == [
        ["taxon", "a", "b"], ["x", "NA", "NA"]]
    staged3, record3 = apply_verified_in_memory(original, staged2, second, reviewer="x")
    assert staged3 == staged2
    assert record3["status"] == "already_resolved_in_memory"
    assert record3["changes_applied"] == 0
    assert record3["already_resolved"] == 2
    assert original == b"taxon,a,b\nx,0.2,0.5\n"
    assert not list(tmp_path.iterdir())
    assert review_completion([{"id": "b"}], {"b": "Resolved by earlier approved edit"}) == (1, 1)


def test_conflicting_prior_edit_requires_bound_current_value_r_proof(monkeypatch):
    original = b"taxon,a\nx,0.2\n"
    first = plan(original, [{"row": 2, "column": 2, "before": "0.2", "after": "NA"}])
    later = plan(original, [{"row": 2, "column": 2, "before": "0.2", "after": "0"}])
    stage, _ = apply_verified_in_memory(original, original, first, reviewer="researcher")
    review = assess_verified_in_memory(original, stage, later)
    assert review["conflicts"] == [{
        "csv_row": 2, "column": 2, "original": "0.2", "current": "NA",
        "proposed": "0", "status": "conflict"}]
    with pytest.raises(PermissionError, match="CSV row 2, column 2"):
        apply_verified_in_memory(original, stage, later, reviewer="researcher")
    import healthcare_data_quality.scoped_r_execution as scoped
    calls = []
    def fake_r(script, input_csv, *, preserve_text):
        calls.append((script, input_csv, preserve_text))
        return "kind,csv_row,csv_column,value\ncell,2,2,0\n"
    monkeypatch.setattr(scoped, "run_r_sandbox", fake_r)
    proof = verify_r_against_current_staged_values(original, stage, later)
    assert len(calls) == 1
    # Current values are isolated input data, never interpolated into R source.
    assert '"NA"' not in calls[0][0]
    assert 'df$value <- df$replacement' in calls[0][0]
    assert ['cell', '2', '2', 'NA', '0'] in list(csv.reader(StringIO(calls[0][1])))
    assert 'cell,2,2,NA' in calls[0][1]
    assert calls[0][2] is True
    result, record = apply_verified_in_memory(original, stage, later,
                                               reviewer="researcher", approved_rebase=proof)
    assert list(csv.reader(StringIO(result.decode())))[1][1] == "0"
    assert record["conflicts_overridden"] == 1
    assert original == b"taxon,a\nx,0.2\n"
    with pytest.raises(PermissionError, match="earlier approved"):
        apply_verified_in_memory(original, stage, later,
            reviewer="researcher", approved_rebase={**proof, "staged_sha256": "stale"})
    with pytest.raises(PermissionError, match="earlier approved"):
        apply_verified_in_memory(original, stage, later,
            reviewer="researcher", approved_rebase={**proof, "verified_r_output_sha256": "fake"})


def test_unauthorized_rebased_r_value_is_rejected(monkeypatch):
    original = b"a,b\nx,0.2\n"
    staged, _ = apply_verified_in_memory(original, original,
        plan(original, [{"row": 2, "column": 2, "before": "0.2", "after": "NA"}]), reviewer="a")
    later = plan(original, [{"row": 2, "column": 2, "before": "0.2", "after": "0"}])
    import healthcare_data_quality.scoped_r_execution as scoped
    monkeypatch.setattr(scoped, "run_r_sandbox", lambda *args, **kwargs:
                        "kind,csv_row,csv_column,value\ncell,2,2,99\n")
    with pytest.raises(PermissionError, match="unexpected coordinate/value"):
        verify_r_against_current_staged_values(original, staged, later)
    assert b"NA" in staged


def test_review_id_entered_once_for_all_approvals(monkeypatch, tmp_path):
    from test_r_auto_preview_insertion_flow import _mock_r
    app, ui, path, render = setup_review(monkeypatch, tmp_path)
    calls = []
    def text_input(*args, **kwargs):
        calls.append(args[0]); return "researcher-once"
    ui.text_input = text_input
    _mock_r(monkeypatch)
    render()
    first_approval_label = "Reviewer identifier (required once for this investigation)"
    assert calls == [first_approval_label]  # shown inside first approval form only
    ui.pressed.add("Approve & Stage in memory")
    with pytest.raises(StopIteration, match="rerun"):
        render()
    assert ui.session_state.cleaning_approved_reviewer == "researcher-once"
    assert ui.session_state.cleaning_history[0]["reviewer"] == "researcher-once"
    ui.pressed.clear()
    render()
    assert calls == [first_approval_label] * 2  # no further reviewer field
    assert ui.session_state.cleaning_review_statuses[FINDING["id"]] == "Applied and verified"


def test_ui_overlapping_conflict_gets_r_rechecked_and_explicitly_approved(monkeypatch, tmp_path):
    from healthcare_data_quality.finding_cleaning import validate_scoped_plan
    from healthcare_data_quality.in_memory_cleaning import apply_verified_in_memory
    import healthcare_data_quality.scoped_r_execution as scoped

    app, ui, source, render = setup_review(monkeypatch, tmp_path)
    ui.warning = lambda *args, **kwargs: None
    ui.success = lambda *args, **kwargs: None
    original = source.read_bytes()
    prior = plan(original, [{"row": 2, "column": 2, "before": "0.2", "after": "NA"}])
    staged, _ = apply_verified_in_memory(original, original, prior, reviewer="one-reviewer")
    ui.session_state.cleaning_staged_bytes_by_dataset["dataset_1"] = staged
    ui.session_state.cleaning_approved_reviewer = "one-reviewer"
    proposed = {"operation": "replace_cells", "dataset_id": "dataset_1",
                "changes": [{"row": 2, "column": 2, "before": "0.2", "after": "0"}]}
    ui.session_state.cleaning_plans_by_finding[FINDING["id"]] = (
        validate_scoped_plan(proposed, FINDING, source))
    r_calls = []
    def fake_r(script, input_csv, *, preserve_text):
        r_calls.append(input_csv)
        return "kind,csv_row,csv_column,value\ncell,2,2,0\n"
    monkeypatch.setattr(scoped, "run_r_sandbox", fake_r)
    checkboxes = []
    def approve_conflict(label, **kwargs):
        checkboxes.append(label)
        return label.startswith("I authorize replacing these previously approved values")
    ui.checkbox = approve_conflict
    render()
    assert len(r_calls) == 2  # original-input verification + staged-input recheck
    assert any("previously approved values" in label for label in checkboxes)
    assert b"a,NA" in ui.session_state.cleaning_staged_bytes_by_dataset["dataset_1"]
    ui.pressed.add("Approve & Stage in memory")
    with pytest.raises(StopIteration, match="rerun"):
        render()
    assert len(r_calls) == 2  # cached R proof is reused on approval
    assert b"a,0" in ui.session_state.cleaning_staged_bytes_by_dataset["dataset_1"]
    assert original == source.read_bytes()
    history = ui.session_state.cleaning_history
    assert history[0]["conflicts_overridden"] == 1
    assert history[0]["reviewer"] == "one-reviewer"
    assert ui.session_state.cleaning_review_statuses[FINDING["id"]] == "Applied and verified"


def test_ui_resolved_overlap_stages_no_additional_change(monkeypatch, tmp_path):
    from healthcare_data_quality.in_memory_cleaning import apply_verified_in_memory
    from test_r_auto_preview_insertion_flow import _mock_r

    app, ui, source, render = setup_review(monkeypatch, tmp_path)
    ui.warning = lambda *args, **kwargs: None
    ui.success = lambda *args, **kwargs: None
    _mock_r(monkeypatch)
    original = source.read_bytes()
    same = plan(original, [{"row": 2, "column": 2, "before": "0.2", "after": "NA"}])
    staged, _ = apply_verified_in_memory(original, original, same, reviewer="one-reviewer")
    ui.session_state.cleaning_staged_bytes_by_dataset["dataset_1"] = staged
    render()
    ui.pressed.add("Approve & Stage in memory")
    with pytest.raises(StopIteration, match="rerun"):
        render()
    assert ui.session_state.cleaning_staged_bytes_by_dataset["dataset_1"] == staged
    assert ui.session_state.cleaning_history[0]["changes"] == 0
    assert ui.session_state.cleaning_review_statuses[FINDING["id"]] == (
        "Resolved by earlier approved edit")


def test_agent_c_sees_latest_staged_values_and_validates_a_staged_before(monkeypatch, tmp_path):
    app, ui, source, render = setup_review(monkeypatch, tmp_path)
    original = source.read_bytes()
    prior = plan(original, [{"row": 2, "column": 2, "before": "0.2", "after": "NA"}])
    staged, _ = apply_verified_in_memory(original, original, prior, reviewer="same-reviewer")
    ui.session_state.cleaning_staged_bytes_by_dataset["dataset_1"] = staged
    fid, thread, old_thread, scoped, metadata = app._prepare_finding_discussion(
        FINDING, source, [SimpleNamespace(name=source.name)], [source], "What now?")
    assert scoped["original_source_cell_examples"] == [
        {"row": 2, "column": 2, "value": "0.2"}]
    assert scoped["current_source_cell_examples"] == [
        {"row": 2, "column": 2, "value": "NA"}]
    assert scoped["latest_working_version"] is True
    assert scoped["latest_staged_sha256"] == sha256_bytes(staged)
    response = {"reply": "Try 0 instead.", "operation": "replace_cells",
                "dataset_id": "dataset_1", "changes": [{"row": 2, "column": 2,
                      "before": "NA", "after": "0"}]}
    app._record_gemma_b_response(fid, FINDING, source, response)
    pending_plan = ui.session_state.cleaning_plans_by_finding[fid]
    assert pending_plan["changes"] == [{"row": 2, "column": 2,
                                         "before": "0.2", "after": "0"}]
    assert source.read_bytes() == original


def test_rebased_r_csv_quoting_does_not_invalidate_verified_values(monkeypatch):
    original = b"a,b\nx,0.2\n"
    staged, _ = apply_verified_in_memory(original, original,
        plan(original, [{"row": 2, "column": 2, "before": "0.2", "after": "NA"}]), reviewer="r")
    later = plan(original, [{"row": 2, "column": 2, "before": "0.2", "after": "0"}])
    import healthcare_data_quality.scoped_r_execution as scoped
    monkeypatch.setattr(scoped, "run_r_sandbox", lambda *args, **kwargs:
        '"kind","csv_row","csv_column","value"\n"cell","2","2","0"\n')
    proof = verify_r_against_current_staged_values(original, staged, later)
    assert proof["raw_r_output_sha256"] != proof["verified_r_output_sha256"]
    new, result = apply_verified_in_memory(original, staged, later,
                                            reviewer="r", approved_rebase=proof)
    assert list(csv.reader(StringIO(new.decode())))[1][1] == "0"
