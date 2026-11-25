"""Past approved methods must influence queued suggestions visibly but never authorize them."""
import json
from pathlib import Path

from healthcare_data_quality.jobs import cleaning_jobs
from healthcare_data_quality.ui.cleaning_review import ask_gemma_for_review_options
from healthcare_data_quality.ui.recommendation_memory_trace import memory_trace


def _finding(fid="dataset_1:qc:1"):
    return {"id": fid, "dataset_id": "dataset_1", "source": "QC",
            "title": "2 negative counts", "check_key": "find_invalid_counts",
            "actionable": True, "evidence": [{"CSV row": 4, "Column": 3, "Original value": "-12"}]}


def test_model_assesses_saved_method_and_reports_advisory_basis(monkeypatch):
    requests = []
    finding = {**_finding(), "reviewed_methods": [{
        "previous_issue": "earlier negative counts", "method": "absolute-value repair",
        "operation_types": ["replace_cells"], "same_original_csv": True}],
        "learned_skills": [{"recommended_method": "correct counts only with approval"}]}

    def fake(prompt, **kwargs):
        requests.append(prompt)
        return json.dumps({"recommendations": [{
            "finding_id": finding["id"], "action_type": "scoped_replacement",
            "action": "Replace negative counts with verified absolute counts.",
            "rationale": "The approved example uses the same source and a matching count type.",
            "memory_assessment": {"decision": "adapted", "reason": "Changed only the evidenced cell coordinates"},
        }]})

    monkeypatch.setattr("healthcare_data_quality.ui.cleaning_review.query_local_model", fake)
    result = ask_gemma_for_review_options([finding])[finding["id"]]
    assert result["memory_assessment"]["decision"] == "adapted"
    assert "previously_approved_methods" in requests[0]
    assert "absolute-value repair" in requests[0]
    assert "learned_skills" in requests[0]


def test_no_memory_never_falsely_claims_reuse(monkeypatch):
    monkeypatch.setattr("healthcare_data_quality.ui.cleaning_review.query_local_model", lambda *a, **k: json.dumps({
        "recommendations": [{"finding_id": _finding()["id"],
                             "action_type": "set_missing", "action": "Replace affected count values with NA.",
                             "rationale": "Invalid observed values cannot be treated as nonnegative counts.",
                             "memory_assessment": {"decision": "reused", "reason": "I remember this"}}]}))
    result = ask_gemma_for_review_options([_finding()])[_finding()["id"]]
    assert result["memory_assessment"]["decision"] == "not_applicable"


def test_queued_finding_reloads_newly_approved_memory_without_exposing_source(tmp_path, monkeypatch):
    file = tmp_path / "source_sensitive.csv"
    file.write_text("taxon,value\nprivate,1\n", encoding="utf-8")
    state = {"approved": False, "seen": []}

    def read_current(finding, source, *, limit):
        assert source == file
        return ([{"previous_issue": "PRIVATE ISSUE", "method": "PRIVATE VALUES",
                  "operation_types": ["replace_cells"],
                  "same_original_csv": True, "issue_relation": "related"}]
                if state["approved"] else [])

    monkeypatch.setattr(cleaning_jobs, "reviewed_method_context", read_current)
    monkeypatch.setattr(cleaning_jobs, "_learned_skills_for_finding", lambda item: [])

    def fake(item):
        state["seen"].append(len(item["reviewed_methods"]))
        if len(state["seen"]) == 1:
            state["approved"] = True
        return {"suggestion": "Replace current negatives after source verification",
                "decision_needed": "Only named approval can execute", "choices": [],
                "memory_assessment": {"decision": "adapted", "reason": "Past review checked"}}

    findings = [_finding("dataset_1:qc:1"), _finding("dataset_1:qc:2")]
    worker = cleaning_jobs.CleaningRecommendationJobs(tmp_path / "queues", recommender=fake)
    key = worker.start("investigation", findings, run_id="new-run",
                       source_paths_by_finding={x["id"]: file for x in findings})
    worker._workers[key].join(3)
    assert state["seen"] == [0, 1]
    snapshot = worker.snapshot(key)
    assert snapshot["entries"][findings[1]["id"]]["method_candidates_considered"] == 1
    assert snapshot["entries"][findings[1]["id"]]["memory_context"][0]["operation_types"] == ["replace_cells"]
    persisted = (tmp_path / "queues" / (key + ".json")).read_text()
    assert "PRIVATE ISSUE" not in persisted and "PRIVATE VALUES" not in persisted
    assert str(file) not in persisted and "private,1" not in persisted


def test_memory_ui_discloses_actual_consultation_without_claiming_verification():
    trace = memory_trace({"method_candidates_considered": 2, "skill_candidates_considered": 1,
                          "memory_assessment": {"decision": "adapted", "reason": "Different target rows"},
                          "memory_context": [{"operation_types": ["replace_cells"],
                                              "same_original_csv": True}]})
    assert "2 approved" in trace["summary"]
    assert "1 learned" in trace["summary"]
    assert any("Gemma reports adapting" in x for x in trace["details"])
    assert all("automatically approved" not in x for x in trace["details"])
    missing = memory_trace({"method_candidates_considered": 0, "skill_candidates_considered": 0})
    assert "no relevant" in missing["summary"]
    prior = memory_trace({"suggestion": "historical cache"})
    assert "unavailable" in prior["summary"]


def test_agent_b_background_completion_does_not_force_full_page_rerun():
    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    portion = app_source.split("def _render_live_agent_b_finding(", 1)[1].split("def _is_blank_control_check(", 1)[0]
    assert "st.rerun()" not in portion.replace("# Do NOT st.rerun()", "")
    assert "_expand_ready_finding_without_rerun(fid)" not in portion
    assert 'st.fragment(run_every="3s")(refresh_cleaning)' in app_source
