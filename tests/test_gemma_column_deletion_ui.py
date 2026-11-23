"""A model-issued deletion is a reviewable column action, not an unchecked edit."""
from types import SimpleNamespace

import pytest

from test_cleaning_review_cards import StubUI, _app


class UI(StubUI):
    submitted = False
    def warning(self, *args, **kwargs):
        self.comments.extend(args)
    def text_input(self, *args, **kwargs):
        return "researcher"
    def form_submit_button(self, label, **kwargs):
        return self.submitted and label.startswith("Yes — Delete column 2")
    def dataframe(self, *args, **kwargs):
        pass


class AgentB:
    def snapshot(self, _):
        return {"entries": {"dataset_1:qc:1": {
            "status": "ready", "suggestion": "Replace the fractional count",
            "decision_needed": "Review measurement type"}}}


def test_deletion_proposal_requires_click_and_stays_export_only(monkeypatch, tmp_path):
    source = tmp_path / "sample.csv"
    source.write_text("taxon,A,B\nr1,0.5,9\nr2,1,8\n", encoding="utf-8")
    original = source.read_bytes()
    finding = {"id": "dataset_1:qc:1", "dataset_id": "dataset_1", "name": "sample.csv",
               "source": "QC", "title": "Fractional count",
               "check_key": "find_invalid_counts", "method": "QC", "actionable": True,
               "evidence": [{"CSV row": 2, "Column": 2}]}
    st = UI()
    app = _app(monkeypatch, st)
    monkeypatch.setattr("healthcare_data_quality.cleaning.finding_cleaning.build_cleaning_findings",
                        lambda *args, **kwargs: [finding])
    monkeypatch.setattr("healthcare_data_quality.jobs.cleaning_jobs.recommendation_manager", lambda: AgentB())

    def render():
        app.render_cleaning_workspace(
            [source], [SimpleNamespace(name="sample.csv")], {}, "a1", [])

    render()  # initializes session and Agent B; no changes
    st.session_state.cleaning_approved_reviewer = "researcher"
    from healthcare_data_quality.cleaning.finding_cleaning import validate_scoped_plan
    st.session_state.cleaning_plans_by_finding[finding["id"]] = validate_scoped_plan(
        {"operation": "drop_column", "dataset_id": "dataset_1",
         "changes": [{"position": 2, "before": "A", "after": ""}]}, finding, source)
    render()  # human has not approved yet
    assert st.session_state.cleaning_history == []
    assert st.session_state.cleaning_approved_columns_by_dataset == {}
    assert source.read_bytes() == original
    st.submitted = True
    with pytest.raises(StopIteration, match="rerun"):
        render()
    assert len(st.session_state.cleaning_history) == 1
    assert st.session_state.cleaning_history[0]["operation"] == "drop_column"
    assert st.session_state.cleaning_staged_bytes_by_dataset == {}
    assert source.read_bytes() == original
    from healthcare_data_quality.cleaning.in_memory_cleaning import apply_approved_column_deletions, _records
    final = apply_approved_column_deletions(
        original, original, st.session_state.cleaning_approved_columns_by_dataset["dataset_1"])
    assert _records(final) == [["taxon", "B"], ["r1", "9"], ["r2", "8"]]
