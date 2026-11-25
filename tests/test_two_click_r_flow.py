"""Fixtures and tests for one acceptance, automatic R preview, and final insertion approval."""
from types import SimpleNamespace

import pytest

from test_cleaning_review_cards import StubUI, _app
from healthcare_data_quality.cleaning.finding_cleaning import validate_scoped_plan


FINDING = {
    "id": "dataset_1:qc:1", "dataset_id": "dataset_1", "name": "one.csv",
    "source": "QC", "title": "1 fractional count", "check_key": "find_invalid_counts",
    "method": "Inspect counts", "evidence": [{"CSV row": 2, "Column": 2}],
    "actionable": True,
}


class ReviewUI(StubUI):
    def __init__(self):
        super().__init__()
        self.r_code = []
        self.pressed = set()
        self.displayed_tables = 0
        self.errors = []
        self.reviewer = "local_researcher"
        self.show_code = False

    def dataframe(self, *args, **kwargs):
        self.displayed_tables += 1

    def code(self, source, **kwargs):
        self.r_code.append(source)

    def checkbox(self, label, **kwargs):
        if label == 'Review R cleaning and insertion code':
            return self.show_code
        return False

    def text_input(self, *args, **kwargs):
        return self.reviewer

    def form_submit_button(self, label, **kwargs):
        if kwargs.get("disabled", False):
            return False
        return (label in self.pressed or
                str(kwargs.get("key", "")).startswith("review_approve_yes_") and
                "Approve & Stage in memory" in self.pressed)

    def error(self, message):
        self.errors.append(message)


def setup_review(monkeypatch, tmp_path):
    from healthcare_data_quality.jobs import cleaning_jobs
    csv_path = tmp_path / "one.csv"
    csv_path.write_text("Taxon,Count\na,0.2\n", encoding="utf-8")
    st = ReviewUI()
    monkeypatch.setenv("HDQ_APPROVED_STAGE_DIR", str(tmp_path / "checkpoint"))
    app = _app(monkeypatch, st)
    monkeypatch.setattr("healthcare_data_quality.cleaning.finding_cleaning.build_cleaning_findings",
                        lambda *a, **kw: [FINDING])

    class Saved:
        def snapshot(self, *args):
            return {"entries": {FINDING["id"]: {"status": "ready",
                "suggestion": "Replace fractional count at row 2 with NA.",
                "decision_needed": "No raw count supports this decimal."}}}
    monkeypatch.setattr(cleaning_jobs, "recommendation_manager", lambda: Saved())
    uploaded = [SimpleNamespace(name="one.csv")]
    audit = {"cleaning_recommendation_run_id": "test-short-flow"}

    def render():
        app.render_cleaning_workspace([csv_path], uploaded, audit, "test-context", [])

    render()  # initialize session state
    candidate = {"operation": "replace_cells", "dataset_id": "dataset_1",
                 "changes": [{"row": 2, "column": 2,
                              "before": "0.2", "after": "NA"}]}
    st.session_state.cleaning_plans_by_finding[FINDING["id"]] = validate_scoped_plan(
        candidate, FINDING, csv_path)
    st.session_state.cleaning_discussion_mode_by_finding[FINDING["id"]] = "accepted"
    st.session_state.cleaning_approved_reviewer = st.reviewer
    return app, st, csv_path, render


def test_full_r_code_available_for_review_without_extra_approval(monkeypatch, tmp_path):
    import healthcare_data_quality.r_runtime.scoped_r_execution as scoped
    app, st, source, render = setup_review(monkeypatch, tmp_path)
    monkeypatch.setattr(scoped, 'run_r_sandbox', lambda *args, **kwargs:
                        'kind,csv_row,csv_column,value\ncell,2,2,NA\n')
    st.show_code = True
    render()
    fid = FINDING['id']
    preview = st.session_state.cleaning_r_previews_by_finding[fid]
    assert preview['r_result']['status'] == 'r_execution_verified'
    assert preview['insertion_plan']['changes'][0]['after'] == 'NA'
    # The new UI intentionally hides R source code, while retaining verified
    # reproducible R artifacts in the completed review state.
    assert st.r_code == []
    assert 'clean_dataset <- function(df)' in preview['r_function']
    assert 'df[[col]][rows[i]] <- verified_changes$replacement[i]' in preview['insertion_r_function']
    assert st.session_state.cleaning_working_paths == {}
    assert not st.errors


def test_preview_failure_does_not_run_insertion(monkeypatch, tmp_path):
    import healthcare_data_quality.r_runtime.scoped_r_execution as scoped
    app, st, source, render = setup_review(monkeypatch, tmp_path)
    monkeypatch.setattr(scoped, 'run_r_sandbox', lambda *args, **kwargs:
                        'kind,csv_row,csv_column,value\ncell,2,2,UNEXPECTED\n')
    render()
    assert st.session_state.cleaning_result is None
    assert st.session_state.cleaning_working_paths == {}
    assert 'Isolated R preview failed' in st.session_state.cleaning_r_errors_by_finding[FINDING['id']]
