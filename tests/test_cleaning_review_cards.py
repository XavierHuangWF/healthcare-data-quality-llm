"""Regression coverage for review-all UI and non-binding Gemma suggestions."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from healthcare_data_quality.ui.cleaning_review import ask_gemma_for_review_options, indexed_review_catalog, review_question


ROOT = Path(__file__).resolve().parents[1]


class Context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class Session(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key, value):
        self[key] = value


class StubUI:
    # Streamlit st.columns returns container objects. Newer approval controls
    # use ``with col:``; model that real context-manager contract here.
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __init__(self):
        self.session_state = Session()
        self.expanders = []
        self.buttons_on = set()
        self.comments = []

    def subheader(self, *args, **kwargs):
        pass

    def caption(self, *args, **kwargs):
        pass

    def markdown(self, *args, **kwargs):
        pass

    def write(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        if args:
            self.comments.append(str(args[0]))

    def checkbox(self, *args, **kwargs):
        return False

    def expander(self, label, **kwargs):
        self.expanders.append(label)
        return Context()

    def button(self, label, *, key=None, **kwargs):
        return key in self.buttons_on

    def columns(self, count):
        return [self] * count

    def form(self, *args, **kwargs):
        return Context()

    def text_area(self, *args, **kwargs):
        return ""

    def form_submit_button(self, *args, **kwargs):
        return False

    def spinner(self, *args, **kwargs):
        return Context()

    def status(self, *args, **kwargs):
        return Context()

    def dataframe(self, data, *args, **kwargs):
        self.comments.append(("table", data))

    def container(self, *args, **kwargs):
        return Context()

    def error(self, text):
        self.comments.append(text)

    def rerun(self):
        raise StopIteration("streamlit rerun")


def _app(monkeypatch, st):
    monkeypatch.setitem(sys.modules, "streamlit", ModuleType("streamlit"))
    spec = importlib.util.spec_from_file_location("hdq_cleaning_review_app", ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "st", st)
    return module


def _catalog():
    return [
        {"id": "dataset_1:qc:1", "dataset_id": "dataset_1", "name": "one.csv",
         "source": "QC", "title": "Fractional counts", "check_key": "find_invalid_counts",
         "method": "Inspect counts", "evidence": [], "actionable": True},
        {"id": "dataset_1:qc:2", "dataset_id": "dataset_1", "name": "one.csv",
         "source": "QC", "title": "Missing counts", "check_key": "find_invalid_counts",
         "method": "Inspect counts", "evidence": [], "actionable": True},
        {"id": "gemma:1:check", "dataset_id": "", "name": "Gemma",
         "source": "Gemma check", "title": "Checked controls: no issue",
         "check_key": "gemma_check", "method": "Sandbox execution", "evidence": [],
         "actionable": False},
    ]


def test_catalog_is_numbered_like_findings_page():
    catalog = _catalog() + [
        {"id": "dataset_2:qc:1", "dataset_id": "dataset_2", "source": "QC"},
        {"id": "gemma:2:finding:1", "source": "Gemma finding"},
    ]
    assert [label for label, _ in indexed_review_catalog(catalog)] == [
        "Finding 1", "Finding 2", "Gemma check 1", "Finding 1", "Gemma finding 1",
    ]
    assert "no data change" in review_question(_catalog()[-1])


def test_batch_recommendations_are_non_executable_and_match_ids(monkeypatch):
    calls = []

    def fake_query(prompt, **kwargs):
        calls.append((prompt, kwargs))
        items = json.loads(prompt.split("DATA:\n", 1)[1])
        return json.dumps({"recommendations": [
            {"finding_id": item["finding_id"],
             "action_type": "no_change" if item["type"] == "Gemma check" else "set_missing",
             "action": "Keep the dataset unchanged for this completed check." if item["type"] == "Gemma check"
                       else "Replace the affected invalid cells with NA in a reviewed CSV copy.",
             "rationale": "No verified data error requires a correction."}
            for item in items
        ] + [{"finding_id": "unknown", "action_type": "set_missing",
              "action": "Replace invalid observations with NA in a separate reviewed copy.",
              "rationale": "No evidence linked to a known finding."}]})

    monkeypatch.setattr("healthcare_data_quality.ui.cleaning_review.query_local_model", fake_query)
    catalog = _catalog()
    replies = ask_gemma_for_review_options(catalog)
    assert set(replies) == {item["id"] for item in catalog}
    assert "unknown" not in replies
    assert replies[catalog[0]["id"]]["suggestion"].startswith("Replace the affected invalid cells")
    assert "researcher" not in replies[catalog[0]["id"]]["suggestion"].lower()
    assert len(calls) == 1
    assert "no code" not in calls[0][0].lower() or "do not generate code" in calls[0][0].lower()
    assert calls[0][1]["json_schema"]


def test_all_findings_and_gemma_checks_visible_without_selection(monkeypatch, tmp_path):
    file = tmp_path / "one.csv"
    file.write_text("Taxon,Count\na,1\n", encoding="utf-8")
    st = StubUI()
    app = _app(monkeypatch, st)
    monkeypatch.setattr("healthcare_data_quality.cleaning.finding_cleaning.build_cleaning_findings",
                        lambda *args, **kwargs: _catalog())
    monkeypatch.setattr("healthcare_data_quality.cleaning.finding_cleaning.allowed_positions",
                        lambda *args: (set(), set()))
    app.render_cleaning_workspace([file], [SimpleNamespace(name="one.csv")], {}, "audit-A", [])
    # Review headings are visible as locked wait rows, not expandable panels.
    assert len(st.expanders) == 0
    assert st.session_state.cleaning_plans_by_finding == {}


def test_one_finding_no_action_does_not_affect_other_cards(monkeypatch, tmp_path):
    file = tmp_path / "one.csv"
    file.write_text("Taxon,Count\na,1\n", encoding="utf-8")
    st = StubUI()
    app = _app(monkeypatch, st)
    monkeypatch.setattr("healthcare_data_quality.cleaning.finding_cleaning.build_cleaning_findings",
                        lambda *args, **kwargs: _catalog())
    monkeypatch.setattr("healthcare_data_quality.cleaning.finding_cleaning.allowed_positions",
                        lambda *args: (set(), set()))
    st.session_state.hdq_registered_reviewer = "ReviewerABC"
    # Only the ready finding can be marked reviewed; queued cards are locked.
    class ReadyAgentB:
        def snapshot(self, key):
            return {"entries": {"dataset_1:qc:2": {
                "status": "ready", "suggestion": "Review evidence"}}}
    monkeypatch.setattr("healthcare_data_quality.jobs.cleaning_jobs.recommendation_manager",
                        lambda: ReadyAgentB())
    st.buttons_on.add("review_no_action_dataset_1:qc:2")
    import pytest
    with pytest.raises(StopIteration, match="rerun"):
        app.render_cleaning_workspace([file], [SimpleNamespace(name="one.csv")], {}, "audit-B", [])
    assert st.session_state.cleaning_review_statuses == {
        "dataset_1:qc:2": "Reviewed — no change"}
    assert len(st.session_state.cleaning_decisions) == 1
    assert st.session_state.cleaning_history == []


def test_accepting_source_backed_suggestion_executes_with_one_click(monkeypatch, tmp_path):
    """The first Yes is the only named approval of the visible exact change."""
    import pytest
    file = tmp_path / "one.csv"
    file.write_text("Taxon,Count\na,0.2\n", encoding="utf-8")
    original = file.read_bytes()
    monkeypatch.setenv("HDQ_APPROVED_STAGE_DIR", str(tmp_path / "checkpoint"))
    st = StubUI()
    app = _app(monkeypatch, st)
    finding = {**_catalog()[0], "evidence": [
        {"CSV row": 2, "Column": 2, "Original value": "0.2"}]}
    monkeypatch.setattr("healthcare_data_quality.cleaning.finding_cleaning.build_cleaning_findings",
                        lambda *args, **kwargs: [finding])
    monkeypatch.setattr("healthcare_data_quality.cleaning.finding_cleaning.allowed_positions",
                        lambda *args: ({(2, 2)}, set()))

    from healthcare_data_quality.services.recommendation_options import source_backed_choices
    from healthcare_data_quality.ui.finding_choice_integrity import scoped_review_choices
    from healthcare_data_quality.services.model_plan_review import option_fingerprint
    suggestion = {"status": "ready", "suggestion": "Replace invalid count 0.2 with 0",
                  "decision_needed": "", "choices": [{"label": "Replace with zero", "operations": [{
                    "operation": "replace_cells", "changes": [{"row": 2, "column": 2, "before": "0.2", "after": "0"}]}]}]}
    choices = scoped_review_choices(finding, file, source_backed_choices(finding, file, suggestion))
    suggestion['gemma_option_reviews'] = [{'fingerprint': option_fingerprint(finding,file,c), 'can_offer': True, 'reason':'Exact scope reviewed'} for c in choices]
    class SavedAgentB:
        def snapshot(self, key):
            return {"entries": {finding["id"]: suggestion}}

    monkeypatch.setattr("healthcare_data_quality.jobs.cleaning_jobs.recommendation_manager",
                        lambda: SavedAgentB())
    uploaded = [SimpleNamespace(name="one.csv")]
    app.render_cleaning_workspace([file], uploaded, {}, "audit-C", [])
    assert any(item[0] == "table" for item in st.comments)
    st.session_state.cleaning_approved_reviewer = 'Reviewer X'
    executed = []
    def queue_once(selected_finding, path, choice):
        executed.append(choice['operations'])
        st.session_state.cleaning_pending_gemma_b[selected_finding['id']] = {'kind':'local_choice'}
        return True
    monkeypatch.setattr(app, '_queue_local_choice', queue_once)
    st.buttons_on.add(f"review_yes_{finding['id']}_0_option_{option_fingerprint(finding, file, choices[0])[:20]}")
    with pytest.raises(StopIteration, match="rerun"):
        app.render_cleaning_workspace([file], uploaded, {}, "audit-C", [])
    assert len(executed) == 1
    assert file.read_bytes() == original
    assert st.session_state.cleaning_history == []
    assert st.session_state.cleaning_pending_gemma_b[finding['id']]['kind'] == 'local_choice'
    assert st.session_state.cleaning_plans_by_finding == {}


def test_existing_open_session_migrates_to_new_card_states(monkeypatch, tmp_path):
    file = tmp_path / "one.csv"
    file.write_text("Taxon,Count\na,1\n", encoding="utf-8")
    st = StubUI()
    st.session_state.cleaning_context_key = "old-session"
    st.session_state.cleaning_working_paths = {"dataset_1": str(file)}
    st.session_state.cleaning_history = [{"finding_id": "old", "dataset_id": "dataset_1"}]
    st.session_state.cleaning_messages_by_finding = {"dataset_1:qc:1": []}
    app = _app(monkeypatch, st)
    monkeypatch.setattr("healthcare_data_quality.cleaning.finding_cleaning.build_cleaning_findings",
                        lambda *args, **kwargs: _catalog())
    monkeypatch.setattr("healthcare_data_quality.cleaning.finding_cleaning.allowed_positions",
                        lambda *args: (set(), set()))
    app.render_cleaning_workspace([file], [SimpleNamespace(name="one.csv")], {}, "old-session", [])
    assert st.session_state.cleaning_plans_by_finding == {}
    assert st.session_state.cleaning_recommendations == {}
    assert st.session_state.cleaning_history[0]["finding_id"] == "old"
    assert len(st.expanders) == 0  # queued cards are visible but cannot expand
    assert st.session_state.cleaning_working_paths == {}
    assert st.session_state.cleaning_staged_bytes_by_dataset["dataset_1"] == file.read_bytes()
