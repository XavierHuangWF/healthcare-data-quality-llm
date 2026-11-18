"""Agent A runs only during a new investigation; cleaning UI is read-only for its advice."""
from types import SimpleNamespace

from test_cleaning_review_cards import StubUI, _app, _catalog


class FakeManager:
    def __init__(self):
        self.start_calls = []
        self.entries = {}

    def start(self, context_key, catalog, **kwargs):
        self.start_calls.append((context_key, [x["id"] for x in catalog], kwargs))
        for item in catalog:
            self.entries.setdefault(item["id"], {"status": "pending"})
        if catalog:
            self.entries[catalog[0]["id"]] = {
                "status": "ready", "suggestion": "Check whether counts are integers",
                "decision_needed": "Were values normalized?"}
        return "fake-job-id"

    def snapshot(self, key):
        return {"entries": self.entries} if self.entries else None


def _setup(monkeypatch, tmp_path, st):
    csv = tmp_path / "one.csv"
    csv.write_text("Taxon,Count\na,1.5\n", encoding="utf-8")
    app = _app(monkeypatch, st)
    monkeypatch.setattr("healthcare_data_quality.finding_cleaning.build_cleaning_findings",
                        lambda *args, **kwargs: _catalog())
    monkeypatch.setattr("healthcare_data_quality.finding_cleaning.allowed_positions",
                        lambda *args: (set(), set()))
    manager = FakeManager()
    monkeypatch.setattr("cleaning_jobs.recommendation_manager", lambda: manager)
    return app, csv, manager


def test_agent_b_queues_during_investigation_before_agent_c_review(monkeypatch, tmp_path):
    st = StubUI()
    st.session_state.audit_source = "new"
    app, csv, manager = _setup(monkeypatch, tmp_path, st)
    before = csv.read_bytes()
    audit = {"cleaning_recommendation_run_id": "once-only-run"}
    app.schedule_agent_b_once("audit-new", audit, [csv], ["one.csv"], [])
    assert len(manager.start_calls) == 1
    assert manager.start_calls[0][2]["run_id"] == "once-only-run"

    # Opening/rerendering cleaning tab never starts a model recommendation.
    app.render_cleaning_workspace([csv], [SimpleNamespace(name="one.csv")], audit,
                                  "audit-new", [])
    app.render_cleaning_workspace([csv], [SimpleNamespace(name="one.csv")], audit,
                                  "audit-new", [])
    assert len(manager.start_calls) == 1
    assert st.session_state.cleaning_recommendations["dataset_1:qc:1"]["_source"] == "background"
    assert "dataset_1:qc:2" not in st.session_state.cleaning_recommendations
    assert csv.read_bytes() == before


def test_saved_investigation_never_offers_regeneration(monkeypatch, tmp_path):
    st = StubUI()
    st.session_state.audit_source = "saved"
    app, csv, manager = _setup(monkeypatch, tmp_path, st)
    manager.entries = {"dataset_1:qc:1": {
        "status": "ready", "suggestion": "Already generated",
        "decision_needed": "Confirm scientific policy"}}
    seen_buttons = []
    original_button = st.button

    def capture_button(label, *, key=None, **kwargs):
        seen_buttons.append((label, key))
        return original_button(label, key=key, **kwargs)

    st.button = capture_button
    audit = {"cleaning_recommendation_run_id": "once-only-run"}
    app.render_cleaning_workspace([csv], [SimpleNamespace(name="one.csv")], audit,
                                  "audit-saved", [])
    assert not manager.start_calls
    assert st.session_state.cleaning_recommendations["dataset_1:qc:1"]["suggestion"] == "Already generated"
    assert all("refresh" not in label.lower() and "retry" not in label.lower()
               and "suggestion" not in label.lower()
               for label, key in seen_buttons)
    assert all(key not in {"cleaning_suggest_all", "review_suggest_dataset_1:qc:1"}
               for label, key in seen_buttons)


def test_agent_c_discussion_does_not_replace_once_only_agent_b_suggestion(monkeypatch, tmp_path):
    st = StubUI()
    app, csv, manager = _setup(monkeypatch, tmp_path, st)
    audit = {"cleaning_recommendation_run_id": "once-only-run"}
    manager.entries = {"dataset_1:qc:1": {
        "status": "ready", "suggestion": "Original Agent A recommendation",
        "decision_needed": "Which dataset policy?"}}
    app.render_cleaning_workspace([csv], [SimpleNamespace(name="one.csv")], audit,
                                  "audit-chat", [])
    monkeypatch.setattr(app, "ask_gemma_about_cleaning", lambda *args, **kwargs: {
        "reply": "Please clarify your scientific policy", "operation": "none",
        "dataset_id": "", "changes": []})
    st.session_state.cleaning_messages_by_finding.setdefault("dataset_1:qc:1", [])
    app._ask_gemma_for_finding(_catalog()[0], csv, [SimpleNamespace(name="one.csv")],
                               [csv], "I disagree with the suggestion")
    assert st.session_state.cleaning_recommendations["dataset_1:qc:1"]["suggestion"] == "Original Agent A recommendation"
    assert len(st.session_state.cleaning_messages_by_finding["dataset_1:qc:1"]) == 2


def test_new_investigation_id_resets_review_context(monkeypatch, tmp_path):
    st = StubUI()
    app, csv, manager = _setup(monkeypatch, tmp_path, st)
    first = {"cleaning_recommendation_run_id": "first-run"}
    second = {"cleaning_recommendation_run_id": "second-run"}
    app.render_cleaning_workspace([csv], [SimpleNamespace(name="one.csv")], first, "same-audit-key", [])
    st.session_state.cleaning_recommendations["dataset_1:qc:1"] = {"suggestion": "old run"}
    app.render_cleaning_workspace([csv], [SimpleNamespace(name="one.csv")], second, "same-audit-key", [])
    assert st.session_state.cleaning_recommendations == {}


def test_legacy_audit_reads_old_queue_without_starting_new_job(monkeypatch, tmp_path):
    st = StubUI()
    app, csv, manager = _setup(monkeypatch, tmp_path, st)
    manager.entries = {"dataset_1:qc:1": {
        "status": "ready", "suggestion": "Saved under previous scheduler",
        "decision_needed": "Original policy?"}}
    app.render_cleaning_workspace([csv], [SimpleNamespace(name="one.csv")], {},
                                  "legacy-audit", [])
    assert not manager.start_calls
    assert st.session_state.cleaning_recommendations["dataset_1:qc:1"]["suggestion"] == "Saved under previous scheduler"
