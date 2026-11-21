"""Initial cleaning cards are concise; Agent C chat is conditional on a decision."""

from types import SimpleNamespace

import pytest

from test_cleaning_review_cards import StubUI, _app, _catalog, Context


class TrackingUI(StubUI):
    def __init__(self):
        super().__init__()
        self.form_names = []
        self.visible = []
        self.check_labels = []
        self.checked = set()
        self.chat_keys = []

    def checkbox(self, label, *, key=None, **kwargs):
        self.check_labels.append(label)
        return key in self.checked

    def form(self, name, **kwargs):
        self.form_names.append(name)
        return Context()

    def chat_message(self, role):
        return Context()

    def chat_input(self, label, *, key=None, **kwargs):
        self.chat_keys.append(key)
        return None

    def write(self, *args, **kwargs):
        self.visible.append((args, kwargs))


class SavedAgentB:
    def snapshot(self, key):
        return {"entries": {"dataset_1:qc:1": {
            "status": "ready",
            "suggestion": "Review the experimental protocol before making changes. " * 15,
            "decision_needed": "Determine whether non-integer counts are allowed."}}}


def setup(monkeypatch, tmp_path):
    st = TrackingUI()
    csv = tmp_path / "one.csv"
    csv.write_text("taxon,count\na,0.2\n", encoding="utf-8")
    app = _app(monkeypatch, st)
    monkeypatch.setattr("healthcare_data_quality.cleaning.finding_cleaning.build_cleaning_findings",
                        lambda *args, **kwargs: _catalog()[:1])
    monkeypatch.setattr("healthcare_data_quality.cleaning.finding_cleaning.allowed_positions",
                        lambda *args: (set(), set()))
    monkeypatch.setattr("healthcare_data_quality.jobs.cleaning_jobs.recommendation_manager", lambda: SavedAgentB())
    def render():
        return app.render_cleaning_workspace(
            [csv], [SimpleNamespace(name="one.csv")], {}, "audit-1", []
        )
    return app, st, csv, render


def test_no_chat_or_unverified_yes_before_reviewer_choice(monkeypatch, tmp_path):
    app, st, csv, render = setup(monkeypatch, tmp_path)
    render()
    fid = 'dataset_1:qc:1'
    assert not st.chat_keys
    assert not st.form_names
    assert st.session_state.cleaning_discussion_mode_by_finding == {}
    # A model's prose-only suggestion cannot be treated as an exact edit.
    assert not any("Show full recommendation" == item for item in st.check_labels)
    assert not any(key and key.startswith(f'review_yes_{fid}') for key in st.buttons_on)
    assert app._recommendation_preview("word " * 80).endswith("…")


def test_no_button_opens_enter_to_send_chat_without_mutating_data(monkeypatch, tmp_path):
    app, st, csv, render = setup(monkeypatch, tmp_path)
    before = csv.read_bytes()
    fid = 'dataset_1:qc:1'
    st.buttons_on.add(f"review_no_{fid}_0_choices")
    submissions = []
    monkeypatch.setattr(app, "_queue_agent_c_for_finding",
                        lambda *args: submissions.append(args))
    with pytest.raises(StopIteration, match="rerun"):
        render()
    assert st.session_state.cleaning_discussion_mode_by_finding == {fid: "alternative"}
    assert submissions == []
    st.buttons_on.clear()
    render()
    assert any(key and key.startswith(f"review_message_{fid}_") for key in st.chat_keys)
    assert csv.read_bytes() == before
    assert st.session_state.cleaning_plans_by_finding == {}


def test_prose_only_advice_cannot_start_unverified_approval(monkeypatch, tmp_path):
    app, st, csv, render = setup(monkeypatch, tmp_path)
    before = csv.read_bytes()
    render()
    assert not st.session_state.cleaning_plans_by_finding
    assert not st.session_state.cleaning_history
    assert not st.chat_keys
    assert csv.read_bytes() == before
