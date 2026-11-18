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
        self.info_messages = []
        self.submitted_text = {}
        self.send_chat = False

    def info(self, text):
        self.info_messages.append(text)

    def text_area(self, label, *, key=None, **kwargs):
        return self.submitted_text.get(key, "")

    def form_submit_button(self, label, *, disabled=False, **kwargs):
        return label == "Send to Agent C" and self.send_chat and not disabled

    def checkbox(self, label, *, key=None, **kwargs):
        self.check_labels.append(label)
        return key in self.checked

    def form(self, name, **kwargs):
        self.form_names.append(name)
        return Context()

    def chat_message(self, role):
        return Context()

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
    monkeypatch.setattr("healthcare_data_quality.finding_cleaning.build_cleaning_findings",
                        lambda *args, **kwargs: _catalog()[:1])
    monkeypatch.setattr("healthcare_data_quality.finding_cleaning.allowed_positions",
                        lambda *args: (set(), set()))
    monkeypatch.setattr("cleaning_jobs.recommendation_manager", lambda: SavedAgentB())
    def render():
        return app.render_cleaning_workspace(
            [csv], [SimpleNamespace(name="one.csv")], {}, "audit-1", []
        )
    return app, st, csv, render


def test_no_chat_before_choice_and_full_recommendation_available(monkeypatch, tmp_path):
    app, st, csv, render = setup(monkeypatch, tmp_path)
    render()
    assert not st.form_names
    assert not st.session_state.cleaning_discussion_mode_by_finding
    assert "Show full recommendation" in st.check_labels
    assert app._recommendation_preview("word " * 80).endswith("…")
    st.checked.add("review_full_advice_dataset_1:qc:1")
    render()
    assert any("Review the experimental protocol" in str(args[0])
               and len(str(args[0])) > 220 for args, _ in st.visible if args)


def test_no_button_opens_chat_without_mutating_data(monkeypatch, tmp_path):
    app, st, csv, render = setup(monkeypatch, tmp_path)
    before = csv.read_bytes()
    st.buttons_on.add("review_no_dataset_1:qc:1")
    submissions = []
    monkeypatch.setattr(app, "_queue_agent_c_for_finding",
                        lambda *args: submissions.append(args))
    with pytest.raises(StopIteration, match="rerun"):
        render()
    assert st.session_state.cleaning_discussion_mode_by_finding == {
        "dataset_1:qc:1": "alternative"
    }
    assert submissions == [], "No must not submit an automatic Gemma request"
    fid = "dataset_1:qc:1"
    assert st.session_state.cleaning_messages_by_finding.get(fid, []) == []
    assert fid not in st.session_state.cleaning_pending_gemma_b
    st.buttons_on.clear()
    render()
    assert f"review_chat_form_{fid}" in st.form_names
    assert any("won't receive a request until" in msg for msg in st.info_messages)
    assert csv.read_bytes() == before
    assert st.session_state.cleaning_plans_by_finding == {}


def test_no_then_custom_message_is_first_and_only_agent_c_submission(monkeypatch, tmp_path):
    app, st, csv, render = setup(monkeypatch, tmp_path)
    fid = "dataset_1:qc:1"
    requests = []
    monkeypatch.setattr(app, "_queue_agent_c_for_finding",
                        lambda *args, **kwargs: requests.append(args[4]))
    st.buttons_on.add(f"review_no_{fid}")
    with pytest.raises(StopIteration, match="rerun"):
        render()
    assert requests == []
    st.buttons_on.clear()
    st.submitted_text[f"review_message_{fid}"] = "Keep column 7; clear row 4 column 11."
    st.send_chat = True
    with pytest.raises(StopIteration, match="rerun"):
        render()
    assert requests == ["Keep column 7; clear row 4 column 11."]


def test_yes_hides_chat_until_agent_c_asks_for_clarification(monkeypatch, tmp_path):
    app, st, csv, render = setup(monkeypatch, tmp_path)
    st.buttons_on.add("review_yes_dataset_1:qc:1")
    monkeypatch.setattr(app, "_queue_agent_c_for_finding", lambda *args, **kwargs: None)
    with pytest.raises(StopIteration, match="rerun"):
        render()
    st.buttons_on.clear()
    render()
    assert st.session_state.cleaning_discussion_mode_by_finding["dataset_1:qc:1"] == "accepted"
    assert not st.form_names
    # Agent C asks for the scientific policy instead of returning exact changes.
    st.session_state.cleaning_messages_by_finding["dataset_1:qc:1"] = [
        {"role": "assistant", "content": "Which measurement policy applies?"}
    ]
    # On the first prose-only response, Agent C automatically retries once.
    with pytest.raises(StopIteration, match="rerun"):
        render()
    assert "dataset_1:qc:1" in st.session_state.cleaning_plan_retry_once
    render()
    assert "review_chat_form_dataset_1:qc:1" in st.form_names
    assert not st.session_state.cleaning_plans_by_finding
