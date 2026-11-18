from pathlib import Path
"""The researcher's submitted text is visible during both in-flight jobs."""

import threading
import time
from types import SimpleNamespace

from interactive_chat_jobs import InteractiveChatJobs
from test_cleaning_review_cards import StubUI, _app, _catalog, Context


class ChatStub(StubUI):
    def __init__(self):
        super().__init__()
        self.sent_buttons = set()
        self.submitted_text = {}
        self.visible = []
        self.current_role = None

    def chat_message(self, role):
        self.current_role = role
        return Context()

    def write(self, *args, **kwargs):
        if self.current_role and args:
            self.visible.append((self.current_role, str(args[0])))

    def text_area(self, label, *, key=None, **kwargs):
        return self.submitted_text.get(key, "")

    def form_submit_button(self, label, *, disabled=False, **kwargs):
        if disabled:
            return False
        if label in self.sent_buttons:
            self.sent_buttons.remove(label)
            return True
        return False


def _one_card(monkeypatch, tmp_path):
    st = ChatStub()
    app = _app(monkeypatch, st)
    f = tmp_path / "one.csv"
    f.write_text("Taxon,Count\na,1.5\n", encoding="utf-8")
    one = _catalog()[0]
    monkeypatch.setattr("healthcare_data_quality.finding_cleaning.build_cleaning_findings",
                        lambda *args, **kwargs: [one])
    monkeypatch.setattr("healthcare_data_quality.finding_cleaning.allowed_positions",
                        lambda *args: (set(), set()))
    audit = {"cleaning_recommendation_run_id": "same-inquiry"}
    def render():
        app.render_cleaning_workspace([f], [SimpleNamespace(name="one.csv")], audit,
                                      "audit-A", [])
    render()
    st.session_state.cleaning_recommendations[one["id"]] = {
        "suggestion": "Check the provenance before changing fractional counts",
        "decision_needed": "Were counts normalized?", "_source": "background"}
    return app, st, one, f, render


def test_agent_c_keeps_question_visible_while_responding(monkeypatch, tmp_path):
    import interactive_chat_jobs
    manager = InteractiveChatJobs()
    monkeypatch.setattr(interactive_chat_jobs, "interactive_chat_manager", lambda: manager)
    app, st, one, f, render = _one_card(monkeypatch, tmp_path)
    event_start = threading.Event()
    release = threading.Event()

    class SavedB:
        def snapshot(self, key):
            return {"entries": {one["id"]: {"status": "ready",
                "suggestion": "Check the provenance before changing fractional counts",
                "decision_needed": "Were these normalized?"}}}
    monkeypatch.setattr('cleaning_jobs.recommendation_manager', lambda: SavedB())

    def slow_c(*args, **kwargs):
        event_start.set()
        assert release.wait(4)
        return {"reply": "Confirm whether the source is normalized.",
                "operation": "none", "dataset_id": "", "changes": []}

    monkeypatch.setattr(app, "ask_gemma_about_cleaning", slow_c)
    fid = one["id"]
    # The user already chose "No — discuss another approach".
    st.session_state.cleaning_discussion_mode_by_finding[fid] = "alternative"
    st.submitted_text[f"review_message_{fid}"] = "Can we keep 0.2 if it is normalized?"
    st.sent_buttons.add("Send to Agent C")
    import pytest
    with pytest.raises(StopIteration, match="rerun"):
        render()
    assert event_start.wait(2)
    assert st.session_state.cleaning_messages_by_finding[fid][0] == {
        "role": "user", "content": "Can we keep 0.2 if it is normalized?"}
    st.current_role = None
    render()
    assert ("user", "Can we keep 0.2 if it is normalized?") in st.visible
    assert fid in st.session_state.cleaning_pending_gemma_b
    assert not hasattr(app, "_agent_a_readonly_answer")
    release.set()
    for _ in range(100):
        render()
        if fid not in st.session_state.cleaning_pending_gemma_b:
            break
        time.sleep(0.01)
    assert fid not in st.session_state.cleaning_pending_gemma_b
    assert [m["role"] for m in st.session_state.cleaning_messages_by_finding[fid]] == ["user", "assistant"]
    assert st.session_state.cleaning_recommendations[fid]["suggestion"] == (
        "Check the provenance before changing fractional counts")
    assert st.session_state.cleaning_plans_by_finding == {}


def test_worker_preserves_independent_completion_and_single_consumption():
    manager = InteractiveChatJobs()
    release = threading.Event()
    started = threading.Event()
    def slow():
        started.set()
        release.wait(2)
        return {"reply": "done"}
    a = manager.submit(slow)
    assert started.wait(1)
    b = manager.submit(lambda: "read-only reply")
    assert manager.take(a)["status"] == "running"
    for _ in range(100):
        result = manager.take(b)
        if result["status"] != "running":
            break
        time.sleep(0.01)
    assert result == {"status": "done", "result": "read-only reply"}
    assert manager.take(b)["status"] == "lost"
    release.set()
    for _ in range(100):
        result = manager.take(a)
        if result["status"] != "running":
            break
        time.sleep(0.01)
    assert result == {"status": "done", "result": {"reply": "done"}}


def test_only_agent_c_handles_human_conversation(monkeypatch):
    st = ChatStub()
    app = _app(monkeypatch, st)
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    assert "_agent_a_readonly_answer" not in source
    assert "_queue_agent_a_clarification" not in source
    assert "Discuss another action with Agent C" in source
    assert "Agent B — proposed cleaning action (not approved)" in source


def test_agent_c_redundant_confirmation_continues_automatically(monkeypatch, tmp_path):
    import interactive_chat_jobs
    manager = InteractiveChatJobs()
    monkeypatch.setattr(interactive_chat_jobs, "interactive_chat_manager", lambda: manager)
    app, st, one, f, render = _one_card(monkeypatch, tmp_path)
    fid = one["id"]

    class SavedB:
        def snapshot(self, key):
            return {"entries": {fid: {"status": "ready",
                "suggestion": "Replace the affected invalid cells with NA in a reviewed copy.",
                "decision_needed": "Confirm the exact replacement value and positions."}}}
    monkeypatch.setattr('cleaning_jobs.recommendation_manager', lambda: SavedB())

    st.session_state.cleaning_discussion_mode_by_finding[fid] = "accepted"
    st.session_state.cleaning_messages_by_finding[fid] = [
        {"role": "assistant", "content":
         "To proceed with generating the structured changes, I need you to confirm the following: "
         "Is the replacement value 'NA' exact, and do you authorize the specific replacements "
         "for the affected positions? Once you provide explicit authorization, I will provide "
         "the structured change objects."}
    ]

    seen = []
    def fake_submit(*args, **kwargs):
        seen.append((args, kwargs))
        return "job-123"
    monkeypatch.setattr(manager, 'submit', fake_submit)

    import pytest
    with pytest.raises(StopIteration, match="rerun"):
        render()

    assert st.session_state.cleaning_review_statuses[fid] == "Agent C replying"
    assert st.session_state.cleaning_messages_by_finding[fid][-1]["automatic"] is True
    assert "does NOT authorize" in st.session_state.cleaning_messages_by_finding[fid][-1]["content"]
    assert st.session_state.cleaning_pending_gemma_b[fid]["job_id"] == "job-123"
    assert seen, "Agent C should continue automatically with no extra approval click"
