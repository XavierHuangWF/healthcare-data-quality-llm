"""Regression for new-run isolation and Agent A -> B -> C handoff.

These tests never invoke a model or alter original datasets.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from healthcare_data_quality.jobs.cleaning_jobs import CleaningRecommendationJobs, job_fingerprint
from test_cleaning_review_cards import StubUI, _app, _catalog


class AuditUI(StubUI):
    def warning(self, *args, **kwargs):
        pass


def _one_finding(monkeypatch):
    finding = _catalog()[0]
    monkeypatch.setattr(
        "healthcare_data_quality.cleaning.finding_cleaning.build_cleaning_findings",
        lambda *args, **kwargs: [finding],
    )
    return finding


def _source(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("Taxon,Count\nA,1.5\n", encoding="utf-8")
    return path


def test_agent_b_is_scheduled_for_each_fresh_run_not_blocked_by_invalid_keyword(
    monkeypatch, tmp_path,
):
    st = AuditUI()
    app = _app(monkeypatch, st)
    path = _source(tmp_path)
    finding = _one_finding(monkeypatch)
    calls = []

    def fake_recommend(item):
        calls.append(item["id"])
        return {"suggestion": "Check units and normalize this value", "decision_needed": "Confirm protocol"}

    worker = CleaningRecommendationJobs(tmp_path / "jobs", recommender=fake_recommend)
    monkeypatch.setattr("healthcare_data_quality.jobs.cleaning_jobs.recommendation_manager", lambda: worker)

    for run_id in ("fresh-first", "fresh-second"):
        assert app.schedule_agent_b_once("same-csv-key", {"cleaning_recommendation_run_id": run_id},
                                         [path], [path.name], [])
        key = job_fingerprint("same-csv-key", [finding], run_id=run_id)
        for _ in range(200):
            state = worker.snapshot(key)
            if state and state["entries"][finding["id"]]["status"] != "pending" and state["entries"][finding["id"]]["status"] != "running":
                break
            time.sleep(.01)
        assert state["entries"][finding["id"]]["status"] == "ready"
    assert calls == [finding["id"], finding["id"]]


def test_new_review_does_not_look_up_or_replay_old_decisions(monkeypatch, tmp_path):
    st = AuditUI()
    app = _app(monkeypatch, st)
    path = _source(tmp_path)
    _one_finding(monkeypatch)
    monkeypatch.setenv("HDQ_APPROVED_STAGE_DIR", str(tmp_path / "no-existing-checkpoints"))

    def unexpected(*args, **kwargs):
        raise AssertionError("Historical review must not be read automatically")

    monkeypatch.setattr("healthcare_data_quality.learning.interaction_memory.find_reviewed_solution", unexpected)
    class EmptyQueue:
        def snapshot(self, key):
            return None
    monkeypatch.setattr("healthcare_data_quality.jobs.cleaning_jobs.recommendation_manager", lambda: EmptyQueue())

    st.session_state.cleaning_fresh_review_run_id = "new-run"
    app.render_cleaning_workspace([path], [SimpleNamespace(name=path.name)],
                                  {"cleaning_recommendation_run_id": "new-run"}, "same-csv-key", [])
    assert st.session_state.cleaning_recovery_dismissed is True
    assert st.session_state.cleaning_recommendations == {}
    assert st.session_state.cleaning_review_statuses == {}
    assert st.session_state.cleaning_discussion_mode_by_finding == {}


def test_direct_agent_c_discussion_is_available_before_agent_b(monkeypatch, tmp_path):
    st = AuditUI()
    app = _app(monkeypatch, st)
    path = _source(tmp_path)
    finding = _one_finding(monkeypatch)
    monkeypatch.setenv("HDQ_APPROVED_STAGE_DIR", str(tmp_path / "no-existing-checkpoints"))

    class NoRecommendation:
        def snapshot(self, key):
            return None
    monkeypatch.setattr("healthcare_data_quality.jobs.cleaning_jobs.recommendation_manager", lambda: NoRecommendation())
    st.buttons_on.add(f"review_start_direct_agent_c_{finding['id']}")
    audit = {"cleaning_recommendation_run_id": "new-run"}
    with pytest.raises(StopIteration, match="streamlit rerun"):
        app.render_cleaning_workspace([path], [SimpleNamespace(name=path.name)],
                                      audit, "same-csv-key", [])
    assert st.session_state.cleaning_discussion_mode_by_finding[finding["id"]] == "alternative"
    assert st.session_state.cleaning_review_statuses[finding["id"]] == "Alternative requested"
    assert st.session_state.cleaning_staged_bytes_by_dataset == {}


def test_fresh_run_context_isolated_from_previous_session(monkeypatch, tmp_path):
    st = AuditUI()
    app = _app(monkeypatch, st)
    path = _source(tmp_path)
    finding = _one_finding(monkeypatch)
    monkeypatch.setenv("HDQ_APPROVED_STAGE_DIR", str(tmp_path / "no-existing-checkpoints"))

    class EmptyQueue:
        def snapshot(self, key):
            return None
    monkeypatch.setattr("healthcare_data_quality.jobs.cleaning_jobs.recommendation_manager", lambda: EmptyQueue())

    a1 = {"cleaning_recommendation_run_id": "old-run"}
    a2 = {"cleaning_recommendation_run_id": "new-run"}
    app.render_cleaning_workspace([path], [SimpleNamespace(name=path.name)], a1, "same-csv-key", [])
    st.session_state.cleaning_recommendations[finding["id"]] = {
        "suggestion": "outdated suggestion", "_source": "manual"
    }
    st.session_state.cleaning_review_statuses[finding["id"]] = "Reviewed solution available"
    st.session_state.cleaning_fresh_review_run_id = "new-run"
    app.render_cleaning_workspace([path], [SimpleNamespace(name=path.name)], a2, "same-csv-key", [])
    assert st.session_state.cleaning_recommendations == {}
    assert st.session_state.cleaning_review_statuses == {}
    assert st.session_state.cleaning_recovery_dismissed is True


def test_existing_run_can_start_missing_agent_b_job_without_rerunning_agent_a(
    monkeypatch, tmp_path,
):
    st = AuditUI()
    app = _app(monkeypatch, st)
    path = _source(tmp_path)
    finding = _one_finding(monkeypatch)
    monkeypatch.setenv("HDQ_APPROVED_STAGE_DIR", str(tmp_path / "no-checkpoints"))
    calls = []

    class LostJob:
        def snapshot(self, key):
            return None

        def start(self, context_key, findings, **kwargs):
            calls.append((context_key, [item["id"] for item in findings], kwargs))
            return "restored-run-job"

    monkeypatch.setattr("healthcare_data_quality.jobs.cleaning_jobs.recommendation_manager", lambda: LostJob())
    run_id = "already-finished-run"
    st.buttons_on.add(f"cleaning_start_missing_agent_b_same-csv-key:{run_id}")
    with pytest.raises(StopIteration, match="streamlit rerun"):
        app.render_cleaning_workspace(
            [path], [SimpleNamespace(name=path.name)],
            {"cleaning_recommendation_run_id": run_id}, "same-csv-key", [],
        )
    assert len(calls) == 1
    context, finding_ids, options = calls[0]
    assert context == "same-csv-key" and finding_ids == [finding["id"]]
    assert options["run_id"] == run_id
    assert st.session_state.cleaning_history == []
    assert st.session_state.cleaning_staged_bytes_by_dataset == {}


def test_direct_agent_c_submission_does_not_require_agent_b_result(monkeypatch, tmp_path):
    st = AuditUI()
    app = _app(monkeypatch, st)
    path = _source(tmp_path)
    finding = _one_finding(monkeypatch)
    monkeypatch.setenv("HDQ_APPROVED_STAGE_DIR", str(tmp_path / "no-checkpoints"))

    class MissingB:
        def snapshot(self, key):
            return None
    monkeypatch.setattr("healthcare_data_quality.jobs.cleaning_jobs.recommendation_manager", lambda: MissingB())
    app.render_cleaning_workspace(
        [path], [SimpleNamespace(name=path.name)],
        {"cleaning_recommendation_run_id": "new-run"}, "same-csv-key", [],
    )
    assert st.session_state.cleaning_recommendations == {}
    fid = finding["id"]
    st.session_state.cleaning_discussion_mode_by_finding[fid] = "alternative"

    submissions = []
    class FakeChatManager:
        def submit_routed(self, callback, *args, **kwargs):
            submissions.append((args, kwargs))
            return "direct-chat-job", "C"
    monkeypatch.setattr("healthcare_data_quality.jobs.interactive_chat_jobs.interactive_chat_manager", lambda: FakeChatManager())
    monkeypatch.setattr(app, "_background_agents_idle_for_run", lambda *args: (False, False))

    app._queue_agent_c_for_finding(
        finding, path, [SimpleNamespace(name=path.name)], [path],
        "Explain whether the fractional count can safely be changed.",
    )
    assert submissions
    assert st.session_state.cleaning_pending_gemma_b[fid]["job_id"] == "direct-chat-job"
    assert st.session_state.cleaning_recommendations == {}
    assert st.session_state.cleaning_staged_bytes_by_dataset == {}
