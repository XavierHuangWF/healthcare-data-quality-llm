"""Cleaning widget actions should refresh their fragment, not the whole page.

A full rerun rebuilds the three-tab app and can jump the browser viewport away
from the finding in which a researcher is typing or approving changes.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from test_cleaning_review_cards import StubUI, _app, _catalog


def test_rerun_cleaning_fragment_uses_fragment_scope(monkeypatch):
    st = StubUI()
    app = _app(monkeypatch, st)
    requested = []

    def scoped_rerun(*, scope=None):
        requested.append(scope)
        raise StopIteration("rerun")

    st.rerun = scoped_rerun
    with pytest.raises(StopIteration, match="rerun"):
        app._rerun_cleaning_fragment()
    assert requested == ["fragment"]


def test_cleaning_review_button_does_not_force_full_app_rerun(monkeypatch, tmp_path):
    st = StubUI()
    app = _app(monkeypatch, st)
    file = tmp_path / "one.csv"
    file.write_text("Taxon,Count\na,1\n", encoding="utf-8")
    monkeypatch.setattr("healthcare_data_quality.cleaning.finding_cleaning.build_cleaning_findings",
                        lambda *args, **kwargs: _catalog())
    monkeypatch.setattr("healthcare_data_quality.cleaning.finding_cleaning.allowed_positions",
                        lambda *args: (set(), set()))
    calls = []

    def scoped_rerun(*, scope=None):
        calls.append(scope)
        raise StopIteration("rerun")

    st.rerun = scoped_rerun
    st.session_state.hdq_registered_reviewer = "ReviewerABC"
    # This action is available only when the matching recommendation is ready.
    class ReadyAgentB:
        def snapshot(self, key):
            return {"entries": {"dataset_1:qc:1": {"status": "ready", "suggestion": "Review counts"}}}
    monkeypatch.setattr("healthcare_data_quality.jobs.cleaning_jobs.recommendation_manager",
                        lambda: ReadyAgentB())
    st.buttons_on.add("review_no_action_dataset_1:qc:1")
    with pytest.raises(StopIteration, match="rerun"):
        app.render_cleaning_workspace([file], [SimpleNamespace(name="one.csv")],
                                      {}, "audit-scroll", [])
    assert calls == ["fragment"]
    assert st.session_state.cleaning_review_statuses["dataset_1:qc:1"] == "Reviewed — no change"


def test_fallback_for_legacy_streamlit(monkeypatch):
    st = StubUI()
    app = _app(monkeypatch, st)
    calls = []

    def old_rerun():
        calls.append("full")
        raise StopIteration("rerun")

    st.rerun = old_rerun
    with pytest.raises(StopIteration, match="rerun"):
        app._rerun_cleaning_fragment()
    assert calls == ["full"]
