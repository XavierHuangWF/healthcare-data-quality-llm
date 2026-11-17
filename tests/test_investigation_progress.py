"""Progress is about completed phases, not a fabricated ETA or total LLM rounds."""
import pytest

from investigation_progress import InvestigationProgress, format_elapsed, live_timer_html


def test_three_phases_with_unknown_exploration_length():
    p = InvestigationProgress(include_exploration=True)
    assert p.phase == 1
    assert p.fraction == 0.0
    p.baseline_event("Running automatic baseline checks...")
    assert p.phase == 1
    p.baseline_event("Automatic baseline checks completed.")
    assert p.phase == 2
    assert p.fraction == pytest.approx(1 / 3)
    p.baseline_event("[Agent] Step 2/8: Gemma is selecting a tool")
    assert p.phase == 2
    p.begin_exploration()
    assert p.phase == 3
    assert p.fraction == pytest.approx(2 / 3)
    p.exploration_event("Gemma is planning exploratory investigation 1...")
    p.verified_round(1)
    p.verified_round(2)
    assert p.fraction == pytest.approx(2 / 3)  # not 2/30 or predicted time
    assert "2" in p.round_label
    p.finish()
    assert p.fraction == 1.0


def test_baseline_only_has_two_phases():
    p = InvestigationProgress(include_exploration=False)
    p.baseline_event("Automatic baseline checks completed.")
    assert p.phase == 2
    assert p.fraction == 0.5
    p.finish()
    assert p.fraction == 1.0
    with pytest.raises(ValueError):
        p.begin_exploration()


def test_failed_work_not_reported_as_complete():
    p = InvestigationProgress(include_exploration=True)
    p.baseline_event("Automatic baseline checks completed.")
    p.begin_exploration()
    p.verified_round(3)
    with pytest.raises(ValueError):
        p.verified_round(2)
    assert not p.finished
    assert p.fraction < 1


def test_live_timer_has_no_sensitive_data_or_network_dependencies():
    html = live_timer_html(1794690123123)
    assert "window.setInterval(updateElapsed, 1000)" in html
    assert "1794690123123" in html
    assert "http://" not in html and "https://" not in html
    assert format_elapsed(9) == "00:09"
    assert format_elapsed(62) == "01:02"
    assert format_elapsed(3671) == "01:01:11"
    with pytest.raises(ValueError):
        live_timer_html("1794690123123")
