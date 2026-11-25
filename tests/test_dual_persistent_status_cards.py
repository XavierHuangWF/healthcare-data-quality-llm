"""Once-only recommendation progress and finished investigation UI cards."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from healthcare_data_quality.ui.exploration_banner import (
    recommendation_progress_counts,
    render_recommendation_banner_html,
    render_finished_investigation_banner_html,
    render_exploration_banner_html,
    recommendation_mascot_data_url,
    sticky_banner_css,
)


def test_agent_a_cards_show_actual_saved_statuses():
    state = {"entries": {
        "f1": {"status": "ready"},
        "f2": {"status": "ready"},
        "f3": {"status": "running"},
        "f4": {"status": "pending"},
        "f5": {"status": "failed"},
    }}
    assert recommendation_progress_counts(state) == {
        "total": 5, "ready": 2, "running": 1, "pending": 1, "failed": 1,
    }
    html = render_recommendation_banner_html(state)
    assert "2 of 5 suggestions ready" in html
    assert "1 generating" in html
    assert "1 queued" in html
    assert "1 failed" in html
    assert "40.00%" in html
    assert "aria-valuenow=\"2\"" in html
    assert "role=\"progressbar\"" in html


def test_finished_cards_persist_without_faking_live_timer_or_completeness():
    for state in ("partial", "review", "limited_clear"):
        html = render_finished_investigation_banner_html(state=state, verified_checks=3)
        assert "Run ended" in html
        assert "Verified exploratory checks: <b>3</b>" in html
        assert "hdq-live-clock" not in html
        assert "window.setInterval(render, 1000)" not in html
        assert "not a certification" in html
    assert "partial results" in render_finished_investigation_banner_html(state="partial", verified_checks=0)


def test_agent_a_completed_and_failed_state_persist():
    ready = render_recommendation_banner_html({"entries": {"a": {"status": "ready"}, "b": {"status": "ready"}}})
    assert "recommendations ready" in ready
    assert "2 of 2 suggestions ready" in ready
    assert "100.00%" in ready
    incomplete = render_recommendation_banner_html({"entries": {"a": {"status": "ready"}, "b": {"status": "failed"}}})
    assert "finished with errors" in incomplete
    assert "1 of 2 suggestions ready" in incomplete
    unscheduled = render_recommendation_banner_html(None)
    assert "not scheduled" in unscheduled
    assert "0 of 0 suggestions ready" in unscheduled


def test_user_provided_mascot_is_local_and_embedded():
    mascot = ROOT / "assets" / "gemma_agent_a_green_dino.png"
    assert mascot.is_file()
    assert mascot.stat().st_size > 1024
    assert recommendation_mascot_data_url().startswith("data:image/png;base64,")
    assert recommendation_mascot_data_url() in render_recommendation_banner_html(None)


def test_same_floating_container_hosts_two_cards_even_after_completion():
    source = (ROOT / "app.py").read_text()
    assert 'with st.container(key="hdq_investigation_banner")' in source
    assert 'with banner_slot.container():' in source
    assert 'st.iframe(investigation_html, height=170)' in source
    assert 'st.iframe(render_recommendation_banner_html(recommendation_state)' in source
    assert 'render_finished_investigation_banner_html(' in source
    assert 'job_fingerprint(cache_key, [], run_id=run_id)' in source
    assert 'banner_slot.empty()' not in source[source.index('def render_live_investigation_status'):source.index('def _toggle_full_dataset_preview')]
    assert 'animation-timeline: --hdq-page-scroll' in sticky_banner_css()


def test_running_exploration_uses_live_activity_unchanged():
    html = render_exploration_banner_html({
        "status": "running", "started_at": "2026-11-16T16:15:00+00:00",
        "activity": "Planning test 1", "runs": [1, 2],
    })
    assert "Planning test 1" in html
    assert "Phase 3 of 3" in html
    assert "Verified exploratory checks: <b>2</b>" in html


def test_agent_a_mascot_moves_only_while_suggestions_are_pending():
    """Match the first mascot's exact bounce/sway, with a reduced-motion fallback."""
    active = render_recommendation_banner_html({"entries": {
        "f1": {"status": "ready"}, "f2": {"status": "running"},
    }})
    assert 'class="card is-active"' in active
    assert ".card.is-active figure img{animation:hdq-bound .95s ease-in-out infinite alternate}" in active
    assert "@keyframes hdq-bound{from{transform:translateY(3px) rotate(-1deg)}to{transform:translateY(-4px) rotate(1deg)}}" in active
    assert "@media(prefers-reduced-motion:reduce){.card.is-active figure img{animation:none}}" in active
    queued = render_recommendation_banner_html({"entries": {"f1": {"status": "pending"}}})
    assert 'class="card is-active"' in queued
    finished = render_recommendation_banner_html({"entries": {"f1": {"status": "ready"}}})
    assert 'class="card is-active"' not in finished
    errored = render_recommendation_banner_html({"entries": {"f1": {"status": "failed"}}})
    assert 'class="card is-active"' not in errored
    unscheduled = render_recommendation_banner_html(None)
    assert 'class="card is-active"' not in unscheduled
