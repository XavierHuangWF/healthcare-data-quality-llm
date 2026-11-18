"""Both mascots park before investigation; no inference or results are fabricated."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exploration_banner import (
    render_idle_investigation_banner_html,
    render_recommendation_banner_html,
    render_finished_investigation_banner_html,
    render_exploration_banner_html,
)


@pytest.mark.parametrize("stage,text", [
    ("no_dataset", "Upload a CSV to begin"),
    ("ready", "Run full investigation to start"),
    ("saved_available", "Load the saved result or run a new investigation"),
])
def test_investigation_idles_with_accurate_state_no_fake_progress(stage, text):
    html = render_idle_investigation_banner_html(state=stage)
    assert "Gemma investigation" in html
    assert text in html
    assert 'aria-label="Not started"' in html
    assert "○ Exploratory checks: <b>Not started</b>" in html
    assert 'id="hdq-live-clock"' not in html
    assert "window.setInterval(render, 1000)" not in html
    assert "animation:none" in html
    assert "Phase 1 of 3" not in html
    assert 'role="status"' in html


def test_idle_agent_b_has_no_jobs_or_approval_and_stops_animation():
    html = render_recommendation_banner_html(None, idle=True)
    assert "Agent B — waiting for QC findings" in html
    assert "Recommendations begin automatically" in html
    assert '<div class="meter">Not started</div>' in html
    assert 'aria-label="Not started"' in html
    assert 'class="card is-active"' not in html
    assert 'aria-valuenow="0"' not in html  # no fake known denominator
    # Normal, loaded-snapshot behavior is preserved.
    active = render_recommendation_banner_html({"entries": {"f1": {"status": "running"}}})
    assert 'class="card is-active"' in active
    assert "1 generating" in active


def test_idle_renderer_rejects_unsupported_states():
    with pytest.raises(ValueError, match="Unsupported idle"):
        render_idle_investigation_banner_html(state="completed")


def test_page_mounts_one_floating_host_before_upload_and_uses_it_for_all_states():
    source = (ROOT / "app.py").read_text()
    beginning = source.index("def render_ui()")
    window = source[beginning:]
    host = 'with st.container(key="hdq_investigation_banner"):'
    assert window.count(host) == 1
    assert window.index(host) < window.index('uploaded = st.file_uploader(')
    assert 'render_pre_investigation_status(live_banner_slot, state="no_dataset")' in window
    assert 'state="saved_available" if previous_runs or legacy_session_result else "ready"' in window
    assert 'live_status(audit, cache_key, header_reports, include_exploration, live_banner_slot)' in window
    assert 'with banner_slot.container():' in source
    assert 'components.html(render_recommendation_banner_html(None, idle=True)' in source
    # Pre-run state is just HTML: starting Agent A requires a real audit.
    idle_func = source[source.index("def render_pre_investigation_status("):
                       source.index("def _toggle_full_dataset_preview(")]
    assert "recommendation_manager().start" not in idle_func
    assert "schedule_agent_a_once" not in idle_func


def test_after_investigation_lifecycle_is_unchanged():
    done = render_finished_investigation_banner_html(state="partial", verified_checks=1)
    assert "Agent A — review partial results" in done
    assert "Verified exploratory checks: <b>1</b>" in done
    underway = render_exploration_banner_html({
        "status": "running", "activity": "Planning exploratory investigation 2",
        "started_at": "2026-11-16T12:00:00+00:00", "runs": [1],
    })
    assert 'Agent A — <span class="gold">exploring</span>' in underway
    assert "Planning exploratory investigation 2" in underway


@pytest.mark.parametrize("state,headline", [
    ("no_dataset", "Agent A — waiting for dataset"),
    ("ready", "Agent A — ready to investigate"),
    ("saved_available", "Agent A — saved result available"),
])
def test_agent_a_uses_agent_named_heading_in_idle_states(state, headline):
    html = render_idle_investigation_banner_html(state=state)
    assert f'<h2 class="hdq-title">{headline}</h2>' in html
    assert "⚡ Agent A ·" not in html


@pytest.mark.parametrize("state,headline", [
    ("partial", "Agent A — review partial results"),
    ("review", "Agent A — investigation finished"),
    ("limited_clear", "Agent A — investigation finished"),
])
def test_agent_a_uses_agent_named_heading_on_completion(state, headline):
    html = render_finished_investigation_banner_html(state=state, verified_checks=1)
    assert f'<h2 class="hdq-title">{headline}</h2>' in html


def test_agent_a_heading_uses_same_title_size_as_b_and_c():
    html = render_idle_investigation_banner_html(state="no_dataset")
    assert ".hdq-title {font-size:18px;font-weight:800" in html
