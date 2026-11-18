"""The exploration hero uses real worker state; no static sample metrics."""
from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exploration_banner import (
    exploration_elapsed, mascot_data_url, render_exploration_banner_html,
    render_investigation_phase_banner_html, sticky_banner_css,
)


def test_elapsed_is_based_on_actual_worker_start():
    assert exploration_elapsed("2026-11-16T16:15:00+00:00", now=datetime(2026, 11, 16, 16, 17, 20, tzinfo=timezone.utc)) == "02:20"
    assert exploration_elapsed("2026-11-16T16:15:00+00:00", now=datetime(2026, 11, 16, 17, 17, 20, tzinfo=timezone.utc)) == "1:02:20"
    assert exploration_elapsed("bad") == "—"
    assert exploration_elapsed(None) == "—"


def test_hero_shows_measured_counts_and_worker_activity_not_static_examples():
    html = render_exploration_banner_html({
        "status": "running", "started_at": "2026-11-16T16:15:00+00:00",
        "activity": "Planning exploratory test 7", "runs": [{}, {}, {}],
    })
    assert 'Agent A — <span class="gold">exploring</span>' in html
    assert "Planning exploratory test 7" in html
    assert "Verified exploratory checks: <b>3</b>" in html
    assert "read-only" in html.lower()
    assert "not estimated completion time" in html
    assert "data:image/webp;base64," in html
    assert "You can review verified findings" in html


def test_hero_escapes_model_controlled_activity():
    html = render_exploration_banner_html({
        "status": "running", "activity": '<img src=x onerror=alert(1)> & "test"', "runs": []
    })
    assert '&lt;img src=x onerror=alert(1)&gt;' in html
    assert '<img src=x onerror=alert(1)>' not in html
    assert '&amp;' in html
    assert '&quot;test&quot;' in html


@pytest.mark.parametrize("status", ["complete", "incomplete", "failed", None])
def test_hero_only_when_running(status):
    with pytest.raises(ValueError, match="running"):
        render_exploration_banner_html({"status": status})


def test_asset_bundled_and_inline_no_external_network():
    assert (ROOT / "assets" / "gemma_electric_explorer.webp").is_file()
    assert mascot_data_url().startswith("data:image/webp;base64,")


def test_banner_replaces_old_status_not_findings_or_cleaning_tabs():
    app = (ROOT / "app.py").read_text()
    assert "investigation_html = render_exploration_banner_html(live)" in app
    assert "[\"Data quality findings\", \"Supporting evidence\", \"Clean data with Gemma\"]" in app
    assert "Gemma is still exploring; no verified exploratory check is available yet." not in app


def test_phase_two_registered_tools_has_same_large_banner():
    html = render_investigation_phase_banner_html(
        phase=2, phase_count=3,
        activity="Summarizing investigation evidence with Gemma...",
        started_epoch_ms=1795000000000,
    )
    assert 'Agent A — <span class="gold">investigating</span>' in html
    assert "Phase 2 of 3" in html
    assert "registered tools" in html
    assert "Summarizing investigation evidence with Gemma" in html
    assert "The registered-tool investigation is still running" in html
    assert "will become available when this phase finishes" in html
    assert "window.setInterval(render, 1000)" in html
    assert "1795000000000" in html
    assert "Charging evidence!" in html
    assert "data:image/webp;base64," in html


def test_phase_three_reuses_same_style_and_can_show_completed_checks():
    html = render_exploration_banner_html({
        "status": "running",
        "started_at": "2026-11-16T16:15:00+00:00",
        "activity": "Verified candidate 2", "runs": [{}, {}],
    })
    assert 'class="hdq-hero"' in html
    assert "Phase 3 of 3" in html
    assert "Verified exploratory checks: <b>2</b>" in html
    assert "window.setInterval(render, 1000)" in html
    assert "You can review verified findings" in html


def test_phase_one_does_not_imply_gemma_has_started():
    html = render_investigation_phase_banner_html(
        phase=1, phase_count=3, activity="Profiling CSV", started_epoch_ms=1,
    )
    assert "Agent A — checking data quality" in html
    assert "Gemma is investigating" not in html
    assert "Phase 1 of 3" in html


def test_phase_two_escapes_activity_and_validates_epoch():
    html = render_investigation_phase_banner_html(
        phase=2, phase_count=3,
        activity='<img src=x onerror=alert(1)>', started_epoch_ms=123,
    )
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert '<img src=x onerror=alert(1)>' not in html
    with pytest.raises(ValueError):
        render_investigation_phase_banner_html(
            phase=2, phase_count=3, activity="test", started_epoch_ms=-1,
        )


def test_streamlit_phase_two_banner_rendered_before_long_model_call():
    app = (ROOT / "app.py").read_text()
    assert 'self.hero_slot = st.empty()' in app
    assert 'self.hero_container = st.container(key="hdq_investigation_banner_blocking")' in app
    assert 'components.html(render_investigation_phase_banner_html(' in app
    assert 'baseline_ready_callback=publish_baseline' in (ROOT / "investigation_jobs.py").read_text()
    assert 'render_investigation_phase_banner_html(' in app
    assert 'self.hero_slot.empty()' in app
    assert 'expanded=False' in app


def test_banner_is_compact_and_keeps_live_state():
    html = render_exploration_banner_html({
        "status": "running", "started_at": "2026-11-16T16:15:00+00:00",
        "activity": "Generating verified exploration 2", "runs": [{}, {}],
    })
    assert "min-height:142px" in html
    assert "max-height:122px" in html
    assert "Verified exploratory checks: <b>2</b>" in html
    assert "Generating verified exploration 2" in html
    assert "window.setInterval(render, 1000)" in html


def test_full_banner_floats_at_viewport_level_not_inside_iframe():
    css = sticky_banner_css()
    app = (ROOT / "app.py").read_text()
    assert ".st-key-hdq_investigation_banner" in css
    assert ".st-key-hdq_investigation_banner_blocking" in css
    assert "position: fixed !important" in css
    assert "right: 1.1rem !important" in css
    assert "width: min(490px" in css
    assert "position: sticky" not in css
    assert "animation-timeline: --hdq-page-scroll" in css
    assert "@keyframes hdq-follow-scroll-thumb" in css
    assert "[data-testid=\"stAppViewContainer\"]" in css
    assert 'st.markdown(sticky_banner_css(), unsafe_allow_html=True)' in app
    assert 'with st.container(key="hdq_investigation_banner"):\n        live_banner_slot = st.empty()' in app
    assert "live_status(audit, cache_key, header_reports, include_exploration, live_banner_slot)" in app
    assert "with banner_slot.container():" in app
    assert "render_recommendation_banner_html(recommendation_state)" in app
    assert "render_finished_investigation_banner_html(" in app
    assert "height=170, scrolling=False" in app


def test_larger_floating_card_keeps_scroll_tracking_and_mobile_fit():
    css = sticky_banner_css()
    html = render_investigation_phase_banner_html(
        phase=2, phase_count=3, activity="Exploring", started_epoch_ms=1795000000000
    )
    assert "width: min(490px" in css
    assert "width: min(390px" in css
    assert "position: fixed !important" in css
    assert "animation-timeline: --hdq-page-scroll" in css
    assert "grid-template-columns:minmax(0,1fr) 144px" in html
    assert "font-size:18px;font-weight:800" in html
    assert "max-height:122px" in html
    app = (ROOT / "app.py").read_text()
    # Synchronous, live, fallback, and parked-before-run investigation cards.
    assert app.count("height=170, scrolling=False") >= 3
