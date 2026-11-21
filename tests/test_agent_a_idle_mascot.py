"""Agent A shows calm artwork when idle/finished, electric artwork only while running."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from healthcare_data_quality.ui.exploration_banner import (
    INVESTIGATION_IDLE_MASCOT,
    investigation_idle_mascot_data_url,
    mascot_data_url,
    render_idle_investigation_banner_html,
    render_investigation_phase_banner_html,
    render_exploration_banner_html,
    render_finished_investigation_banner_html,
)


def test_uploaded_idle_asset_is_bundled_and_local():
    assert INVESTIGATION_IDLE_MASCOT == ROOT / "assets" / "gemma_agent_a_idle_mascot.png"
    assert INVESTIGATION_IDLE_MASCOT.is_file()
    assert INVESTIGATION_IDLE_MASCOT.stat().st_size > 1000
    assert investigation_idle_mascot_data_url().startswith("data:image/png;base64,")


@pytest.mark.parametrize("state", ["no_dataset", "ready", "saved_available"])
def test_waiting_idle_shows_new_mascot_without_animation(state):
    html = render_idle_investigation_banner_html(state=state)
    assert investigation_idle_mascot_data_url() in html
    assert mascot_data_url() not in html
    assert 'style="animation:none"' in html
    assert 'id="hdq-live-clock"' not in html


@pytest.mark.parametrize("phase", [1, 2, 3])
def test_active_investigation_preserves_animated_electric_mascot(phase):
    html = render_investigation_phase_banner_html(
        phase=phase, phase_count=3, activity="Investigating", started_epoch_ms=1795000000000,
    )
    assert mascot_data_url() in html
    assert investigation_idle_mascot_data_url() not in html
    assert 'animation:hdq-bound .95s ease-in-out infinite alternate' in html
    assert '<img class="hdq-mascot"' in html
    assert 'style=""' in html


def test_async_exploration_preserves_running_mascot():
    html = render_exploration_banner_html({"status": "running", "activity": "Planning", "runs": []})
    assert mascot_data_url() in html
    assert investigation_idle_mascot_data_url() not in html


@pytest.mark.parametrize("state", ["review", "partial", "limited_clear"])
def test_completed_investigation_returns_to_calm_mascot(state):
    html = render_finished_investigation_banner_html(state=state, verified_checks=2)
    assert investigation_idle_mascot_data_url() in html
    assert mascot_data_url() not in html
    assert 'style="animation:none"' in html
