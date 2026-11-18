"""Regression: Captain mascot for idle/done; electric mascot only while running."""
import base64
from datetime import datetime, timezone
from pathlib import Path
from exploration_banner import (
    render_idle_investigation_banner_html,
    render_investigation_phase_banner_html,
    render_finished_investigation_banner_html,
    MASCOT, IDLE_MASCOT,
)


def _source_fragment(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")[:90]


def test_waiting_uses_uploaded_captain_not_electric():
    html = render_idle_investigation_banner_html(state="no_dataset")
    assert _source_fragment(IDLE_MASCOT) in html
    assert _source_fragment(MASCOT) not in html
    assert 'animation:none' in html


def test_ready_and_saved_result_still_use_captain():
    for state in ("ready", "saved_available"):
        html = render_idle_investigation_banner_html(state=state)
        assert _source_fragment(IDLE_MASCOT) in html
        assert _source_fragment(MASCOT) not in html


def test_active_investigation_uses_electric_with_animation():
    for phase in (1, 2, 3):
        html = render_investigation_phase_banner_html(
            phase=phase, phase_count=3, activity="Investigating",
            started_epoch_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
        )
        assert _source_fragment(MASCOT) in html
        assert _source_fragment(IDLE_MASCOT) not in html
        assert 'class="hdq-mascot"' in html
        assert 'style="animation:none"' not in html


def test_finished_states_use_captain_stationary():
    for state in ("review", "partial", "limited_clear"):
        html = render_finished_investigation_banner_html(state=state, verified_checks=1)
        assert _source_fragment(IDLE_MASCOT) in html
        assert _source_fragment(MASCOT) not in html
        assert 'style="animation:none"' in html
