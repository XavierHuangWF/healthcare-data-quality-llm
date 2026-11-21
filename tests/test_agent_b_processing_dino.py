"""Agent B dinosaur changes with recommendation jobs; no model/API calls."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from healthcare_data_quality.ui.exploration_banner import (
    recommendation_mascot_data_url,
    recommendation_processing_mascot_data_url,
    render_recommendation_banner_html,
)


def test_user_supplied_processing_dino_embedded_in_active_card():
    assert (ROOT / "assets" / "gemma_agent_b_processing_dino.webp").is_file()
    active_uri = recommendation_processing_mascot_data_url()
    calm_uri = recommendation_mascot_data_url()
    assert active_uri.startswith("data:image/webp;base64,")
    for status in ("running", "pending"):
        html = render_recommendation_banner_html({"entries": {"f1": {"status": status}}})
        assert 'class="card is-active"' in html
        assert f'<img src="{active_uri}"' in html
        assert calm_uri not in html
        assert '.card.is-active figure img{animation:hdq-bound .95s ease-in-out infinite alternate}' in html


def test_calm_dino_retained_while_idle_or_finished():
    active_uri = recommendation_processing_mascot_data_url()
    calm_uri = recommendation_mascot_data_url()
    for snapshot, idle in ((None, True), (None, False),
                           ({"entries": {"f1": {"status": "ready"}}}, False),
                           ({"entries": {"f1": {"status": "failed"}}}, False)):
        html = render_recommendation_banner_html(snapshot, idle=idle)
        assert 'class="card is-active"' not in html
        assert f'<img src="{calm_uri}"' in html
        assert active_uri not in html
