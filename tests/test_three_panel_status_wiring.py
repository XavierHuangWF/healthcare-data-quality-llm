"""Regression for three stacked Streamlit status cards and state-based mascots."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from exploration_banner import (
    agent_c_idle_mascot_data_url,
    agent_c_working_mascot_data_url,
    recommendation_mascot_data_url,
    recommendation_processing_mascot_data_url,
    recommendation_progress_counts,
    render_agent_c_banner_html,
    render_recommendation_banner_html,
)

ROOT = Path(__file__).resolve().parents[1]


def _function_source(name: str) -> str:
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    mod = ast.parse(source)
    node = next(node for node in mod.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
    return ast.get_source_segment(source, node)


def test_agent_c_panel_is_actually_mounted_in_every_rendering_stage():
    live = _function_source("render_live_investigation_status")
    parked = _function_source("render_pre_investigation_status")
    app = _function_source("render_ui")
    assert 'with st.container(key="hdq_agent_c_status_card")' in live
    assert 'render_agent_c_banner_html(' in live
    assert 'render_agent_c_banner_html(idle=True)' in parked
    assert 'render_agent_c_banner_html(' in app
    # Same scroll-following host is used; no independently fixed third overlay.
    assert 'with banner_slot.container():' in live
    assert 'with banner_slot.container():' in parked
    assert 'with live_banner_slot.container():' in app


def test_agent_c_active_state_follows_only_pending_requests_from_current_ready_findings():
    mod = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    node = next(node for node in mod.body
                if isinstance(node, ast.FunctionDef) and node.name == "_agent_c_status_counts")
    namespace = {
        "recommendation_progress_counts": recommendation_progress_counts,
        "st": SimpleNamespace(session_state={"cleaning_pending_gemma_b": {
            "current": {"job_id": "job-a"},
            "old": {"job_id": "job-from-older-audit"},
        }}),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / "app.py"), "exec"), namespace)
    assert namespace["_agent_c_status_counts"]({"entries": {
        "current": {"status": "ready"},
        "new": {"status": "pending"},
    }}) == (1, 2, 1)
    assert namespace["_agent_c_status_counts"](None) == (0, 0, 0)


def test_agent_b_uses_uploaded_angry_dino_only_during_active_jobs():
    angry = recommendation_processing_mascot_data_url()
    calm = recommendation_mascot_data_url()
    assert angry.startswith("data:image/webp;base64,")
    assert angry != calm
    for status in ("running", "pending"):
        html = render_recommendation_banner_html({"entries": {"x": {"status": status}}})
        assert f'<img src="{angry}"' in html
        assert 'class="card is-active"' in html
        assert calm not in html
    for snapshot, is_idle in ((None, True), ({"entries": {"x": {"status": "ready"}}}, False),
                              ({"entries": {"x": {"status": "failed"}}}, False)):
        html = render_recommendation_banner_html(snapshot, idle=is_idle)
        assert f'<img src="{calm}"' in html
        assert 'class="card is-active"' not in html
        assert angry not in html
        assert '.card:not(.is-active) figure img{animation:none!important;transform:none!important}' in html


def test_agent_c_idle_and_replying_use_supplied_distinct_images():
    idle = agent_c_idle_mascot_data_url()
    working = agent_c_working_mascot_data_url()
    assert idle.startswith("data:image/png;base64,") and working.startswith("data:image/png;base64,")
    assert idle != working
    waiting = render_agent_c_banner_html(idle=True)
    ready = render_agent_c_banner_html(ready_findings=2, total_findings=5)
    replying = render_agent_c_banner_html(ready_findings=2, total_findings=5, active_conversations=1)
    assert idle in waiting and working not in waiting
    assert idle in ready and working not in ready
    assert working in replying and idle not in replying
    assert 'class="card is-active"' in replying
    assert 'class="card is-active"' not in ready
    assert '.card:not(.is-active) figure img{animation:none!important;transform:none!important}' in ready
