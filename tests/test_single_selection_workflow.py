"""An actionable selection is the only reviewer authorization click."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from healthcare_data_quality.ui.single_selection_flow import proposal_request_once

ROOT = Path(__file__).resolve().parents[1]


def test_read_only_proposal_is_deduplicated_across_reruns_and_model_replies():
    state = {}
    original = b'taxon,a\nfirst,0.2\n'
    one = proposal_request_once(state, 'finding-a', 'I recommend deleting column a',
                                original, user_turn='3:delete column a')
    assert len(one) == 64
    assert proposal_request_once(state, 'finding-a',
                                 'I will prepare a change for column a', original,
                                 user_turn='3:delete column a') is None
    # Different reviewer turn and different finding can each request a plan.
    assert proposal_request_once(state, 'finding-a', 'I recommend deleting column a',
                                 original, user_turn='5:delete column a') != one
    assert proposal_request_once(state, 'finding-b', 'I recommend deleting column a',
                                 original, user_turn='3:delete column a') != one
    assert not state.get('cleaning_history')
    assert not state.get('cleaning_staged_bytes_by_dataset')


def test_read_only_request_refuses_missing_identity_or_source_type():
    import pytest
    with pytest.raises(ValueError):
        proposal_request_once({}, '', 'make change', b'csv')
    with pytest.raises(TypeError):
        proposal_request_once({}, 'finding', 'make change', 'csv')
    assert proposal_request_once({}, 'finding', ' ', b'csv') is None


def test_agent_c_never_asks_yes_to_prepare_before_the_final_preview():
    app = (ROOT / 'app.py').read_text(encoding='utf8')
    assert '"Yes — Prepare Changes"' not in app
    assert 'proposal_request_once(' in app
    assert 'READ-ONLY planning, not authorization to execute.' in app
    assert 'from healthcare_data_quality.ui.one_click_choice import execute_verified_choice' in app
    # The selected Agent B option already has Before→After before its one click.
    ready = app.split('def _render_live_agent_b_finding(', 1)[1].split(
        'def _expand_ready_finding_without_rerun(', 1)[0]
    assert "st.dataframe(pd.DataFrame(option['rows'])" in ready
    assert 'outcome = execute_verified_choice(' in ready
    assert '_record_gemma_b_response(fid, finding, source, {' not in ready


def test_non_destructive_installer_is_idempotent_and_rejects_unknown_layout():
    spec = importlib.util.spec_from_file_location(
        'hdq_selection_installer', ROOT / 'tools/apply_single_selection_workflow.py')
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    source = (ROOT / 'app.py').read_text(encoding='utf8')
    assert installer.patch_app(source) == source
    import pytest
    with pytest.raises(ValueError, match='layout differs'):
        installer.patch_app(source.replace(installer.START, '    # altered layout\n'))
