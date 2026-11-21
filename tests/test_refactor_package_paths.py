"""Protect import and local file paths after the package directory refactor."""
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_relocated_components_are_importable_under_single_package_name():
    for name in (
        'healthcare_data_quality.workflow',
        'healthcare_data_quality.agents.agent_loop',
        'healthcare_data_quality.investigation.baseline_audit',
        'healthcare_data_quality.cleaning.finding_cleaning',
        'healthcare_data_quality.cleaning.in_memory_cleaning',
        'healthcare_data_quality.services.llm_client',
        'healthcare_data_quality.ui.exploration_banner',
        'healthcare_data_quality.learning.interaction_memory',
        'healthcare_data_quality.learning.adaptive_skill_memory',
    ):
        assert import_module(name).__name__ == name


def test_bundled_mascots_still_resolve_after_moving_banner():
    from healthcare_data_quality.ui.exploration_banner import (
        MASCOT, IDLE_MASCOT, INVESTIGATION_IDLE_MASCOT,
        investigation_idle_mascot_data_url,
    )
    assert MASCOT == ROOT / 'assets' / 'gemma_electric_explorer.webp'
    assert IDLE_MASCOT == INVESTIGATION_IDLE_MASCOT
    assert IDLE_MASCOT == ROOT / 'assets' / 'gemma_agent_a_idle_mascot.png'
    assert investigation_idle_mascot_data_url().startswith('data:image/png;base64,')


def test_project_local_memory_and_skills_remain_outside_src():
    from healthcare_data_quality.learning import adaptive_skill_memory, interaction_memory
    assert adaptive_skill_memory._PROJECT == ROOT
    assert interaction_memory.DEFAULT_PATH == ROOT / 'memory' / 'approved_decisions.jsonl'


def test_installed_qc_source_is_still_available_for_evidence_review():
    from healthcare_data_quality.ui.evidence_view import deterministic_detection_code
    codes = deterministic_detection_code('find_invalid_counts')
    assert codes
    assert codes[0]['path'].startswith('src/healthcare_data_quality/')
    assert 'def ' in codes[0]['code']
