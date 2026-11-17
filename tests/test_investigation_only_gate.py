"""Regression: default CLI and Streamlit audit never call the proposal model."""

import csv
from pathlib import Path
from unittest.mock import patch

from healthcare_data_quality.workflow import run_initial_audit


def setup_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "sample.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows([["Taxon", "S1"], ["x", "1"]])
    return path


def test_default_audit_does_not_generate_or_validate_remediation(tmp_path):
    path = setup_dataset(tmp_path)
    baseline = {"verified_contexts": [], "checks": {}}
    investigation = {"status": "done", "history": [
        {"step": 1, "tool": "observe_matrix_structure", "observations": [
            {"rows": 1, "columns": 2}
        ]}
    ], "report": {"findings": [], "unresolved_questions": []}}
    with (patch("healthcare_data_quality.workflow.run_baseline_audit", return_value=baseline),
          patch("healthcare_data_quality.workflow.run_investigation", return_value=investigation),
          patch("healthcare_data_quality.workflow.get_decision_context", return_value=[]),
          patch("healthcare_data_quality.workflow.get_verified_outcome_context", return_value=[]),
          patch("healthcare_data_quality.workflow.generate_remediation_proposals") as generation,
          patch("healthcare_data_quality.workflow.validate_remediation_proposals") as validation):
        result = run_initial_audit(path)
    generation.assert_not_called()
    validation.assert_not_called()
    assert result["proposals"] == []
    assert result["proposal_generation_error"] is None


def test_legacy_mode_requires_explicit_proposal_opt_in(tmp_path):
    path = setup_dataset(tmp_path)
    baseline = {"verified_contexts": [], "checks": {}}
    investigation = {"status": "done", "history": [
        {"step": 1, "tool": "observe_matrix_structure", "observations": [
            {"rows": 1, "columns": 2}
        ]}
    ], "report": {"findings": [], "unresolved_questions": []}}
    with (patch("healthcare_data_quality.workflow.run_baseline_audit", return_value=baseline),
          patch("healthcare_data_quality.workflow.run_investigation", return_value=investigation),
          patch("healthcare_data_quality.workflow.get_decision_context", return_value=[]),
          patch("healthcare_data_quality.workflow.get_verified_outcome_context", return_value=[]),
          patch("healthcare_data_quality.workflow.generate_remediation_proposals", return_value={"proposals": []}) as generation,
          patch("healthcare_data_quality.workflow.validate_remediation_proposals", return_value=[]) as validation):
        result = run_initial_audit(path, generate_proposals=True)
    generation.assert_called_once()
    validation.assert_called_once()
    assert result["proposals"] == []
