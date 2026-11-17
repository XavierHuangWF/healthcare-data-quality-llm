"""Citations do not establish semantic correctness."""

import json
from unittest.mock import patch

import pytest

from healthcare_data_quality.evidence_claim_validation import (
    canonical_observation_fact, verify_report_claims,
)
from healthcare_data_quality.investigation_report import summarize_investigation


EVIDENCE = {"T1.O1": {"tool": "compare_columns", "observation": {
    "left_position": 105, "right_position": 106,
    "matching_nonmissing": 12, "both_missing": 2,
    "one_missing": 0, "conflicting_nonmissing": 0,
}}}


def report(claim, ids=None):
    return {"findings": [{"claim": claim, "evidence_ids": ids or ["T1.O1"]}],
            "unresolved_questions": [], "recommended_next_step": "Check provenance"}


def test_model_speculation_not_verified_even_with_valid_citation():
    result = verify_report_claims(report("These are the same biological specimen."), EVIDENCE)
    check = result["findings"][0]["claim_verification"]
    assert check["citation_ids_valid"]
    assert check["status"] == "requires_scientific_review"
    assert not check["scientific_interpretation_verified"]
    assert "Columns 105 and 106:" in check["verified_observation_facts"][0]


def test_exact_deterministic_fact_can_be_verified():
    fact = canonical_observation_fact("compare_columns", EVIDENCE["T1.O1"]["observation"])
    result = verify_report_claims(report(fact), EVIDENCE)
    assert result["findings"][0]["claim_verification"]["status"] == "verified_exact_fact"


def test_unsupported_or_fabricated_reference_is_rejected():
    with pytest.raises(ValueError, match="Unknown evidence reference"):
        verify_report_claims(report("Unsupported", ["T8.O3"]), EVIDENCE)
    with pytest.raises(ValueError, match="require at least one"):
        verify_report_claims({"findings": [{"claim": "Unsupported", "evidence_ids": []}]}, {})


def test_summarizer_returns_evidence_check_marks():
    history = [{"step": 1, "tool": "compare_columns",
                "observations": [EVIDENCE["T1.O1"]["observation"]]}]
    with patch("healthcare_data_quality.investigation_report.query_local_model",
               return_value=json.dumps(report("These are the same biological specimen."))):
        output = summarize_investigation(history)
    assert output["findings"][0]["claim_verification"]["status"] == "requires_scientific_review"
