"""Agent B must produce an action, not a question assigned to the researcher."""
import json

import pytest
from cleaning_review import ask_gemma_for_review_options


def _finding(source="QC", actionable=True):
    return {"id": "sample:qc:1", "source": source, "actionable": actionable,
            "title": "2 fractional count values", "method": "Check numeric counts",
            "evidence": [], "total_evidence_records": 2, "check_key": "counts"}


def test_direct_action_is_returned_and_keeps_legacy_handoff(monkeypatch):
    seen = []
    def fake_query(prompt, **kw):
        seen.append(prompt)
        return json.dumps({"recommendations": [{
            "finding_id": "sample:qc:1", "action_type": "set_missing",
            "action": "Replace both invalid raw-count cells with NA in a derived CSV, retaining originals in the audit.",
            "rationale": "The dataset does not establish that these are raw integer counts."
        }]})
    monkeypatch.setattr("cleaning_review.query_local_model", fake_query)
    item = _finding()
    rec = ask_gemma_for_review_options([item])[item["id"]]
    assert rec["suggestion"].startswith("Replace both invalid")
    assert "decision_needed" in rec  # persisted field consumed by Agent C
    assert "ONE concrete, usable" in seen[0]
    assert "No flag_and_preserve operation exists" in seen[0]
    assert "DATA:" in seen[0]


@pytest.mark.parametrize("bad", [
    "Researcher must determine whether these are counts.",
    "Ask the user which replacement is appropriate.",
    "Determine whether the cells should be rounded.",
])
def test_generic_researcher_tasks_are_rejected(monkeypatch, bad):
    monkeypatch.setattr("cleaning_review.query_local_model", lambda *a, **k:
        json.dumps({"recommendations": [{
            "finding_id": "sample:qc:1", "action_type": "scoped_replacement",
            "action": bad, "rationale": "The source policy is unavailable."
        }]}))
    with pytest.raises(ValueError, match="remediation|defers the decision"):
        ask_gemma_for_review_options([_finding()])


def test_zero_finding_check_cannot_generate_data_edit(monkeypatch):
    monkeypatch.setattr("cleaning_review.query_local_model", lambda *a, **k:
        json.dumps({"recommendations": [{
            "finding_id": "sample:qc:1", "action_type": "scoped_replacement",
            "action": "Replace all values with zero.",
            "rationale": "No issues observed in the completed check."
        }]}))
    with pytest.raises(ValueError, match="zero-finding"):
        ask_gemma_for_review_options([_finding(source="Gemma check", actionable=False)])


def test_zero_finding_check_can_preserve_data(monkeypatch):
    monkeypatch.setattr("cleaning_review.query_local_model", lambda *a, **k:
        json.dumps({"recommendations": [{
            "finding_id": "sample:qc:1", "action_type": "no_change",
            "action": "Keep the dataset unchanged for this completed check.",
            "rationale": "The check found no additional candidate data issues."
        }]}))
    assert "Keep the dataset" in ask_gemma_for_review_options(
        [_finding(source="Gemma check", actionable=False)])["sample:qc:1"]["suggestion"]


@pytest.mark.parametrize("action_type, action", [
    ("flag_and_preserve", "Flag the suspicious cells for later review."),
    ("scoped_replacement", "Flag the suspicious cells but leave all original values untouched."),
    ("no_change", "Keep the observed invalid counts as they are."),
])
def test_observed_issue_must_receive_real_remediation(monkeypatch, action_type, action):
    monkeypatch.setattr("cleaning_review.query_local_model", lambda *a, **k:
        json.dumps({"recommendations": [{
            "finding_id": "sample:qc:1", "action_type": action_type,
            "action": action, "rationale": "This is not a valid repair."
        }]}))
    with pytest.raises((ValueError, __import__("jsonschema").ValidationError)):
        ask_gemma_for_review_options([_finding()])


@pytest.mark.parametrize("kind, method", [
    ("pad_taxonomy", "Pad truncated seven-rank lineages with NA in absent terminal rank positions, preserving all known names."),
    ("impute_missing", "Impute missing measurement values using a validated sample-specific model with explicit covariates."),
    ("standardize_format", "Standardize alternate taxonomy separators to semicolons without modifying taxon labels."),
    ("rename_header", "Rename repeated physical-column headers with unique position suffixes while preserving measurements."),
])
def test_scientifically_specific_repair_methods_are_accepted(monkeypatch, kind, method):
    monkeypatch.setattr("cleaning_review.query_local_model", lambda *a, **k:
        json.dumps({"recommendations": [{
            "finding_id": "sample:qc:1", "action_type": kind,
            "action": method,
            "rationale": "This preserves observed source information and follows consistent-data conventions."
        }]}))
    suggestion = ask_gemma_for_review_options([_finding()])["sample:qc:1"]["suggestion"]
    assert suggestion == method


def test_the_prompt_preserves_total_evidence_count_and_real_cell_sample(monkeypatch):
    captured = {}
    def fake_query(prompt, **kwargs):
        captured["payload"] = json.loads(prompt.split("DATA:\n", 1)[1])
        return json.dumps({"recommendations": [{
            "finding_id": "sample:qc:1", "action_type": "set_missing",
            "action": "Replace the two invalid count observations with NA in a reviewed working copy.",
            "rationale": "Nonnegative integer counts should not contain fractions, assuming raw reads."
        }]})
    monkeypatch.setattr("cleaning_review.query_local_model", fake_query)
    finding = _finding()
    finding["evidence"] = [{"CSV row": 11, "Column": 7, "Original value": "0.2"}]
    ask_gemma_for_review_options([finding])
    assert captured["payload"][0]["recorded_evidence_count"] == 2
    assert captured["payload"][0]["evidence_records_provided"] == 1
    assert captured["payload"][0]["evidence_sample"][0]["Original value"] == "0.2"
