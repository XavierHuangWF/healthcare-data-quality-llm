"""One-to-one evidence and no-silent-cleaning regression tests."""
import csv
import json
from pathlib import Path

import pytest

from healthcare_data_quality.cleaning.cleaning_authorization import file_sha256
from healthcare_data_quality.cleaning.cleaning_workspace import (
    validate_header_plan, create_header_request,
    execute_confirmed_header_request, ask_gemma_about_cleaning,
)
from healthcare_data_quality.investigation.finding_evidence import (
    dataset_finding_evidence, exploratory_finding_evidence,
)
from healthcare_data_quality.ui.finding_presentation import baseline_issues
from healthcare_data_quality.investigation.ingestion import inspect_headers


def _csv(tmp_path: Path, columns=None):
    path = tmp_path / "input.csv"
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(columns or ["Taxon", "Sample", "Sample"])
        writer.writerows([["k__Bacteria", "1", "1"], ["NA", "", ""]])
    return path


def test_evidence_order_matches_display_findings_and_has_methods(tmp_path):
    path = _csv(tmp_path)
    baseline = {"checks": {"find_invalid_counts": {"status": "completed", "observations": [{
        "row": 3, "column": 2, "sample_id": "Sample", "taxon": "NA",
        "original_value": "", "issue_type": "missing",
    }]}}}
    entries = dataset_finding_evidence(baseline, path, inspect_headers(path))
    assert [x["title"] for x in entries] == [
        *[f["title"] for f in baseline_issues(baseline, path)],
        "Columns 2 and 3 (‘Sample’) contain identical values",
    ]
    assert entries[0]["check_key"] == "find_invalid_counts"
    assert entries[0]["evidence"][0]["CSV row"] == 3
    assert "Zero counts are valid" in entries[0]["method"]
    assert entries[1]["evidence"][0]["Column A"] == 2
    assert entries[1]["evidence"][0]["Matching rows"] == 2


def test_exploratory_evidence_keeps_verification_distinct_from_scientific_review():
    runs = [{"proposal": {"hypothesis": "Check outliers", "method": "Compare values", "code": "print(1)"},
             "result": {"findings": [{"title": "Candidate", "evidence": [{"csv_row": 2}]}]},
             "verification": {"source_cells_verified": True}}]
    catalog = exploratory_finding_evidence(runs)
    assert catalog[0]["title"] == "Candidate"
    assert catalog[0]["evidence"][0]["csv_row"] == 2
    assert "unvalidated" in catalog[0]["verification"]
    assert catalog[0]["verification_details"]["source_cells_verified"] is True


def test_cleaning_plan_anchors_original_positions_and_hash(tmp_path):
    path = _csv(tmp_path)
    plan = validate_header_plan({"operation": "rename_headers", "dataset_id": "dataset_1",
        "changes": [{"position": 2, "before": "Sample", "after": "Sample_A"}]}, [path])
    assert plan["source_sha256"] == file_sha256(path)
    assert plan["changes"][0]["position"] == 2


@pytest.mark.parametrize("changes", [
    [{"position": 2, "before": "wrong", "after": "New"}],
    [{"position": 2, "before": "Sample", "after": "Taxon"}],
    [{"position": 2, "before": "Sample", "after": "Bad,New"}],
    [{"position": 0, "before": "Taxon", "after": "New"}],
    [{"position": True, "before": "Taxon", "after": "New"}],
    [{"position": 2, "before": "Sample", "after": "New"},
     {"position": 2, "before": "Sample", "after": "Other"}],
])
def test_reject_inaccurate_or_unsafe_model_plans(tmp_path, changes):
    path = _csv(tmp_path)
    with pytest.raises(ValueError):
        validate_header_plan({"operation": "rename_headers", "dataset_id": "dataset_1",
                              "changes": changes}, [path])


def test_authorization_is_required_and_derived_copy_preserves_records(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = _csv(tmp_path)
    original_bytes = path.read_bytes()
    approved = validate_header_plan({"operation": "rename_headers", "dataset_id": "dataset_1",
        "changes": [{"position": 2, "before": "Sample", "after": "Sample_A"}]}, [path])
    request = create_header_request(approved, path, tmp_path / "workspace")
    with pytest.raises(PermissionError):
        execute_confirmed_header_request(request, "reviewer", "no")
    assert not list((tmp_path / "workspace").glob("cleaned_*.csv"))
    record = execute_confirmed_header_request(request, "reviewer", "AUTHORIZE HEADER CLEANING TO COPY")
    assert record["status"] == "verified"
    cleaned = Path(record["output"])
    assert cleaned.is_file()
    assert cleaned.read_bytes().split(b"\r\n", 1)[-1] == original_bytes.split(b"\r\n", 1)[-1]
    assert path.read_bytes() == original_bytes
    with cleaned.open(newline="") as stream:
        assert next(csv.reader(stream)) == ["Taxon", "Sample_A", "Sample"]
    log = tmp_path / "data" / "decision_memory" / "operation_authorizations.jsonl"
    assert json.loads(log.read_text().strip())["human_confirmation"] is True


def test_changed_source_rejected_between_review_and_execution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = _csv(tmp_path)
    plan = validate_header_plan({"operation": "rename_headers", "dataset_id": "dataset_1",
         "changes": [{"position": 2, "before": "Sample", "after": "NewSample"}]}, [source])
    request = create_header_request(plan, source, tmp_path / "workspace")
    source.write_bytes(source.read_bytes() + b"EXTRA\r\n")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        execute_confirmed_header_request(request, "reviewer", "AUTHORIZE HEADER CLEANING TO COPY")


def test_model_output_is_discussion_only_until_validated(monkeypatch):
    payload = {"reply": "The dataset needs review.", "operation": "none",
               "dataset_id": "", "changes": []}
    def fake_llm(prompt, **kwargs):
        assert "never treat NA taxonomy ranks as invalid" in prompt
        assert kwargs["json_schema"]
        return json.dumps(payload)
    monkeypatch.setattr("healthcare_data_quality.cleaning.cleaning_workspace.query_local_model", fake_llm)
    assert ask_gemma_about_cleaning("What to do?", [], [], [])["operation"] == "none"
