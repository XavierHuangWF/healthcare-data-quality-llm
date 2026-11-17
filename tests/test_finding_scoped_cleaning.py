"""A human can apply one finding without silently fixing another."""
import csv
from pathlib import Path

import pytest

from healthcare_data_quality.cleaning_authorization import file_sha256
from healthcare_data_quality.finding_cleaning import (
    APPROVAL_PHRASE, allowed_positions, build_cleaning_findings,
    execute_approved_scoped_plan, reviewed_python_code, validate_scoped_plan,
)


def sample(tmp_path):
    source = tmp_path / "sample.csv"
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Taxon", "Sample_A", "Sample_B"])
        writer.writerows([
            ["k__Bacteria; p__X; c__X; o__X; f__X; g__X; NA", "0.5", ""],
            ["k__Bacteria| p__Y| c__Y| o__Y| f__Y| g__Y| NA", "-3", "4"],
        ])
    return source


def catalog(source):
    baseline = {"checks": {"find_invalid_counts": {"status": "completed", "observations": [
        {"row": 2, "column": 2, "issue_type": "fractional", "sample_id": "Sample_A", "taxon": "k__A", "original_value": "0.5"},
        {"row": 2, "column": 3, "issue_type": "missing", "sample_id": "Sample_B", "taxon": "k__A", "original_value": ""},
    ]}}}
    audit = {"dataset_baselines": {"dataset_1": baseline}}
    return build_cleaning_findings(audit, [source], [source.name], [])


def find(items, phrase):
    return next(x for x in items if phrase in x["title"])


def test_finding_catalog_is_aligned_with_original_observations(tmp_path):
    source = sample(tmp_path)
    results = catalog(source)
    assert find(results, "fractional count")["source"] == "QC"
    assert (2, 2) in allowed_positions(find(results, "fractional count"), source)[0]
    assert (3, 2) not in allowed_positions(find(results, "fractional count"), source)[0]
    assert (2, 3) in allowed_positions(find(results, "missing value"), source)[0]


def test_one_approved_finding_edits_only_scoped_cell_and_can_chain(tmp_path):
    source = sample(tmp_path)
    original = source.read_bytes()
    findings = catalog(source)
    fractional = find(findings, "fractional count")
    missing = find(findings, "missing value")
    plan_1 = validate_scoped_plan({"operation": "replace_cells", "dataset_id": "dataset_1", "changes": [
        {"row": 2, "column": 2, "before": "0.5", "after": "1"}
    ]}, fractional, source)
    assert "CHANGES" in reviewed_python_code(plan_1)
    with pytest.raises(PermissionError):
        execute_approved_scoped_plan(plan_1, fractional, source, tmp_path / "blocked", "researcher", "no")
    result_1 = execute_approved_scoped_plan(plan_1, fractional, source, tmp_path / "step1",
                                            "researcher", APPROVAL_PHRASE)
    working = Path(result_1["output"])
    assert result_1["all_other_cells_preserved"] is True
    with working.open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[1][1:] == ["1", ""]
    assert rows[2][1:] == ["-3", "4"]
    assert source.read_bytes() == original
    # The second finding must operate against version 1, not the original.
    plan_2 = validate_scoped_plan({"operation": "replace_cells", "dataset_id": "dataset_1", "changes": [
        {"row": 2, "column": 3, "before": "", "after": "0"}
    ]}, missing, working)
    result_2 = execute_approved_scoped_plan(plan_2, missing, working, tmp_path / "step2",
                                            "researcher", APPROVAL_PHRASE)
    with Path(result_2["output"]).open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[1][1:] == ["1", "0"]
    assert rows[2][1:] == ["-3", "4"]
    assert source.read_bytes() == original


def test_cross_finding_changes_and_stale_plans_are_rejected(tmp_path):
    source = sample(tmp_path)
    fractional = find(catalog(source), "fractional count")
    with pytest.raises(ValueError, match="outside this finding"):
        validate_scoped_plan({"operation": "replace_cells", "dataset_id": "dataset_1", "changes": [
            {"row": 3, "column": 2, "before": "-3", "after": "3"}
        ]}, fractional, source)
    with pytest.raises(ValueError, match="dataset differs"):
        validate_scoped_plan({"operation": "replace_cells", "dataset_id": "dataset_2", "changes": [
            {"row": 2, "column": 2, "before": "0.5", "after": "1"}
        ]}, fractional, source)
    validated = validate_scoped_plan({"operation": "replace_cells", "dataset_id": "dataset_1", "changes": [
        {"row": 2, "column": 2, "before": "0.5", "after": "1"}
    ]}, fractional, source)
    source.write_bytes(source.read_bytes().replace(b"0.5", b"0.6"))
    with pytest.raises(ValueError, match="Original cell mismatch"):
        execute_approved_scoped_plan(validated, fractional, source, tmp_path / "stale", "researcher", APPROVAL_PHRASE)


def test_unverified_gemma_finding_does_not_authorize_edits(tmp_path):
    source = sample(tmp_path)
    run = {"proposal": {"hypothesis": "Check a rare value", "method": "Inspect column"},
           "result": {"summary": "Candidate", "findings": [
               {"title": "Possible issue", "evidence": [{"dataset_id": "dataset_1", "csv_row": 2, "column": 2}]}]},
           "verification": {"status": "not_verified"}}
    items = build_cleaning_findings({"dataset_baselines": {"dataset_1": {"checks": {}}}},
                                     [source], [source.name], [run])
    gemma = find(items, "Possible issue")
    assert not gemma["actionable"]
    with pytest.raises(ValueError, match="discussion-only"):
        validate_scoped_plan({"operation": "replace_cells", "dataset_id": "dataset_1",
                              "changes": [{"row": 2, "column": 2, "before": "0.5", "after": "1"}]}, gemma, source)


def test_single_header_change_cannot_rename_other_columns(tmp_path):
    source = sample(tmp_path)
    source.write_text(source.read_text().replace("Taxon,Sample_A", ",Sample_A"))
    finding = find(catalog(source), "has no header name")
    with pytest.raises(ValueError, match="outside this finding"):
        validate_scoped_plan({"operation": "rename_headers", "dataset_id": "dataset_1",
                              "changes": [{"position": 2, "before": "Sample_A", "after": "A"}]}, finding, source)
    plan = validate_scoped_plan({"operation": "rename_headers", "dataset_id": "dataset_1",
                                 "changes": [{"position": 1, "before": "", "after": "Taxon"}]}, finding, source)
    assert plan["source_sha256"] == file_sha256(source)


def test_original_fixture_qc_results_and_gemma_check_are_selectable():
    from healthcare_data_quality.ingestion import inspect_headers
    from healthcare_data_quality.baseline_audit import run_baseline_audit
    source = Path(__file__).resolve().parent.parent / "data" / "dirtyL7PVDforXav.csv"
    if not source.is_file():
        pytest.skip("Optional local fixture is not included in source-only deployments")
    audit = {"dataset_baselines": {"dataset_1": run_baseline_audit(source)}}
    check = {"proposal": {"hypothesis": "Check blank controls"},
             "result": {"summary": "Checked Blank controls for non-zero counts", "findings": []},
             "verification": {"status": "source_cells_verified_and_output_reproduced"}}
    findings = build_cleaning_findings(audit, [source], [source.name], [check])
    assert len([f for f in findings if f["source"] == "QC"]) == 9
    assert len([f for f in findings if f["source"] == "Gemma check"]) == 1
    assert not findings[-1]["actionable"]
    assert len(find(findings, "missing values")["evidence"]) == 618


def test_local_gemma_prompt_is_scoped_to_selected_finding(monkeypatch):
    import json
    from healthcare_data_quality.cleaning_workspace import ask_gemma_about_cleaning
    reply = {"reply": "Review the missing data policy before replacing anything.",
             "operation": "none", "dataset_id": "", "changes": []}

    def fake_query(prompt, **options):
        assert "ONLY SELECTED FINDING" in prompt
        assert "Check 1" in prompt
        assert "never treat NA taxonomy ranks as invalid" in prompt
        assert "only" in prompt.lower()
        assert options["json_schema"]
        return json.dumps(reply)

    monkeypatch.setattr("healthcare_data_quality.cleaning_workspace.query_local_model", fake_query)
    assert ask_gemma_about_cleaning("What does this mean?", [], [], [],
                                    selected_finding={"title": "Check 1"}) == reply
