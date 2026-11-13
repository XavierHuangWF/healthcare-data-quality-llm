"""User-facing assertions: exact positions, truthfulness, and complete evidence."""
import csv
from pathlib import Path

from healthcare_data_quality.finding_presentation import (
    baseline_issues,
    header_findings,
    header_issues,
    history_observations,
    readable_objective_title,
)
from healthcare_data_quality.ingestion import inspect_headers


def make_csv(tmp_path: Path, headers: list[str], records: list[list[str]]) -> Path:
    path = tmp_path / "example.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows([headers, *records])
    return path


def test_duplicate_column_data_does_not_imply_duplicate_specimens(tmp_path):
    path = make_csv(tmp_path, ["", "ABC99", "ABC99"], [["x", "1", "1"], ["y", "", ""]])
    findings = header_findings(path, inspect_headers(path))
    assert len(findings) == 1
    issue = findings[0]
    assert "Columns 2 and 3" in issue["title"]
    assert "identical values" in issue["title"]
    assert "All 2 data rows match" in issue["explanation"]
    assert "not that the biological specimens are the same" in issue["explanation"]
    assert "Column 1 has no header name" in header_issues(inspect_headers(path))[0]["title"]


def test_identical_names_with_different_values_not_called_identical(tmp_path):
    path = make_csv(tmp_path, ["ABC99", "ABC99"], [["1", "2"], ["", ""]])
    finding = header_findings(path, inspect_headers(path))[0]
    assert "different values" in finding["title"]
    assert finding["different_rows"] == 1
    assert finding["matching_rows"] == 1


def test_ragged_records_prevent_false_whole_column_equality(tmp_path):
    path = make_csv(tmp_path, ["A", "A"], [["1", "1"], ["2"]])
    report = inspect_headers(path)
    finding = header_findings(path, report)[0]
    assert "comparison is incomplete" in finding["title"]
    assert finding["uncomparable_rows"] == 1
    ragged = header_issues(report)[0]
    assert ragged["rows"] == [{"CSV row": 3, "Fields found": 1, "Expected": 2}]


def test_missing_and_invalid_count_evidence_is_not_truncated():
    baseline = {"checks": {"find_invalid_counts": {"status": "completed", "observations": [
        {"row": i + 2, "column": 105, "sample_id": "ABC99", "taxon": "T", "original_value": "",
         "issue_type": "missing"} for i in range(30)
    ]}}}
    issue = baseline_issues(baseline)[0]
    assert "30 cells contain missing values" == issue["title"]
    assert len(issue["evidence"]) == 30
    assert issue["evidence"][-1]["CSV row"] == 31
    assert issue["evidence"][-1]["Column"] == 105


def test_other_issue_categories_are_friendly_and_evidence_is_preserved():
    baseline = {"checks": {
        "find_duplicate_ids": {"status": "completed", "observations": [
            {"field": "SampleID", "value": "s2", "records": [3, 5]}]},
        "find_duplicate_conflicts": {"status": "completed", "observations": [
            {"sample_id": "s2", "field": "Sex", "values_by_record": {3: "A", 5: "B"}}]},
        "find_duplicate_taxa": {"status": "completed", "observations": [
            {"taxon": "taxon", "records": [2, 4], "counts_identical": False}]},
        "find_unclassified_species": {"status": "completed", "observations": [
            {"record": 2, "original_taxonomy": "unknown", "species_value": "unknown"}]},
        "inspect_taxonomy_structure": {"status": "completed", "observations": [
            {"incomplete_taxonomy": [{"record": 6, "level_count": 6}], "alternate_delimiters": [7]}]},
        "inspect_taxonomy_naming": {"status": "completed", "counts_by_style": {
            "unprefixed": 2, "mixed": 1}, "observations": [{"unprefixed": [2, 3], "mixed": [4]}]},
    }}
    issues = baseline_issues(baseline)
    titles = " ".join(i["title"] for i in issues)
    assert "identifier values" in titles
    assert "metadata fields" in titles
    assert "taxon labels" in titles
    assert "species" in titles
    assert "classification" in titles
    assert "taxonomy separator" in titles
    assert "naming formats" in titles
    assert "1 row uses" in titles
    assert next(i for i in issues if "metadata fields" in i["title"])["evidence"] == [
        {"Sample identifier": "s2", "Conflicting field": "Sex", "CSV row": 3, "Recorded value": "A"},
        {"Sample identifier": "s2", "Conflicting field": "Sex", "CSV row": 5, "Recorded value": "B"},
    ]


def test_objective_and_history_titles_are_presentable():
    assert readable_objective_title({"objective_id": "dataset_1:compare_columns:105:106",
                                    "title": "Compare duplicate-header columns 105 and 106"}) == (
        "Compare the values in Columns 105 and 106")
    assert readable_objective_title({"title": "Measure find_invalid_counts"}) == "Missing or invalid counts"
    assert history_observations("inspect_columns", [
        {"position": 105, "name": "ABC99", "missing": 4, "nonmissing": 154, "distinct_count": 10}
    ]) == [{"Column": 105, "Column name": "ABC99", "Filled values": 154,
            "Missing values": 4, "Distinct values": 10}]


def test_whitespace_only_header_is_reported_as_unnamed(tmp_path):
    path = make_csv(tmp_path, [" ", "X"], [["a", "b"]])
    details = inspect_headers(path)
    assert details["blank_header_positions"] == [1]
    assert header_issues(details)[0]["title"] == "Column 1 has no header name"


def test_taxonomy_shows_original_classification_and_unassigned_rank_slots(tmp_path):
    from healthcare_data_quality.taxonomy_validation import inspect_taxonomy_structure

    source = make_csv(tmp_path, ["Taxonomy", "Sample A"], [
        ["Bacteria; Actinobacteria; Actinobacteria; Actinomycetales; Actinomycetaceae; Arcanobacterium", "5"],
        ["Bacteria; Actinobacteria; Actinobacteria; Actinomycetales; Actinomycetaceae", "6"],
        ["Bacteria; Actinobacteria (SGB12345678)", "7"],
        ["Bacteria; Actinobacteria; Actinobacteria; Micrococcales; Micrococcaceae; Rothia", "8"],
    ])
    incomplete = inspect_taxonomy_structure(source)["incomplete_taxonomy"]
    issues = baseline_issues({"checks": {"inspect_taxonomy_structure": {
        "status": "completed", "observations": [{"incomplete_taxonomy": incomplete}]
    }}}, source_path=source)
    issue = issues[0]
    assert issue["type"] == "taxonomy_structure"
    assert [r["CSV row"] for r in issue["evidence"]] == [2, 3, 4, 5]
    assert [r["Ranks not provided (provisional)"] for r in issue["evidence"]] == [
        "Species", "Genus, Species", "Class, Order, Family, Genus, Species", "Species"
    ]
    assert issue["evidence"][0]["Original classification (verbatim)"].endswith("Arcanobacterium")
    first = issue["details"][0]
    assert first["rank_table"][5]["Original entry (provisional rank)"] == "Arcanobacterium"
    assert first["rank_table"][6]["Original entry (provisional rank)"] == "Not provided"
    assert "not been confirmed" in first["placement_note"]
    assert issue["details"][2]["rank_table"][1]["Original entry (provisional rank)"] == (
        "Actinobacteria (SGB12345678)"
    )
    assert issue["details"][2]["rank_table"][2]["Status"] == "Not provided in source"


def test_taxonomy_prefixes_identify_missing_internal_ranks(tmp_path):
    source = make_csv(tmp_path, ["Taxonomy", "Sample"], [
        ["k__Bacteria; p__Actinobacteria; g__Rothia; s__mucilaginosa", "1"]
    ])
    issue = baseline_issues({"checks": {"inspect_taxonomy_structure": {
        "status": "completed", "observations": [{"incomplete_taxonomy": [
            {"record": 2, "level_count": 4}
        ]}]
    }}}, source_path=source)[0]
    assert issue["evidence"][0]["Ranks not provided (provisional)"] == "Class, Order, Family"
    assert issue["details"][0]["rank_table"][5]["Original entry (provisional rank)"] == "g__Rothia"
    assert "prefixes identify" in issue["details"][0]["placement_note"]


def test_taxonomy_legacy_audit_without_source_does_not_fabricate_original():
    baseline = {"checks": {"inspect_taxonomy_structure": {
        "status": "completed", "observations": [{"incomplete_taxonomy": [
            {"record": 3, "level_count": 6}
        ]}]
    }}}
    issue = baseline_issues(baseline)[0]
    assert issue["evidence"][0]["Original classification (verbatim)"] == "Source text unavailable"
    assert issue["details"][0]["original"] is None
    assert "exact positions unavailable" in issue["evidence"][0]["Ranks not provided (provisional)"]
