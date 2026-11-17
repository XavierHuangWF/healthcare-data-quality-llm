"""Regression: NA is a valid rank position, not a data-quality problem."""
import csv
from pathlib import Path

from healthcare_data_quality.baseline_audit import run_baseline_audit
from healthcare_data_quality.taxonomy_validation import inspect_taxonomy_structure
from healthcare_data_quality.finding_presentation import baseline_issues
from healthcare_data_quality.workflow import build_investigation_objective_ledger


def make_matrix(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "taxa.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows([
            ["Taxonomy", "ABC1", "ABC2"],
            *([taxonomy, "1", "0"] for taxonomy in rows),
        ])
    return path


def test_multiple_na_at_any_rank_still_gives_valid_seven_levels(tmp_path):
    path = make_matrix(tmp_path, [
        "NA;p__Firmicutes;c__Bacilli;o__Lactobacillales;f__NA;g__NA;s__NA",
        "NA;NA;NA;NA;NA;NA;NA",
        "Bacteria;Actinobacteria;Actinobacteria;Micrococcales;NA;NA;NA",
        "k__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;f__Family;g__Genus;s__Species",
    ])
    assert inspect_taxonomy_structure(path)["incomplete_taxonomy"] == []
    checks = run_baseline_audit(path)["checks"]
    assert "find_unclassified_species" not in checks
    assert checks["inspect_taxonomy_structure"]["incomplete_count"] == 0
    issues = baseline_issues({"checks": checks}, source_path=path)
    assert not any(issue.get("type") == "taxonomy_structure" for issue in issues)
    assert not any("unassigned species" in issue["title"].lower() for issue in issues)
    ledger = build_investigation_objective_ledger(
        {"dataset_1": path}, {"dataset_1": {"checks": checks, "verified_contexts": []}}, []
    )
    assert not any(o["objective_id"].endswith(":decision:taxonomy_mapping_policy")
                       for o in ledger["objectives"])


def test_only_non_seven_rank_counts_are_structural_errors(tmp_path):
    path = make_matrix(tmp_path, [
        "NA;NA;NA;NA;NA;NA",  # six slots
        "NA;NA;NA;NA;NA;NA;NA;NA", # eight slots
        "NA;NA;NA;NA;NA;NA;NA", # exactly seven: valid
    ])
    report = inspect_taxonomy_structure(path)
    assert report["incomplete_taxonomy"] == [
        {"record": 2, "level_count": 6},
        {"record": 3, "level_count": 8},
    ]
    issues = baseline_issues({"checks": {
        "inspect_taxonomy_structure": {"status": "completed", "observations": [report]},
    }}, source_path=path)
    structure = next(issue for issue in issues if issue["type"] == "taxonomy_structure")
    assert [entry["CSV row"] for entry in structure["evidence"]] == [2, 3]
    assert [entry["Levels found"] for entry in structure["evidence"]] == [6, 8]
    assert "NA is a valid value" in structure["explanation"]


def test_legacy_cached_unassigned_na_is_not_rendered(tmp_path):
    path = make_matrix(tmp_path, ["NA;NA;NA;NA;NA;NA;NA"])
    legacy = {"checks": {
        "find_unclassified_species": {"status": "completed", "observations": [
            {"record": 2, "original_taxonomy": "NA;NA;NA;NA;NA;NA;NA", "species_value": "NA"}
        ]},
    }}
    assert baseline_issues(legacy, source_path=path) == []
