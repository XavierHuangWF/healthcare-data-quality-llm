"""Regression coverage for ragged counts, taxonomy NA, and UI findings."""

import csv
from pathlib import Path

from healthcare_data_quality.validation.count_validation import find_invalid_counts
from healthcare_data_quality.ui.finding_presentation import baseline_issues
from healthcare_data_quality.validation.taxonomy_validation import (
    inspect_taxonomy_naming, inspect_taxonomy_structure,
)


def write_csv(tmp_path: Path, headers: list[str], rows: list[list[str]]) -> Path:
    path = tmp_path / "audit_regression.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows([headers, *rows])
    return path


def test_short_rows_report_missing_fields_and_extra_fields_never_crash(tmp_path):
    path = write_csv(tmp_path, ["Taxonomy", "A", "B"], [
        ["taxon", "7"],
        ["taxon2", "4", "6", "not-a-labeled-count"],
        [],
    ])
    issues = find_invalid_counts(path)
    assert [(r["row"], r["column"], r["issue_type"]) for r in issues] == [
        (2, 3, "missing"),
        (3, 4, "unexpected_column"),
        (4, 2, "missing"),
        (4, 3, "missing"),
    ]
    assert issues[1]["sample_id"] is None


def test_no_valid_header_returns_empty_evidence(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    assert find_invalid_counts(path) == []


def test_na_occupies_a_rank_and_does_not_create_mixed_naming(tmp_path):
    path = write_csv(tmp_path, ["Taxonomy", "A"], [
        ["k__Bacteria;p__Firmicutes;NA;o__Order;f__Fam;NA;s__species", "1"],
        ["Bacteria;NA;Class;Order;NA;Genus;NA", "2"],
        ["NA;NA;NA;NA;NA;NA;NA", "3"],
        ["d__Bacteria;p__Firmicutes;c__Class;o__Order;f__Fam;g__Genus;s__species", "4"],
        ["k__Bacteria;p__Firmicutes;c__Class;o__Order;NA;g__Genus", "5"],
    ])
    assert inspect_taxonomy_structure(path)["incomplete_taxonomy"] == [
        {"record": 6, "level_count": 6}
    ]
    naming = inspect_taxonomy_naming(path)
    assert naming["fully_prefixed"] == [2, 5]
    assert naming["unprefixed"] == [3]
    assert naming["not_assessable"] == [4]
    assert naming["mixed"] == []


def test_legacy_species_finding_is_suppressed_and_naming_stays_separate():
    baseline = {"checks": {
        "find_unclassified_species": {"status": "completed", "observations": [
            {"record": 2, "original_taxonomy": "A;B;C;D;E;F;NA", "species_value": "NA"},
        ]},
        "inspect_taxonomy_naming": {"status": "completed", "counts_by_style": {
            "fully_prefixed": 2, "unprefixed": 1, "not_assessable": 1,
        }, "observations": [{
            "fully_prefixed": [2, 3], "unprefixed": [4], "not_assessable": [5]
        }]},
    }}
    issues = baseline_issues(baseline)
    # Legacy cached audits may contain species observations. They should
    # never surface as issues, since NA is a valid rank value.
    assert not any(r.get("type") == "unclassified_species" for r in issues)
    assert not any("unassigned species" in r["title"].lower() for r in issues)
    assert len([r for r in issues if "naming formats" in r["title"]]) == 1
    naming = next(r for r in issues if "naming formats" in r["title"])
    assert [r["CSV row"] for r in naming["evidence"]] == [4]


def test_all_na_style_does_not_create_unnecessary_policy_decision(tmp_path):
    from healthcare_data_quality.workflow import build_investigation_objective_ledger
    path = write_csv(tmp_path, ["Taxonomy", "A"], [["NA;NA;NA;NA;NA;NA;NA", "1"]])
    baseline = {"verified_contexts": [], "checks": {
        "inspect_taxonomy_naming": {
            "status": "completed", "observations": [{
                "unprefixed": [2], "not_assessable": [3]
            }], "counts_by_style": {"unprefixed": 1, "not_assessable": 1},
        }
    }}
    ledger = build_investigation_objective_ledger({"dataset_1": path}, {"dataset_1": baseline}, [])
    assert not [o for o in ledger["objectives"]
                if o["objective_id"].endswith(":decision:taxonomy_naming_convention")]
