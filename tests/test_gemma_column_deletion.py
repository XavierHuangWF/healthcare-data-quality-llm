"""Agent C proposes whole-column deletion; exact scope and export remain deterministic."""
from __future__ import annotations

import csv
import json
from io import StringIO

import pytest

from healthcare_data_quality.cleaning.cleaning_authorization import file_sha256
from healthcare_data_quality.cleaning.cleaning_workspace import ask_gemma_about_cleaning
from healthcare_data_quality.cleaning.finding_cleaning import validate_scoped_plan
from healthcare_data_quality.cleaning.in_memory_cleaning import (
    _records, apply_approved_column_deletions, sha256_bytes,
)


def fixture(tmp_path):
    p = tmp_path / "count.csv"
    p.write_text("Taxon,A,B,C\nr1,1,0.5,3\nr2,4,5,6\n", encoding="utf-8")
    finding = {"id": "dataset_1:qc:1", "dataset_id": "dataset_1",
               "source": "QC", "title": "fractional count",
               "check_key": "find_invalid_counts", "actionable": True,
               "evidence": [{"CSV row": 2, "Column": 3}]}
    return p, finding


def proposed(pos=3, header="B"):
    return {"operation": "drop_column", "dataset_id": "dataset_1",
            "changes": [{"position": pos, "before": header, "after": ""}]}


def authorization(plan):
    item = plan["changes"][0]
    return {"position": item["position"], "before": item["before"],
            "source_sha256": plan["source_sha256"], "reviewer": "Researcher",
            "finding_id": plan["finding_id"]}


def test_gemma_discusses_deletion_and_never_executes_it(monkeypatch):
    received = []
    def fake_query(prompt, *, json_schema, **kwargs):
        received.append((prompt, json_schema))
        return json.dumps({"reply": "I can propose removing all of column 3 (B). Confirm in the UI.",
                           **proposed()})
    monkeypatch.setattr("healthcare_data_quality.cleaning.cleaning_workspace.query_local_model", fake_query)
    response = ask_gemma_about_cleaning("Delete column 3 entirely", [], [],
                                        [{"dataset_id": "dataset_1", "columns": [
                                            {"position": 3, "header": "B"}]}],
                                        selected_finding={"dataset_id": "dataset_1",
                                                          "allowed_cell_positions": [[2, 3]]})
    assert response["operation"] == "drop_column"
    assert "drop_column" in received[0][0]
    assert "drop_column" in received[0][1]["properties"]["operation"]["enum"]


def test_exact_column_deletion_keeps_original_and_other_staged_edits(tmp_path):
    source, finding = fixture(tmp_path)
    original = source.read_bytes()
    plan = validate_scoped_plan(proposed(), finding, source)
    assert plan["source_sha256"] == file_sha256(source)
    assert _records(original)[0][2] == "B"
    staged = original.replace(b"r2,4,5,6", b"r2,4,5,999")
    updated = apply_approved_column_deletions(original, staged, [authorization(plan)])
    assert _records(updated) == [["Taxon", "A", "C"], ["r1", "1", "3"], ["r2", "4", "999"]]
    assert source.read_bytes() == original
    assert _records(staged)[0] == ["Taxon", "A", "B", "C"]


@pytest.mark.parametrize("proposal,expected", [
    (proposed(2, "A"), "not evidenced"),
    (proposed(3, "A"), "header mismatch"),
    ({"operation": "drop_column", "dataset_id": "dataset_1", "changes": []}, "exactly one"),
    ({"operation": "drop_column", "dataset_id": "dataset_1", "changes": [
        {"position": 3, "before": "B", "after": "NA"}]}, "unsupported"),
])
def test_reject_unverified_or_mismatched_column(tmp_path, proposal, expected):
    source, finding = fixture(tmp_path)
    with pytest.raises(ValueError, match=expected):
        validate_scoped_plan(proposal, finding, source)


def test_export_blocks_duplicate_stale_and_unreviewed_deletions(tmp_path):
    source, finding = fixture(tmp_path)
    original = source.read_bytes()
    plan = validate_scoped_plan(proposed(), finding, source)
    entry = authorization(plan)
    with pytest.raises(PermissionError, match="Stale, duplicate"):
        apply_approved_column_deletions(original, original, [{**entry, "reviewer": ""}])
    with pytest.raises(PermissionError, match="Stale, duplicate"):
        apply_approved_column_deletions(original, original, [entry, entry])
    with pytest.raises(PermissionError, match="Stale, duplicate"):
        apply_approved_column_deletions(original, original, [{**entry, "source_sha256": "bad"}])
    with pytest.raises(PermissionError, match="Staged CSV dimensions"):
        apply_approved_column_deletions(original, b"Taxon,A,B\nr1,1,0.5\nr2,4,5\n", [entry])
    assert apply_approved_column_deletions(original, original, []) == original


def test_column_delete_must_be_finding_scoped(tmp_path):
    source, finding = fixture(tmp_path)
    with pytest.raises(ValueError, match="discussion-only"):
        validate_scoped_plan(proposed(), {**finding, "actionable": False}, source)
    with pytest.raises(ValueError, match="dataset differs"):
        validate_scoped_plan({**proposed(), "dataset_id": "dataset_2"}, finding, source)
