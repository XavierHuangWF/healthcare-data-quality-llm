"""No working CSV is written until the user downloads the fully reviewed result."""
from io import BytesIO, StringIO
from pathlib import Path
import csv
from zipfile import ZipFile

import pytest

from healthcare_data_quality.in_memory_cleaning import (
    apply_verified_in_memory, build_final_download, review_completion, sha256_bytes,
)


def _plan(original: bytes, *, row=2, col=2, before="0.2", after="NA") -> dict:
    return {
        "operation": "replace_cells", "dataset_id": "dataset_1", "finding_id": "qc:1",
        "source_sha256": sha256_bytes(original),
        "changes": [{"row": row, "column": col, "before": before, "after": after}],
    }


def test_sequential_approved_edits_live_only_in_memory(tmp_path):
    original = b"Taxon,Count,Other\na,0.2,\nb,1,2\n"
    files_before = set(tmp_path.iterdir())
    stage1, record1 = apply_verified_in_memory(
        original, original, _plan(original), reviewer="xavier")
    assert record1["status"] == "verified_in_memory"
    assert b"a,NA," in stage1 and b"a,0.2," in original
    stage2, record2 = apply_verified_in_memory(
        original, stage1, _plan(original, row=3, col=3, before="2", after="5"), reviewer="xavier")
    assert b"a,NA," in stage2 and b"b,1,5" in stage2
    assert record2["previous_staged_sha256"] == record1["staged_sha256"]
    assert set(tmp_path.iterdir()) == files_before


def test_attempt_to_overwrite_earlier_approved_cell_fails_closed():
    original = b"Taxon,Count\na,0.2\n"
    stage1, _ = apply_verified_in_memory(original, original, _plan(original), reviewer="r")
    with pytest.raises(PermissionError, match="earlier approved finding"):
        apply_verified_in_memory(original, stage1, _plan(original, after="0"), reviewer="r")


def test_original_hash_and_scope_are_bound():
    original = b"Taxon,Count\na,0.2\n"
    with pytest.raises(PermissionError, match="original uploaded"):
        apply_verified_in_memory(b"Taxon,Count\na,0.3\n", original,
                                 _plan(original), reviewer="r")
    bad = _plan(original, col=99)
    with pytest.raises(ValueError, match="invalid or duplicate"):
        apply_verified_in_memory(original, original, bad, reviewer="r")
    with pytest.raises(PermissionError, match="Reviewer"):
        apply_verified_in_memory(original, original, _plan(original), reviewer="")


def test_approved_header_rename_stays_in_memory():
    original = b",Count\na,0.2\n"
    plan = {"operation": "rename_headers", "dataset_id": "dataset_1", "finding_id": "header",
            "source_sha256": sha256_bytes(original),
            "changes": [{"position": 1, "before": "", "after": "Taxon"}]}
    staged, result = apply_verified_in_memory(original, original, plan, reviewer="r")
    assert staged.startswith(b"Taxon,Count")
    assert result["all_other_cells_preserved"]


def test_review_gate_counts_no_issue_gemma_checks():
    catalog = [{"id": "qc1"}, {"id": "qc2"}, {"id": "gemma:no_issue"}]
    assert review_completion(catalog, {"qc1": "Applied and verified"}) == (1, 3)
    status = {"qc1": "Applied and verified", "qc2": "Reviewed — no change",
              "gemma:no_issue": "Reviewed — no change"}
    assert review_completion(catalog, status) == (3, 3)


def test_final_download_assembled_in_memory_for_one_or_multiple_csvs(tmp_path):
    first = b"Taxon,Count\na,NA\n"
    second = b"Taxon,Count\nb,2\n"
    filename, mime, bytes_out = build_final_download([("dataset_1", "sample.csv", first)])
    assert filename == "dataset_1_cleaned_sample.csv" and mime == "text/csv"
    assert bytes_out == first
    filename, mime, bundle = build_final_download([
        ("dataset_1", "sample.csv", first), ("dataset_2", "other.csv", second)])
    assert filename == "cleaned_datasets.zip" and mime == "application/zip"
    with ZipFile(BytesIO(bundle)) as archive:
        assert archive.read("dataset_1_cleaned_sample.csv") == first
        assert archive.read("dataset_2_cleaned_other.csv") == second
    assert not list(tmp_path.iterdir())
