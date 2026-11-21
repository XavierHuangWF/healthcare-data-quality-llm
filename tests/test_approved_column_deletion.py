"""Regression tests for the current source-bound, export-only column-deletion API.

The old authorize_column_drop/apply_approved_column_drops entry points were
retired when structural actions moved to the shared named-review workflow.
These tests exercise the real export gate rather than resurrecting that API.
"""
import csv
from io import StringIO

import pytest

from healthcare_data_quality.cleaning.in_memory_cleaning import (
    apply_approved_column_deletions,
    apply_verified_in_memory,
    build_final_download,
    sha256_bytes,
)


def _approved_drop(source: bytes, position: int, *, reviewer: str = "researcher") -> dict:
    """Represent an already named/approved UI decision bound to original CSV."""
    headers = next(csv.reader(StringIO(source.decode("utf-8-sig"))))
    return {
        "position": position,
        "before": headers[position - 1],
        "source_sha256": sha256_bytes(source),
        "reviewer": reviewer,
    }


def _rows(data: bytes) -> list[list[str]]:
    return list(csv.reader(StringIO(data.decode("utf-8-sig"))))


def test_approved_column_is_dropped_only_at_export_after_cell_edit():
    original = b"Taxon,ABb100,ABC104,End\na,0.2,0.5,x\nb,1,99,y\n"
    proposal = {
        "operation": "replace_cells", "dataset_id": "dataset_1",
        "finding_id": "dataset_1:qc:1", "source_sha256": sha256_bytes(original),
        "changes": [{"row": 2, "column": 2, "before": "0.2", "after": "2"}],
    }
    staged, _ = apply_verified_in_memory(original, original, proposal, reviewer="researcher")
    assert _rows(staged)[1] == ["a", "2", "0.5", "x"]
    exported = apply_approved_column_deletions(
        original, staged, [_approved_drop(original, 3)])
    assert _rows(exported) == [
        ["Taxon", "ABb100", "End"], ["a", "2", "x"], ["b", "1", "y"]
    ]
    assert b"ABC104" in staged and original.endswith(b"b,1,99,y\n")
    filename, _, data = build_final_download([("dataset_1", "counts.csv", exported)])
    assert filename.endswith("counts.csv") and data == exported


def test_wrong_source_duplicate_and_header_mismatch_fail_closed():
    original = b"a,b,c\n1,2,3\n"
    approval = _approved_drop(original, 3)
    with pytest.raises(PermissionError, match="Stale"):
        apply_approved_column_deletions(b"a,b,c\n1,2,4\n", original, [approval])
    with pytest.raises(PermissionError, match="duplicate"):
        apply_approved_column_deletions(original, original, [approval, approval])
    with pytest.raises(PermissionError, match="Stale"):
        apply_approved_column_deletions(original, original, [{**approval, "before": "wrong"}])
    with pytest.raises(PermissionError, match="dimensions"):
        apply_approved_column_deletions(original, b"a,b,c,d\n1,2,3,4\n", [approval])


def test_reviewer_authorization_and_column_bounds_are_enforced():
    original = b"Taxon,Count\na,2\n"
    with pytest.raises(PermissionError, match="unapproved"):
        apply_approved_column_deletions(original, original,
                                        [_approved_drop(original, 2, reviewer="")])
    with pytest.raises(PermissionError, match="unapproved"):
        apply_approved_column_deletions(original, original,
                                        [{**_approved_drop(original, 2), "position": True}])
    with pytest.raises(PermissionError, match="every column"):
        apply_approved_column_deletions(original, original,
                                        [_approved_drop(original, 1), _approved_drop(original, 2)])
    assert _rows(apply_approved_column_deletions(
        original, original, [_approved_drop(original, 2)])) == [["Taxon"], ["a"]]


def test_original_positions_remain_stable_for_multiple_drops_and_bom():
    original = b"\xef\xbb\xbfA,B,C,D\n0,1,2,3\n"
    approvals = [_approved_drop(original, 2), _approved_drop(original, 4)]
    output = apply_approved_column_deletions(original, original, approvals)
    assert output.startswith(b"\xef\xbb\xbf")
    assert _rows(output) == [["A", "C"], ["0", "2"]]
    assert apply_approved_column_deletions(original, original, []) == original


def test_export_does_not_modify_source_or_staged_bytes():
    original = b"Taxon,Sample1,Sample2\na,10,20\n"
    staged = bytes(original)
    approval = _approved_drop(original, 3)
    exported = apply_approved_column_deletions(original, staged, [approval])
    assert _rows(exported) == [["Taxon", "Sample1"], ["a", "10"]]
    assert original == staged
