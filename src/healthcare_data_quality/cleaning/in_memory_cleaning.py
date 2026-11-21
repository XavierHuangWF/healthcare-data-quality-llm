"""Session-only staging for independently verified, finding-scoped CSV edits.

No derived CSV path is created. The original uploaded bytes and current staged
bytes stay separate. Each final, human-approved merge checks every coordinate
and reconstructs an in-memory CSV, then verifies all unchanged values.
"""
from __future__ import annotations

import csv
import hashlib
from decimal import Decimal, InvalidOperation
from io import BytesIO, StringIO
from zipfile import ZipFile, ZIP_DEFLATED


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _records(data: bytes) -> list[list[str]]:
    if not isinstance(data, bytes) or not data or b"\x00" in data:
        raise ValueError("The staged CSV bytes are invalid")
    try:
        rows = list(csv.reader(StringIO(data.decode("utf-8-sig"), newline=""), strict=True))
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("The staged CSV is not valid UTF-8 CSV") from exc
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("A ragged or empty CSV cannot be staged")
    return rows


def assess_verified_in_memory(original: bytes, staged: bytes, insertion: dict) -> dict:
    """Classify a verified R plan against the *latest in-memory* dataset.

    The original CSV's SHA and before-values remain the evidence anchor. An
    already satisfied replacement is idempotent; a different staged value is
    a genuine conflict, not permission to overwrite another reviewed edit.
    This function never changes data and is suitable for displaying a preview.
    """
    if not isinstance(insertion, dict) or insertion.get("operation") not in {"replace_cells", "rename_headers"}:
        raise ValueError("Unsupported staged operation")
    if sha256_bytes(original) != insertion.get("source_sha256"):
        raise PermissionError("The original uploaded CSV differs from the verified R plan")
    original_rows, current = _records(original), _records(staged)
    if (len(current) != len(original_rows)
            or any(len(a) != len(b) for a, b in zip(current, original_rows))):
        raise PermissionError("The staged dataset dimensions have changed")
    changes = insertion.get("changes")
    from .finding_cleaning import MAX_CELL_CHANGES
    if not isinstance(changes, list) or not 1 <= len(changes) <= MAX_CELL_CHANGES:
        raise ValueError(f"Expected 1–{MAX_CELL_CHANGES} verified edits")
    to_apply, already_satisfied, conflicts = [], [], []
    seen = set()
    for change in changes:
        if not isinstance(change, dict):
            raise ValueError("Invalid verified change")
        is_header = insertion["operation"] == "rename_headers"
        csv_row = 1 if is_header else change.get("row")
        col = change.get("position") if is_header else change.get("column")
        if type(csv_row) is not int or type(col) is not int:
            raise ValueError("The proposed edit has an invalid coordinate")
        r, c = csv_row - 1, col - 1
        before, after = change.get("before"), change.get("after")
        if (r < 0 or r >= len(current) or c < 0 or c >= len(current[0])
                or (r, c) in seen or not isinstance(before, str)
                or not isinstance(after, str) or len(after) > 2048
                or '\x00' in after or before == after):
            raise ValueError("The proposed edit has an invalid or duplicate coordinate/value")
        if original_rows[r][c] != before:
            raise PermissionError(
                f"The verified original value differs at CSV row {csv_row}, column {col}")
        seen.add((r, c))
        record = {
            "csv_row": csv_row, "column": col,
            "original": before, "current": current[r][c], "proposed": after,
            "status": "",
        }
        if current[r][c] == after:
            record["status"] = "already_resolved"
            already_satisfied.append(record)
        elif current[r][c] == before:
            record["status"] = "ready_to_apply"
            to_apply.append(record)
        else:
            record["status"] = "conflict"
            conflicts.append(record)
    if insertion["operation"] == "rename_headers":
        target_headers = list(current[0])
        for record in to_apply + conflicts:
            target_headers[record["column"] - 1] = record["proposed"]
        for record in to_apply + conflicts:
            if not record["proposed"].strip():
                raise PermissionError("Cannot apply an empty replacement header")
            if target_headers.count(record["proposed"]) != 1:
                raise PermissionError("Replacement header conflicts with another column")
    return {
        "source_sha256": sha256_bytes(original),
        "staged_sha256": sha256_bytes(staged),
        "operation": insertion["operation"],
        "ready_to_apply": to_apply,
        "already_resolved": already_satisfied,
        "conflicts": conflicts,
    }


def _rebase_expected_output_sha256(review: dict) -> str:
    """Bind a rebased R proof to the exact authorized current-value changes."""
    buf = StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["kind", "csv_row", "csv_column", "value"])
    kind = "header" if review["operation"] == "rename_headers" else "cell"
    for record in review["ready_to_apply"] + review["conflicts"]:
        writer.writerow([kind, record["csv_row"], record["column"], record["proposed"]])
    return sha256_bytes(buf.getvalue().encode("utf-8"))


def apply_verified_in_memory(original: bytes, staged: bytes, insertion: dict, *,
                             reviewer: str, approved_rebase: dict | None = None) -> tuple[bytes, dict]:
    """Atomically merge verified R output, honoring prior in-memory decisions.

    Matching earlier corrections are skipped. A divergent earlier correction
    requires a *new* isolated R proof against staged values plus an explicit
    conflict-override approval; no silent overwrite is permitted.
    """
    if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 120:
        raise PermissionError("Reviewer identifier is required")
    review = assess_verified_in_memory(original, staged, insertion)
    conflicting = review["conflicts"]
    if conflicting:
        if (not isinstance(approved_rebase, dict)
                or approved_rebase.get("staged_sha256") != review["staged_sha256"]
                or approved_rebase.get("source_sha256") != review["source_sha256"]
                or approved_rebase.get("conflicts") != conflicting
                or approved_rebase.get("verified_r_output_sha256") !=
                _rebase_expected_output_sha256(review)):
            first = conflicting[0]
            raise PermissionError(
                f"CSV row {first['csv_row']}, column {first['column']} now contains "
                f"{first['current']!r}; the verified proposal expects "
                f"{first['original']!r} and requests {first['proposed']!r}. "
                "Review the conflict before overriding an earlier approved finding.")
    # `staged` has already been inspected in its entirety. Always apply from
    # exactly those bytes, not from an obsolete source version.
    new = _records(staged)
    targets = {}
    for record in review["ready_to_apply"] + (conflicting if approved_rebase else []):
        r, c = record["csv_row"] - 1, record["column"] - 1
        assert new[r][c] == record["current"]
        targets[(r, c)] = record["proposed"]
        new[r][c] = record["proposed"]
    if not targets:
        output = staged   # An earlier approved finding already resolved all cells.
    else:
        buffer = StringIO(newline="")
        csv.writer(buffer).writerows(new)
        output = (b"\xef\xbb\xbf" if staged.startswith(b"\xef\xbb\xbf") else b"") + buffer.getvalue().encode("utf-8")
    verified, previous = _records(output), _records(staged)
    if (len(verified) != len(previous)
            or any(len(a) != len(b) for a, b in zip(verified, previous))):
        raise AssertionError("Staged CSV dimensions changed")
    for r, (before_row, after_row) in enumerate(zip(previous, verified)):
        for c, (before, after) in enumerate(zip(before_row, after_row)):
            if after != targets.get((r, c), before):
                raise AssertionError("An unrelated cell changed in the staged dataset")
    return output, {
        "status": "verified_in_memory" if targets else "already_resolved_in_memory",
        "finding_id": insertion.get("finding_id"),
        "operation": insertion["operation"],
        "changes_applied": len(targets),
        "already_resolved": len(review["already_resolved"]),
        "conflicts_overridden": len(conflicting) if approved_rebase else 0,
        "reviewer": reviewer.strip(),
        "source_sha256": review["source_sha256"],
        "previous_staged_sha256": review["staged_sha256"],
        "staged_sha256": sha256_bytes(output),
        "all_other_cells_preserved": True,
    }


def _sum_count_cells(a: str, b: str) -> str:
    """Merge a count field without silently overwriting either measurement."""
    absent = {"", "NA"}
    aa, bb = a.strip().upper(), b.strip().upper()
    if aa in absent and bb in absent:
        return a if a else b
    if aa in absent:
        return b
    if bb in absent:
        return a
    try:
        total = Decimal(a) + Decimal(b)
    except InvalidOperation as exc:
        raise ValueError("Merging requires numeric counts or empty/NA values") from exc
    if not total.is_finite():
        raise ValueError("Cannot merge non-finite counts")
    value = format(total, "f")
    return value.rstrip("0").rstrip(".") if "." in value else value


def apply_approved_row_operations(original: bytes, staged: bytes,
                                  approvals: list[dict]) -> bytes:
    """Apply reviewed row deletion/merge at EXPORT only using original CSV row IDs.

    An edited row stays at its original coordinate in session memory. Row merges
    use the latest staged values and sum count columns, retaining the first row.
    Every dropped original row is bound to the immutable original taxon label.
    """
    original_rows, staged_rows = _records(original), _records(staged)
    if len(original_rows) != len(staged_rows) or any(
            len(a) != len(b) for a, b in zip(original_rows, staged_rows)):
        raise PermissionError("Staged CSV dimensions differ from original row evidence")
    if not isinstance(approvals, list):
        raise ValueError("Row approvals must be a list")
    if not approvals:
        return staged
    output_rows = [list(row) for row in staged_rows]
    touched, dropped = set(), set()
    for approval in approvals:
        if (not isinstance(approval, dict) or
                approval.get("operation") not in {"merge_rows", "drop_rows"} or
                approval.get("source_sha256") != sha256_bytes(original) or
                not isinstance(approval.get("reviewer"), str) or
                not approval["reviewer"].strip()):
            raise PermissionError("Invalid, stale or unapproved row operation")
        changes = approval.get("changes")
        limit = (min(len(original_rows) - 2, 20000)
                 if approval["operation"] == "drop_rows" else 32)
        if not isinstance(changes, list) or not 1 <= len(changes) <= limit:
            raise PermissionError("Invalid row change count")
        for change in changes:
            if not isinstance(change, dict):
                raise PermissionError("Invalid row change")
            row = change.get("row")
            if type(row) is not int or not 2 <= row <= len(original_rows):
                raise PermissionError(f"Invalid original CSV row in approved {approval['operation']} operation: {row!r}")
            if row in touched:
                raise PermissionError(f"CSV row {row} is targeted more than once by approved row operations")
            if change.get("before") != original_rows[row - 1][0]:
                raise PermissionError(
                    f"CSV row {row} no longer matches the approved original row label")
            # Original CSV coordinates remain stable until export. A separately
            # verified cell correction may legitimately update the staged taxon
            # label; this must not invalidate a previously approved row DELETE.
            # Source SHA, original row label and the exact row coordinate still
            # bind the deletion to the immutable uploaded CSV.
            touched.add(row)
            if approval["operation"] == "drop_rows":
                if change.get("after") != "":
                    raise PermissionError("Invalid delete-row payload")
                dropped.add(row)
                continue
            other = change.get("source_row")
            if (type(other) is not int or not 2 <= other <= len(original_rows)
                    or other == row or change.get("after") != "sum_numeric"
                    or original_rows[other - 1][0] != original_rows[row - 1][0]):
                raise PermissionError(
                    f"CSV rows {row} and {other!r} do not match the approved original merge")
            if other in touched:
                raise PermissionError(
                    f"CSV row {other} is targeted more than once by approved row operations")
            # Unlike a deletion, a merge must still represent the same current
            # taxon on BOTH sides. If only one label was edited, require the user
            # to revisit the merge rather than silently combine distinct taxa.
            if staged_rows[row - 1][0] != staged_rows[other - 1][0]:
                raise PermissionError(
                    f"CSV rows {row} and {other} now have different staged taxon "
                    "labels; review this merge before export")
            touched.add(other)
            merged = [staged_rows[row-1][0]]
            for a, b in zip(staged_rows[row-1][1:], staged_rows[other-1][1:]):
                merged.append(_sum_count_cells(a, b))
            output_rows[row-1] = merged
            dropped.add(other)
    if len(dropped) >= len(original_rows) - 1:
        raise PermissionError("Cannot remove every data row")
    kept = [row for i, row in enumerate(output_rows, 1) if i not in dropped]
    buffer = StringIO(newline="")
    csv.writer(buffer).writerows(kept)
    encoded = (b"\xef\xbb\xbf" if staged.startswith(b"\xef\xbb\xbf") else b"") + buffer.getvalue().encode("utf-8")
    if _records(encoded) != kept:
        raise AssertionError("Row export did not reproduce the verified result")
    return encoded


def apply_approved_column_deletions(original: bytes, staged: bytes,
                                    deletions: list[dict], *,
                                    approved_row_operations: list[dict] | None = None,
                                    staged_before_rows: bytes | None = None) -> bytes:
    """Apply explicitly approved physical-column removals only at final export.

    Cell/rename modifications remain in the coordinate-stable staged CSV until
    export. This prevents column-shift bugs when later Agent C edits use the
    original numbered evidence. No source file or in-session intermediate is
    changed. Each retained value is checked against the latest staged input.
    """
    original_rows, staged_rows = _records(original), _records(staged)
    if approved_row_operations:
        if staged_before_rows is None or apply_approved_row_operations(
                original, staged_before_rows, approved_row_operations) != staged:
            raise PermissionError("Exported rows do not match the approved structural operations")
        if any(len(row) != len(original_rows[0]) for row in staged_rows):
            raise PermissionError("Exported CSV column dimensions changed unexpectedly")
    elif (len(original_rows) != len(staged_rows) or
            any(len(a) != len(b) for a, b in zip(original_rows, staged_rows))):
        raise PermissionError("Staged CSV dimensions differ from original evidence")
    if not isinstance(deletions, list):
        raise ValueError("Deletion approvals must be a list")
    if not deletions:
        return staged
    dropped: set[int] = set()
    for approval in deletions:
        if not isinstance(approval, dict):
            raise PermissionError("Invalid column deletion approval")
        pos = approval.get("position")
        if (type(pos) is not int or not 1 <= pos <= len(original_rows[0])
                or pos in dropped or approval.get("source_sha256") != sha256_bytes(original)
                or approval.get("before") != original_rows[0][pos - 1]
                or not isinstance(approval.get("reviewer"), str)
                or not approval["reviewer"].strip()):
            raise PermissionError("Stale, duplicate or unapproved column deletion")
        dropped.add(pos)
    if len(dropped) >= len(original_rows[0]):
        raise PermissionError("Cannot delete every column from the CSV")
    keep = [i for i in range(len(original_rows[0])) if i + 1 not in dropped]
    output_rows = [[row[i] for i in keep] for row in staged_rows]
    buffer = StringIO(newline="")
    csv.writer(buffer).writerows(output_rows)
    output = (b"\xef\xbb\xbf" if staged.startswith(b"\xef\xbb\xbf") else b"") + buffer.getvalue().encode("utf-8")
    observed = _records(output)
    if observed != output_rows or len(observed) != len(staged_rows):
        raise AssertionError("Final column deletion changed an unexpected value or row")
    return output


def build_final_download(dataset_names: list[tuple[str, str, bytes]]) -> tuple[str, str, bytes]:
    """Produce a final download entirely in memory (CSV or multi-dataset ZIP)."""
    if not dataset_names:
        raise ValueError("No datasets available")
    cleaned = []
    for dataset_id, original_name, data in dataset_names:
        _records(data)
        clean_name = original_name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if not clean_name.lower().endswith(".csv"):
            raise ValueError("Only CSV datasets can be exported")
        cleaned.append((f"{dataset_id}_cleaned_{clean_name}", data))
    if len(cleaned) == 1:
        return cleaned[0][0], "text/csv", cleaned[0][1]
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for filename, data in cleaned:
            archive.writestr(filename, data)
    return "cleaned_datasets.zip", "application/zip", output.getvalue()


def review_completion(findings: list[dict], statuses: dict[str, str]) -> tuple[int, int]:
    """Count all findings AND completed no-issue Gemma checks, not only edits."""
    total = len(findings)
    done = sum(statuses.get(f["id"]) in {
        "Applied and verified", "Resolved by earlier approved edit",
        "Approved column deletion for final export",
        "Approved row changes for final export",
        "Partially approved — remaining action rejected",
        "Reviewed — no change",
    } for f in findings)
    return done, total
