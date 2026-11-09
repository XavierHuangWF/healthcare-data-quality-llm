
import csv
from io import StringIO

from run_workflow import (
    merge_verified_qc_column,
    consolidated_qc_release_status,
)

# Synthetic fixtures: no private datasets or saved outputs required.
original = (
    "record_id,count\n"
    "A,5\n"
    "B,\n"
    "C,-2\n"
)

missing_candidate = (
    "record_id,count,missing_flag\n"
    "A,5,COMPLETE\n"
    "B,,MISSING\n"
    "C,-2,COMPLETE\n"
)

negative_candidate = (
    "record_id,count,negative_flag\n"
    "A,5,VALID\n"
    "B,,VALID\n"
    "C,-2,INVALID\n"
)

first = merge_verified_qc_column(
    original, original, missing_candidate, "missing_flag"
)

combined = merge_verified_qc_column(
    original, first, negative_candidate, "negative_flag"
)

source_rows = list(csv.reader(StringIO(original)))
combined_rows = list(csv.reader(StringIO(combined)))

assert len(combined_rows) == len(source_rows) == 4
assert len(combined_rows[0]) == 4
assert combined_rows[0][-2:] == [
    "missing_flag",
    "negative_flag",
]

for source_row, combined_row in zip(source_rows, combined_rows):
    assert combined_row[:2] == source_row

assert [row[-2] for row in combined_rows[1:]] == [
    "COMPLETE", "MISSING", "COMPLETE"
]
assert [row[-1] for row in combined_rows[1:]] == [
    "VALID", "VALID", "INVALID"
]

# Reject an output that modifies an original cell.
tampered = negative_candidate.replace("C,-2,INVALID", "C,0,INVALID")

try:
    merge_verified_qc_column(
        original, first, tampered, "negative_flag"
    )
except ValueError:
    pass
else:
    raise AssertionError("Tampered source cell was accepted")

# Reject a duplicate QC column.
try:
    merge_verified_qc_column(
        original, combined, missing_candidate, "missing_flag"
    )
except ValueError:
    pass
else:
    raise AssertionError("Duplicate QC column was accepted")

# Release only when all authorized operations pass verification.
assert consolidated_qc_release_status(
    2, 2, "original_hash", "original_hash"
) == "ready"

# One of two authorized operations failed.
assert consolidated_qc_release_status(
    2, 1, "original_hash", "original_hash"
) == "verification_incomplete"

# Neither authorized operation passed.
assert consolidated_qc_release_status(
    2, 0, "original_hash", "original_hash"
) == "verification_incomplete"

# Never release more operations than were authorized.
assert consolidated_qc_release_status(
    1, 2, "original_hash", "original_hash"
) == "verification_incomplete"

# Original source changed during execution.
assert consolidated_qc_release_status(
    2, 2, "original_hash", "changed_hash"
) == "source_changed"

# No execution was authorized.
assert consolidated_qc_release_status(
    0, 0, "original_hash", "original_hash"
) == "not_authorized"

print("ALL-OR-NOTHING RELEASE GATE: PASS")
print("PARTIAL VERIFICATION BLOCKED: PASS")
print("SOURCE MUTATION BLOCKED: PASS")
print("UNAUTHORIZED RELEASE BLOCKED: PASS")

print("TWO-OPERATION CUMULATIVE MERGE: PASS")
print("ORIGINAL CELL PRESERVATION: PASS")
print("TAMPER REJECTION: PASS")
print("DUPLICATE QC COLUMN REJECTION: PASS")
print("SYNTHETIC RESULT: 3 records, 4 columns")
