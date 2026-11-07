
from healthcare_data_quality.r_verification import (
    verify_r_equivalence,
    verify_duplicate_column_draft,
)
r_function = (
    'clean_dataset <- function(df) { '
    'df$status <- "verified"; '
    'return(df) '
    '}'
)

synthetic_csv = "sample_id,value\nS1,10\nS2,20\n"

expected_csv = (
    "sample_id,value,status\n"
    "S1,10,verified\n"
    "S2,20,verified\n"
)

result = verify_r_equivalence(
    r_function, synthetic_csv, expected_csv
)

assert result["status"] == "verified"
assert result["records"] == 2
assert result["columns"] == 3

print("CORRECT R TRANSFORMATION: PASS")

incorrect_csv = (
    "sample_id,value,status\n"
    "S1,10,verified\n"
    "S2,99,verified\n"
)

try:
    verify_r_equivalence(
        r_function, synthetic_csv, incorrect_csv
    )
except ValueError:
    print("INCORRECT R TRANSFORMATION REJECTED: PASS")
else:
    raise AssertionError("Incorrect transformation was accepted")


# Distinct synthetic values reveal reversed column removal.
source_csv = (
    "keep,drop,other\n"
    "KEEP_A,DROP_A,X\n"
    "KEEP_B,DROP_B,Y\n"
)

expected_csv = (
    "keep,other\n"
    "KEEP_A,X\n"
    "KEEP_B,Y\n"
)

correct_removal = (
    "clean_dataset <- function(df) { "
    "df[, -2, drop=FALSE] "
    "}"
)

reversed_removal = (
    "clean_dataset <- function(df) { "
    "df[, -1, drop=FALSE] "
    "}"
)

assert verify_r_equivalence(
    correct_removal, source_csv, expected_csv
)["status"] == "verified"

print("CORRECT COLUMN REMOVAL: PASS")

try:
    verify_r_equivalence(
        reversed_removal, source_csv, expected_csv
    )
except ValueError:
    print("REVERSED COLUMN REMOVAL REJECTED: PASS")
else:
    raise AssertionError("Wrong column was removed")



result = verify_duplicate_column_draft(
    "clean_dataset <- function(df) df[, -106, drop=FALSE]",
    keep_position=105,
    drop_position=106,
)

assert result["status"] == "verified"
assert result["columns"] == 105
assert result["records"] == 2

print("106-COLUMN REMOVAL: PASS")

try:
    verify_duplicate_column_draft(
        "clean_dataset <- function(df) df[, -105, drop=FALSE]",
        keep_position=105,
        drop_position=106,
    )
except ValueError:
    print("105/106 REVERSED REMOVAL REJECTED: PASS")
else:
    raise AssertionError("Wrong column removal accepted")
