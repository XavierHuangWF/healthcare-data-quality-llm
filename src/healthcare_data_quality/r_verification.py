
import csv
from io import StringIO

from .r_sandbox import run_r_sandbox

def inspect_r_transformation(
    r_function: str,
    input_csv: str,
) -> dict:
    """Run generated R in isolation and inspect its output structure."""

    def parse_csv(content: str) -> list[list[str]]:
        rows = list(csv.reader(StringIO(content), strict=True))

        if not rows or not rows[0]:
            raise ValueError("CSV has no valid header")

        width = len(rows[0])

        if any(len(row) != width for row in rows):
            raise ValueError("CSV contains inconsistent row widths")

        return rows

    source = parse_csv(input_csv)

    output_csv = run_r_sandbox(
        r_function=r_function,
        synthetic_csv=input_csv,
    )

    result = parse_csv(output_csv)

    return {
        "status": "sandbox_executed",
        "semantic_verification": "not_established",
        "input_records": len(source) - 1,
        "output_records": len(result) - 1,
        "input_columns": len(source[0]),
        "output_columns": len(result[0]),
        "headers_changed": source[0] != result[0],
        "content_changed": source != result,
        "output_csv": output_csv,
    }

def verify_exact_cell_edits(
    source_csv: str,
    output_csv: str,
    expected_edits: list[dict],
) -> dict:
    """Verify exact approved CSV cell edits, independently of generated R."""

    def parse(content):
        rows = list(csv.reader(StringIO(content), strict=True))
        if not rows or not rows[0]:
            raise ValueError("CSV header missing")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("Inconsistent CSV row widths")
        return rows

    source = parse(source_csv)
    output = parse(output_csv)

    if len(source) != len(output):
        raise ValueError("Record count changed")

    if source[0] != output[0]:
        raise ValueError("Column headers changed")

    expected = {}

    for edit in expected_edits:
        row = edit["row"] - 1  # CSV row includes header
        col = edit["column"] - 1

        if not (1 <= row < len(source)):
            raise ValueError("Invalid CSV row position")
        if not (0 <= col < len(source[0])):
            raise ValueError("Invalid CSV column position")

        position = (row, col)

        if position in expected:
            raise ValueError("Duplicate expected cell edit")

        if source[row][col] != edit["original_value"]:
            raise ValueError("Original value does not match evidence")

        if edit["original_value"] == edit["new_value"]:
            raise ValueError("Expected edit does not change the value")

        expected[position] = edit["new_value"]

    if not expected:
        raise ValueError("No expected edits supplied")

    changed = set()

    for row in range(1, len(source)):
        for col in range(len(source[0])):
            before = source[row][col]
            after = output[row][col]
            position = (row, col)

            if before == after:
                continue

            if position not in expected:
                raise ValueError(
                    f"Unauthorized change at CSV row {row + 1}, "
                    f"column {col + 1}"
                )

            if after != expected[position]:
                raise ValueError(
                    f"Incorrect replacement at CSV row {row + 1}, "
                    f"column {col + 1}"
                )

            changed.add(position)

    if changed != set(expected):
        raise ValueError("Expected changes were not all performed")

    return {
        "status": "verified",
        "changed_cells": len(changed),
        "records_preserved": True,
        "columns_preserved": True,
    }

def verify_qc_flag_column(
    source_csv: str,
    output_csv: str,
    *,
    flag_column: str,
    flagged_csv_rows: set[int],
    flagged_value: str,
    other_value: str,
) -> dict:
    """Check an appended QC column against independently supplied expected rows."""

    def parse(content: str) -> list[list[str]]:
        rows = list(csv.reader(StringIO(content), strict=True))

        if not rows or not rows[0]:
            raise ValueError("Missing CSV header")

        width = len(rows[0])

        if any(len(row) != width for row in rows):
            raise ValueError("Inconsistent CSV row widths")

        return rows

    source = parse(source_csv)
    output = parse(output_csv)

    if len(source) != len(output):
        raise ValueError("Record count changed")

    if flag_column in source[0]:
        raise ValueError("QC column already exists in source")

    if output[0] != source[0] + [flag_column]:
        raise ValueError("Expected exactly one appended QC column")

    if not flagged_csv_rows:
        raise ValueError("No expected flagged rows supplied")

    if any(
        type(row) is not int or row < 2 or row > len(source)
        for row in flagged_csv_rows
    ):
        raise ValueError("Invalid physical CSV row number")

    if flagged_value == other_value:
        raise ValueError("Flag values must differ")

    for csv_row in range(2, len(source) + 1):
        original = source[csv_row - 1]
        modified = output[csv_row - 1]

        if modified[:-1] != original:
            raise ValueError(
                f"Original data changed at CSV row {csv_row}"
            )

        expected = (
            flagged_value
            if csv_row in flagged_csv_rows
            else other_value
        )

        if modified[-1] != expected:
            raise ValueError(
                f"Incorrect QC flag at CSV row {csv_row}: "
                f"expected {expected!r}, got {modified[-1]!r}"
            )

    return {
        "status": "qc_flag_contract_matched",
        "records_preserved": len(source) - 1,
        "original_columns_preserved": len(source[0]),
        "flagged_records": len(flagged_csv_rows),
    }

def verify_qc_contract_from_evidence(
    source_csv: str,
    output_csv: str,
    contract: dict,
    evidence_index: dict,
) -> dict:
    """Verify a QC transformation using independently collected evidence."""

    if not isinstance(contract, dict):
        raise ValueError("Missing verification contract")

    if contract.get("kind") != "flag_rows_from_evidence":
        raise ValueError("Unsupported verification contract")

    evidence_id = contract.get("evidence_id")
    evidence = evidence_index.get(evidence_id)

    if not isinstance(evidence, dict):
        raise ValueError("Verification evidence not found")

    # Initially support only the independently tested
    # taxonomy-completeness evidence structure.
    if (
        evidence.get("tool") != "inspect_taxonomy_structure"
        or contract.get("observation_field") != "incomplete_taxonomy"
        or contract.get("record_field") != "record"
    ):
        raise ValueError("Unsupported evidence structure")

    observation = evidence.get("observation")

    if not isinstance(observation, dict):
        raise ValueError("Invalid evidence observation")

    records = observation.get("incomplete_taxonomy")

    if not isinstance(records, list) or not records:
        raise ValueError("Missing record-level evidence")

    if any(
        not isinstance(item, dict)
        or type(item.get("record")) is not int
        for item in records
    ):
        raise ValueError("Invalid record positions in evidence")

    expected_rows = {
        item["record"]
        for item in records
    }

    result = verify_qc_flag_column(
        source_csv,
        output_csv,
        flag_column=contract["output_column"],
        flagged_csv_rows=expected_rows,
        flagged_value=contract["flagged_value"],
        other_value=contract["other_value"],
    )

    return {
        **result,
        "evidence_id": evidence_id,
        "verification": "independent_evidence_match",
    }

def verify_numeric_qc_from_baseline(
    source_csv: str,
    output_csv: str,
    baseline: dict,
    *,
    conditions: list[dict],
    flag_column: str,
    flagged_value: str,
    other_value: str,
) -> dict:
    """Verify numeric QC flags against independently collected baseline findings."""

    allowed_types = {"negative", "fractional", "missing"}
    criteria = set()

    for condition in conditions:
        issue_type = condition.get("issue_type")
        columns = condition.get("columns")

        if (
            issue_type not in allowed_types
            or not isinstance(columns, list)
            or not columns
            or any(type(c) is not int or c < 1 for c in columns)
        ):
            raise ValueError("Invalid numeric QC condition")

        criteria.update(
            (issue_type, column)
            for column in columns
        )

    if not criteria:
        raise ValueError("No numeric QC criteria specified")

    observations = (
        baseline.get("checks", {})
        .get("find_invalid_counts", {})
        .get("observations")
    )

    if not isinstance(observations, list):
        raise ValueError("Baseline observations unavailable")

    source_rows = list(csv.reader(StringIO(source_csv), strict=True))

    if not source_rows:
        raise ValueError("Empty source CSV")

    expected_rows = set()

    for issue in observations:
        if not isinstance(issue, dict):
            raise ValueError("Invalid baseline observation")

        if (issue.get("issue_type"), issue.get("column")) not in criteria:
            continue

        row = issue.get("row")
        column = issue.get("column")
        original = issue.get("original_value")

        if (
            type(row) is not int
            or type(column) is not int
            or row < 2
            or row > len(source_rows)
            or column < 1
            or column > len(source_rows[0])
        ):
            raise ValueError("Invalid baseline cell position")

        if source_rows[row - 1][column - 1] != original:
            raise ValueError(
                f"Baseline evidence mismatch at CSV row {row}, column {column}"
            )

        expected_rows.add(row)

    if not expected_rows:
        raise ValueError("No records match approved QC criteria")

    result = verify_qc_flag_column(
        source_csv,
        output_csv,
        flag_column=flag_column,
        flagged_csv_rows=expected_rows,
        flagged_value=flagged_value,
        other_value=other_value,
    )

    return {
        **result,
        "verification": "independent_baseline_match",
        "matched_records": len(expected_rows),
        "matched_conditions": len(criteria),
    }

def verify_r_equivalence(
    r_function: str,
    synthetic_csv: str,
    expected_csv: str,
) -> dict:
    if not expected_csv.strip():
        raise ValueError("Expected CSV cannot be empty")

    actual_csv = run_r_sandbox(r_function, synthetic_csv)

    actual = list(csv.reader(StringIO(actual_csv), strict=True))
    expected = list(csv.reader(StringIO(expected_csv), strict=True))

    if not expected or not expected[0]:
        raise ValueError("Expected CSV has no header")

    width = len(expected[0])

    if any(len(row) != width for row in actual + expected):
        raise ValueError("CSV column count mismatch")

    if actual != expected:
        raise ValueError("R transformation output mismatch")

    return {
        "status": "verified",
        "records": len(actual) - 1,
        "columns": width,
    }


def verify_duplicate_column_draft(
    r_function: str,
    keep_position: int,
    drop_position: int,
) -> dict:
    if (
        type(keep_position) is not int
        or type(drop_position) is not int
        or keep_position < 1
        or drop_position < 1
        or keep_position == drop_position
    ):
        raise ValueError("Invalid column positions")

    width = max(keep_position, drop_position)

    if width > 512:
        raise ValueError("Synthetic fixture width limit exceeded")

    headers = [f"col_{i}" for i in range(1, width + 1)]
    rows = [
        [f"R{r}C{c}" for c in range(1, width + 1)]
        for r in range(1, 3)
    ]

    def to_csv(data: list[list[str]]) -> str:
        buffer = StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerows(data)
        return buffer.getvalue()

    source_csv = to_csv([headers, *rows])

    expected_csv = to_csv([
        [value for i, value in enumerate(row, 1)
         if i != drop_position]
        for row in [headers, *rows]
    ])

    return verify_r_equivalence(
        r_function,
        source_csv,
        expected_csv,
    )
