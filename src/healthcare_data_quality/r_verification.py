
import csv
from io import StringIO

from .r_sandbox import run_r_sandbox


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
