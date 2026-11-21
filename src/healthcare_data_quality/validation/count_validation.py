"""Read-only, position-preserving validation of taxonomy count CSV cells."""

import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path


def find_invalid_counts(path: Path) -> list[dict]:
    """Find invalid sample counts without losing ragged-row evidence.

    Expected columns are taken from the header. Missing trailing fields are
    reported as missing counts; extra fields are reported as unexpected fields
    rather than indexing beyond the header. An empty record is treated as a
    record with every count missing. No values are changed.
    """
    issues: list[dict] = []

    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        headers = next(reader, [])
        if not headers:
            return issues

        for row_number, row in enumerate(reader, start=2):
            taxon = row[0] if row else ""
            for column_number in range(2, len(headers) + 1):
                raw = row[column_number - 1] if column_number <= len(row) else ""
                value = raw.strip()
                if not value:
                    reason = "missing"
                else:
                    try:
                        number = Decimal(value)
                    except InvalidOperation:
                        reason = "non_numeric"
                    else:
                        if not number.is_finite():
                            reason = "non_numeric"
                        elif number < 0:
                            reason = "negative"
                        elif number != number.to_integral_value():
                            reason = "fractional"
                        else:
                            continue
                issues.append({
                    "row": row_number,
                    "column": column_number,
                    "sample_id": headers[column_number - 1],
                    "taxon": taxon,
                    "original_value": raw,
                    "issue_type": reason,
                })

            # Every unexpected field is recorded, including numeric values.
            # A valid-looking number in an unlabelled field is not safe data.
            for column_number in range(len(headers) + 1, len(row) + 1):
                issues.append({
                    "row": row_number,
                    "column": column_number,
                    "sample_id": None,
                    "taxon": taxon,
                    "original_value": row[column_number - 1],
                    "issue_type": "unexpected_column",
                })

    return issues


if __name__ == "__main__":
    import sys
    from collections import Counter

    issues = find_invalid_counts(Path(sys.argv[1]))
    counts = Counter(issue["issue_type"] for issue in issues)
    print(f"Invalid cells: {len(issues)}")
    print(f"Issues by type: {dict(counts)}")
