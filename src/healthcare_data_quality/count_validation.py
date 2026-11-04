import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path


def find_invalid_counts(path: Path) -> list[dict]:
    issues = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        headers = next(reader)

        for row_number, row in enumerate(reader, start=2):
            taxon = row[0]

            for column_number, raw in enumerate(row[1:], start=2):
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

    return issues

if __name__ == "__main__":
    import sys
    from collections import Counter

    issues = find_invalid_counts(Path(sys.argv[1]))
    counts = Counter(issue["issue_type"] for issue in issues)

    print(f"Invalid cells: {len(issues)}")
    print(f"Issues by type: {dict(counts)}")