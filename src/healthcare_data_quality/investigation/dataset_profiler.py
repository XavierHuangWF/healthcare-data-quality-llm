
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path


def profile_csv(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        headers = next(reader)

        columns = [
            {
                "position": i,
                "name": name,
                "missing": 0,
                "nonmissing": 0,
                "numeric": 0,
                "examples": [],
                "unique_values": set(),
            }
            for i, name in enumerate(headers, start=1)
        ]

        row_count = 0

        for record_number, row in enumerate(reader, start=2):
            if len(row) != len(headers):
                raise ValueError(
                    f"Record {record_number}: inconsistent field count"
                )

            row_count += 1

            for column, raw in zip(columns, row):
                value = raw.strip()

                if not value:
                    column["missing"] += 1
                    continue

                column["nonmissing"] += 1
                column["unique_values"].add(value)

                if (value not in column["examples"]
                        and len(column["examples"]) < 3):
                    column["examples"].append(value)

                try:
                    if Decimal(value).is_finite():
                        column["numeric"] += 1
                except InvalidOperation:
                    pass

    for column in columns:
        column["distinct_count"] = len(
            column.pop("unique_values")
        )

    return {
        "dataset": path.name,
        "rows": row_count,
        "columns": columns,
    }


if __name__ == "__main__":
    import sys

    result = profile_csv(Path(sys.argv[1]))
    print(f"Rows: {result['rows']}")
    print(f"Columns: {len(result['columns'])}")

    for column in result["columns"][:3]:
        print(column)
