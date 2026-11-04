import csv
from collections import defaultdict
from pathlib import Path


def inspect_headers(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        headers = next(reader)

        row_count = 0
        ragged_records = []

        for record_number, row in enumerate(reader, start=2):
            row_count += 1

            if len(row) != len(headers):
                ragged_records.append((record_number, len(row)))

    positions = defaultdict(list)

    for column_number, header in enumerate(headers, start=1):
        positions[header].append(column_number)

    print(f"Total columns: {len(headers)}")
    print(f"Data rows: {row_count}")
    print(f"Ragged records: {len(ragged_records)}")

    for record_number, field_count in ragged_records:
        print(
            f"Record {record_number}: "
            f"expected {len(headers)} fields, found {field_count}"
        )

    print(f"Blank header positions: {positions.get('', [])}")

    for header, columns in positions.items():
        if len(columns) > 1:
            print(f"Duplicate header: {header!r}, columns: {columns}")

    return {
        "path": str(path),
        "headers": headers,
        "total_columns": len(headers),
        "data_rows": row_count,
        "blank_header_positions": positions.get("", []),
        "duplicate_headers": {
            name: columns
            for name, columns in positions.items()
            if len(columns) > 1
        },
        "ragged_records": ragged_records,
    }


if __name__ == "__main__":
    import sys

    inspect_headers(Path(sys.argv[1]))
