import csv
from collections import defaultdict
from pathlib import Path


def inspect_headers(path: Path) -> dict:
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
    import json
    import sys

    result = inspect_headers(Path(sys.argv[1]))
    print(json.dumps(result, indent=2))

