import csv
from collections import defaultdict
from pathlib import Path


def find_duplicate_ids(path: Path) -> list[dict]:
    groups = defaultdict(list)

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for record_number, row in enumerate(reader, start=2):
            for field in ("SampleID", "NPDNum"):
                value = row[field]

                if value:
                    groups[(field, value)].append(record_number)

    return [
        {
            "field": field,
            "value": value,
            "records": records,
        }
        for (field, value), records in groups.items()
        if len(records) > 1
    ]

def find_duplicate_conflicts(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = {
            number: row
            for number, row in enumerate(reader, start=2)
        }
        columns = [
            col for col in reader.fieldnames
            if col not in ("SampleID", "NPDNum")
        ]

    conflicts = []

    for group in find_duplicate_ids(path):
        if group["field"] != "SampleID":
            continue

        for column in columns:
            values = {
                number: rows[number][column]
                for number in group["records"]
            }

            if len(set(values.values())) > 1:
                conflicts.append({
                    "sample_id": group["value"],
                    "field": column,
                    "values_by_record": values,
                })

    return conflicts


if __name__ == "__main__":
    import sys

    duplicates = find_duplicate_ids(Path(sys.argv[1]))

    for duplicate in duplicates:
        print(duplicate)

    print(f"Duplicate identifier groups: {len(duplicates)}")

    conflicts = find_duplicate_conflicts(Path(sys.argv[1]))

    for conflict in conflicts:
        print(conflict)

    print(f"Conflicting fields: {len(conflicts)}")
