from pathlib import Path
from ..dataset_profiler import profile_csv
import csv

def inspect_column(path: Path, position: int) -> list[dict]:
    profile = profile_csv(path)
    columns = profile["columns"]
    if not 1 <= position <= len(columns):
        raise ValueError(
            f"Invalid column position: {position}. "
            f"Valid range: 1–{len(columns)}"
        )
    return [columns[position - 1]]


def inspect_columns(path: Path, positions: list[int]) -> list[dict]:
    if not isinstance(positions, list):
        raise ValueError("positions must be a list")

    if not 1 <= len(positions) <= 4:
        raise ValueError("Specify between 1 and 4 columns")

    if any(type(pos) is not int for pos in positions):
        raise ValueError("Column positions must be integers")

    if len(set(positions)) != len(positions):
        raise ValueError("Duplicate positions are not allowed")

    columns = profile_csv(path)["columns"]

    if any(not 1 <= pos <= len(columns) for pos in positions):
        raise ValueError("Column position out of range")

    return [columns[pos - 1] for pos in positions]


def compare_columns(
    path: Path,
    left: int,
    right: int,
) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        headers = next(reader)

        if (
            type(left) is not int
            or type(right) is not int
            or left == right
            or not 1 <= left <= len(headers)
            or not 1 <= right <= len(headers)
        ):
            raise ValueError("Invalid column positions")

        matches = 0
        conflicts = 0
        both_missing = 0
        one_missing = 0
        examples = []

        for row_number, row in enumerate(reader, start=1):
            if len(row) != len(headers):
                raise ValueError(
                    f"Unexpected field count in row {row_number}"
                )

            a = row[left - 1]
            b = row[right - 1]

            if a == "" and b == "":
                both_missing += 1
            elif a == "" or b == "":
                one_missing += 1
                if len(examples) < 5:
                    examples.append({
                        "row": row_number,
                        "left": a,
                        "right": b,
                    })
            elif a == b:
                matches += 1
            else:
                conflicts += 1
                if len(examples) < 5:
                    examples.append({
                        "row": row_number,
                        "left": a,
                        "right": b,
                    })

    return [{
        "left_position": left,
        "right_position": right,
        "left_name": headers[left - 1],
        "right_name": headers[right - 1],
        "matching_nonmissing": matches,
        "both_missing": both_missing,
        "one_missing": one_missing,
        "conflicting_nonmissing": conflicts,
        "identical": conflicts == 0 and one_missing == 0,
        "difference_examples": examples,
    }]
