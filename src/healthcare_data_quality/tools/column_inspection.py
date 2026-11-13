import csv
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

from ..dataset_profiler import profile_csv


MAX_VALUE_DETAILS = 24
MAX_PATTERN_DETAILS = 20
MAX_EXAMPLE_ROWS = 3


def _value_shape(value: str) -> str:
    """Describe character classes without assuming a naming convention."""
    classes = (
        "A" if char.isupper() else
        "a" if char.islower() else
        "9" if char.isdigit() else
        char
        for char in value
    )
    parts = []
    for char_class, group in groupby(classes):
        count = sum(1 for _ in group)
        parts.append(
            f"{char_class}{{{count}}}" if count > 1 else char_class
        )
    return "".join(parts)


def _bounded_items(
    counter: Counter,
    limit: int,
) -> list[tuple[str, int]]:
    """Retain common and rare entries, rather than dropping high-cardinality evidence."""
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) <= limit:
        return ranked

    common_count = limit // 2
    common = ranked[:common_count]
    rare = sorted(
        ranked[common_count:],
        key=lambda item: (item[1], item[0]),
    )[:limit - common_count]
    return common + rare


def inspect_column(path: Path, position: int) -> list[dict]:
    return inspect_columns(path, [position])


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

    value_counts = {pos: Counter() for pos in positions}
    value_rows = {pos: defaultdict(list) for pos in positions}
    pattern_counts = {pos: Counter() for pos in positions}
    pattern_examples = {pos: defaultdict(list) for pos in positions}

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        next(reader)  # Already validated by profile_csv().
        for csv_row, row in enumerate(reader, start=2):
            if len(row) != len(columns):
                raise ValueError(
                    f"Record {csv_row}: inconsistent field count"
                )
            for pos in positions:
                raw = row[pos - 1]
                if not raw.strip():
                    continue

                value_counts[pos][raw] += 1
                if len(value_rows[pos][raw]) < MAX_EXAMPLE_ROWS:
                    value_rows[pos][raw].append(csv_row)

                shape = _value_shape(raw)
                pattern_counts[pos][shape] += 1
                if len(pattern_examples[pos][shape]) < MAX_EXAMPLE_ROWS:
                    pattern_examples[pos][shape].append({
                        "value": raw,
                        "csv_row": csv_row,
                    })

    results = []
    for pos in positions:
        observation = dict(columns[pos - 1])
        values = value_counts[pos]
        patterns = pattern_counts[pos]

        observation["value_frequencies"] = [
            {
                "value": value,
                "count": count,
                "csv_rows": value_rows[pos][value],
            }
            for value, count in _bounded_items(
                values, MAX_VALUE_DETAILS
            )
        ]
        observation["value_frequencies_truncated"] = (
            len(values) > MAX_VALUE_DETAILS
        )
        observation["format_patterns"] = [
            {
                "shape": shape,
                "count": count,
                "examples": pattern_examples[pos][shape],
            }
            for shape, count in _bounded_items(
                patterns, MAX_PATTERN_DETAILS
            )
        ]
        observation["format_patterns_truncated"] = (
            len(patterns) > MAX_PATTERN_DETAILS
        )
        results.append(observation)

    return results


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
