from collections import Counter
from pathlib import Path

from ..count_validation import find_invalid_counts
from ..dataset_profiler import profile_csv


def summarize_invalid_counts(path: Path) -> list[dict]:
    issues = find_invalid_counts(path)
    counts = Counter(issue["issue_type"] for issue in issues)

    return [
        {
            "issue_type": issue_type,
            "count": count,
            "examples": [
                issue for issue in issues
                if issue["issue_type"] == issue_type
            ][:2],
        }
        for issue_type, count in sorted(counts.items())
    ]


def inspect_matrix_structure(path: Path) -> dict:
    profile = profile_csv(path)
    columns = profile["columns"]

    candidate_columns = columns[1:]

    numeric = sum(
        col["numeric"] for col in candidate_columns
    )
    nonmissing = sum(
        col["nonmissing"] for col in candidate_columns
    )

    return {
        "rows": profile["rows"],
        "columns": len(columns),
        "first_column": columns[0] if columns else None,
        "columns_after_first": len(candidate_columns),
        "numeric_fraction_of_nonmissing_after_first_column": (
            numeric / nonmissing if nonmissing else None
        ),
        "semantics_verified": False,
    }


def observe_matrix_structure(path: Path) -> list[dict]:
    return [inspect_matrix_structure(path)]
