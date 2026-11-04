
import argparse
import json
from pathlib import Path

from healthcare_data_quality.tools.column_inspection import (
    compare_columns,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("left", type=int)
    parser.add_argument("right", type=int)
    args = parser.parse_args()

    if args.left == args.right:
        raise ValueError("Two different positions are required")

    comparison = compare_columns(
        args.dataset,
        args.left,
        args.right,
    )[0]

    if not comparison["identical"]:
        raise ValueError(
            "Columns are not identical. Removal cannot be proposed."
        )

    if comparison["left_name"] != comparison["right_name"]:
        raise ValueError(
            "Column names differ. Manual investigation required."
        )

    preview = {
        "operation": "drop_duplicate_column",
        "keep_position": args.left,
        "candidate_drop_position": args.right,
        "column_name": comparison["left_name"],
        "identical_values_verified": True,
        "matching_nonmissing": comparison["matching_nonmissing"],
        "both_missing": comparison["both_missing"],
        "semantic_equivalence_verified": False,
        "human_approval_required": True,
        "execution_authorized": False,
        "execution_status": "not_executed",
    }

    print(json.dumps(preview, indent=2))
    print("\nREAD-ONLY REMEDIATION PREVIEW: PASS")


if __name__ == "__main__":
    main()
