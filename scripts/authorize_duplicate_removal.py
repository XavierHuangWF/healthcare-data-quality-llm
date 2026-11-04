
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from healthcare_data_quality.decision_memory import fingerprint_dataset
from healthcare_data_quality.tools.column_inspection import compare_columns


AUTHORIZATION_PATH = Path(
    "data/decision_memory/operation_authorizations.jsonl"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.source
    output = args.output

    keep_position = 105
    drop_position = 106

    if source.resolve() == output.resolve():
        raise ValueError("Cannot overwrite source dataset")

    if output.exists():
        raise FileExistsError("Output already exists")

    comparison = compare_columns(
        source, keep_position, drop_position
    )[0]

    if not comparison["identical"]:
        raise ValueError("Columns are not identical")

    if comparison["left_name"] != comparison["right_name"]:
        raise ValueError("Column names differ")

    source_hash = fingerprint_dataset(source)

    print("\nPROPOSED REMEDIATION")
    print("Operation: drop_duplicate_column")
    print("Keep column:", keep_position)
    print("Remove column:", drop_position)
    print("Column name:", comparison["left_name"])
    print("Output:", output.resolve())
    print("Original dataset will be preserved.")
    print("Semantic equivalence is NOT independently verified.")

    reviewer = input("\nReviewer ID: ").strip()
    evidence = input(
        "Supporting scientific documentation reference: "
    ).strip()

    if not reviewer or not evidence:
        print("DEFERRED: Reviewer and evidence required")
        return

    confirmation = input(
        "\nType AUTHORIZE DROP 106 TO COPY: "
    ).strip()

    if confirmation != "AUTHORIZE DROP 106 TO COPY":
        print("DEFERRED: Authorization not granted")
        return

    if fingerprint_dataset(source) != source_hash:
        raise RuntimeError("Source dataset changed during review")

    record = {
        "authorization_id": str(uuid4()),
        "operation_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_fingerprint": source_hash,
        "output_path": str(output.resolve()),
        "operation": "drop_duplicate_column",
        "keep_position": keep_position,
        "drop_position": drop_position,
        "reviewer": reviewer,
        "semantic_evidence_reference": evidence,
        "authorization_scope": "derived_copy_only",
        "human_confirmation": True,
    }

    AUTHORIZATION_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    with AUTHORIZATION_PATH.open(
        "a", encoding="utf-8"
    ) as f:
        f.write(json.dumps(record) + "\n")

    print("\nAUTHORIZATION RECORDED")
    print("Authorization ID:", record["authorization_id"])
    print("Operation ID:", record["operation_id"])
    print("Execution status: not_executed")
    print("No dataset changes performed.")


if __name__ == "__main__":
    main()
