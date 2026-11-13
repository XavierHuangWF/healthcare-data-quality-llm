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


def authorize_duplicate_removal(
        source: Path,
        output: Path,
        keep_position: int,
        drop_position: int,
        decision_id: str,
) -> dict | None:
    if (
            keep_position < 1
            or drop_position < 1
            or keep_position == drop_position
    ):
        raise ValueError("Invalid column positions")

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

    decisions_path = Path(
        "data/decision_memory/decisions.jsonl"
    )

    if not decisions_path.is_file():
        raise PermissionError("Human review history not found")

    with decisions_path.open(encoding="utf-8") as f:
        matches = [
            json.loads(line)
            for line in f
            if line.strip()
               and json.loads(line).get("decision_id")
               == decision_id
        ]

    if len(matches) != 1:
        raise PermissionError(
            "Review decision not found or duplicated"
        )

    decision = matches[0]

    if decision.get("dataset_fingerprint") != source_hash:
        raise PermissionError("Review decision dataset mismatch")

    if decision.get("human_decision") != "accepted":
        raise PermissionError("Proposal was not accepted")

    if decision.get("action_kind") != "data_change":
        raise PermissionError("Not a data-change proposal")

    if decision.get("validation_status") != "needs_human_review":
        raise PermissionError("Invalid proposal validation status")

    if decision.get("target_verified") is not True:
        raise PermissionError("Target was not verified")

    if decision.get("execution_authorized") is not False:
        raise PermissionError("Unexpected execution authorization")

    target = decision.get("target_ref") or {}

    if (
            target.get("kind") != "column_pair"
            or target.get("positions") != [keep_position, drop_position]
    ):
        raise PermissionError("Review target does not match operation")

    if "identical_columns" not in decision.get(
            "verified_fact_types", []
    ):
        raise PermissionError(
            "Required duplicate-column evidence missing"
        )

    proposal_id = decision.get("proposal_id")

    if not isinstance(proposal_id, str) or not proposal_id:
        raise PermissionError("Missing proposal ID")

    latest_decision = None

    with decisions_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            entry = json.loads(line)

            if (
                    entry.get("dataset_fingerprint") == source_hash
                    and entry.get("proposal_id") == proposal_id
            ):
                latest_decision = entry

    if (
            latest_decision is None
            or latest_decision.get("decision_id") != decision_id
    ):
        raise PermissionError(
            "Review decision has been superseded"
        )

    print("Prior accepted review verified:", decision_id)

    print("\nPROPOSED REMEDIATION")
    print("Operation: drop_duplicate_column")
    print("Keep column:", keep_position)
    print("Remove column:", drop_position)
    print("Column name:", comparison["left_name"])
    print("Output:", output.resolve())
    print("Original dataset will be preserved.")
    print("Semantic equivalence is NOT independently verified.")
    print(
        "\nSEPARATE EXECUTION AUTHORIZATION"
        "\n  Reviewer ID       - Identify the person authorizing this operation."
        "\n  Evidence reference - Identify documentation supporting the proposed change."
        "\n  Confirmation       - Explicitly authorize the specified column removal."
        "\n"
        "\n  Accepting a proposal does NOT authorize execution."
        "\n  Synthetic verification does NOT prove scientific correctness."
        "\n  Authorization creates a derived copy; the original is preserved."
        "\n  Leave a required field blank to defer authorization."
    )

    reviewer = input("\nReviewer ID: ").strip()
    evidence = input(
        "Supporting scientific documentation reference: "
    ).strip()

    if not reviewer or not evidence:
        print("DEFERRED: Reviewer and evidence required")
        return

    confirmation_phrase = (
        f"AUTHORIZE DROP {drop_position} TO COPY"
    )

    confirmation = input(
        f"\nType {confirmation_phrase}: "
    ).strip()

    if confirmation != confirmation_phrase:
        print("DEFERRED: Authorization not granted")
        return

    if fingerprint_dataset(source) != source_hash:
        raise RuntimeError("Source dataset changed during review")

    record = {
        "authorization_id": str(uuid4()),
        "operation_id": str(uuid4()),
        "review_decision_id": decision_id,
        "proposal_id": proposal_id,
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
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--keep-position", type=int, default=105)
    parser.add_argument("--drop-position", type=int, default=106)
    parser.add_argument("--decision-id", required=True)

    args = parser.parse_args()

    authorize_duplicate_removal(
        source=args.source,
        output=args.output,
        keep_position=args.keep_position,
        drop_position=args.drop_position,
        decision_id=args.decision_id,
    )


if __name__ == "__main__":
    main()
