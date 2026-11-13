
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from healthcare_data_quality.tools.column_inspection import compare_columns


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_duplicate_removal(
    source: Path,
    output: Path,
    authorization_id: str,
) -> dict:


    if source.resolve() == output.resolve():
        raise ValueError("Cannot overwrite the original dataset")

    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    # New authorization verification
    source_hash = sha256(source)

    authorization_path = Path(
        "data/decision_memory/operation_authorizations.jsonl"
    )

    if not authorization_path.is_file():
        raise PermissionError("No authorization record exists")

    matches = []

    with authorization_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("authorization_id") == authorization_id:
                matches.append(record)

    if len(matches) != 1:
        raise PermissionError("Authorization missing or duplicated")

    authorization = matches[0]

    keep_position = authorization.get("keep_position")
    drop_position = authorization.get("drop_position")

    if (
            type(keep_position) is not int
            or type(drop_position) is not int
            or keep_position < 1
            or drop_position < 1
            or keep_position == drop_position
    ):
        raise PermissionError(
            "Invalid authorized column positions"
        )

    expected = {
        "source_fingerprint": source_hash,
        "output_path": str(output.resolve()),
        "operation": "drop_duplicate_column",
        "keep_position": keep_position,
        "drop_position": drop_position,
        "authorization_scope": "derived_copy_only",
        "human_confirmation": True,
    }

    for key, value in expected.items():
        if authorization.get(key) != value:
            raise PermissionError(
                f"Authorization mismatch: {key}"
            )

    comparison = compare_columns(
        source, keep_position, drop_position
    )[0]

    if not comparison["identical"]:
        raise ValueError("Column values are not identical")

    if comparison["left_name"] != comparison["right_name"]:
        raise ValueError("Column headers are different")

    with source.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError("Empty dataset")

    width = len(rows[0])

    if width < max(keep_position, drop_position):
        raise ValueError("Column position exceeds dataset width")

    if any(len(row) != width for row in rows):
        raise ValueError("Inconsistent row widths")

    keep_index = keep_position - 1
    drop_index = drop_position - 1

    if any(
        row[keep_index] != row[drop_index]
        for row in rows
    ):
        raise ValueError("Columns differ at the row level")

    print("Duplicate-column inspection: PASS")
    print("Original columns:", width)
    print("Original records:", len(rows) - 1)
    print("Keep:", keep_position)
    print("Candidate removal:", drop_position)
    print("Original dataset will not be modified.")




    # Revalidate human approval immediately before execution.
    review_id = authorization.get("review_decision_id")
    proposal_id = authorization.get("proposal_id")

    if not all(
        isinstance(value, str) and value
        for value in (review_id, proposal_id)
    ):
        raise PermissionError("Authorization lacks review linkage")

    decisions_path = Path("data/decision_memory/decisions.jsonl")

    if not decisions_path.is_file():
        raise PermissionError("Review history missing")

    latest_review = None

    with decisions_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            entry = json.loads(line)

            if (
                entry.get("dataset_fingerprint") == source_hash
                and entry.get("proposal_id") == proposal_id
            ):
                latest_review = entry

    if (
        latest_review is None
        or latest_review.get("decision_id") != review_id
        or latest_review.get("human_decision") != "accepted"
    ):
        raise PermissionError(
            "Authorization revoked or review superseded"
        )

    if (
        latest_review.get("action_kind") != "data_change"
        or latest_review.get("validation_status")
        != "needs_human_review"
        or latest_review.get("target_verified") is not True
        or latest_review.get("execution_authorized") is not False
    ):
        raise PermissionError("Review validation no longer valid")

    target = latest_review.get("target_ref") or {}

    if (
        target.get("kind") != "column_pair"
        or target.get("positions")
        != [keep_position, drop_position]
        or "identical_columns" not in latest_review.get(
            "verified_fact_types", []
        )
    ):
        raise PermissionError("Review evidence or target mismatch")

    print("Current human approval verified:", review_id)


    operation_id = authorization.get("operation_id")
    authorization_id = authorization["authorization_id"]
    reviewer = authorization.get("reviewer")
    rationale = authorization.get("semantic_evidence_reference")

    if not isinstance(operation_id, str) or not operation_id:
        raise PermissionError("Missing operation ID")

    if not isinstance(reviewer, str) or not reviewer.strip():
        raise PermissionError("Missing reviewer")

    if not isinstance(rationale, str) or not rationale.strip():
        raise PermissionError("Missing semantic evidence")

    audit_path = Path(
        "data/decision_memory/remediation_runs.jsonl"
    )

    if audit_path.is_file():
        with audit_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                previous = json.loads(line)

                if (
                    previous.get("authorization_id") == authorization_id
                    or previous.get("operation_id") == operation_id
                ):
                    raise PermissionError(
                        "Authorization already used"
                    )

    print("Existing authorization verified:", authorization_id)



    transformed = [
        row[:drop_index] + row[drop_index + 1:]
        for row in rows
    ]

    output.parent.mkdir(parents=True, exist_ok=True)

    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        suffix=".csv",
        prefix=".remediation-",
        dir=output.parent,
        delete=False,
    ) as f:
        temp_path = Path(f.name)
        csv.writer(f).writerows(transformed)

    try:
        with temp_path.open(
            "r", encoding="utf-8", newline=""
        ) as f:
            verified = list(csv.reader(f))

        if verified != transformed:
            raise RuntimeError("Output verification failed")

        if sha256(source) != source_hash:
            raise RuntimeError("Original dataset changed")

        if output.exists():
            raise FileExistsError("Output already exists")

        temp_path.rename(output)

    finally:
        temp_path.unlink(missing_ok=True)

    audit = {
        "operation_id": operation_id,
        "authorization_id": authorization_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reviewer": reviewer,
        "semantic_attestation": rationale,
        "operation": "drop_duplicate_column",
        "source_fingerprint": source_hash,
        "output_fingerprint": sha256(output),
        "keep_position": keep_position,
        "drop_position": drop_position,
        "original_modified": False,
        "authorization_scope": "derived_copy_only",
        "execution_status": "copy_created_and_verified",
    }

    audit_path = Path(
        "data/decision_memory/remediation_runs.jsonl"
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    with audit_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(audit) + "\n")

    print("\nREMEDIATION COPY: VERIFIED")
    print("Output:", output)
    print("Records:", len(transformed) - 1)
    print("Columns:", len(transformed[0]))
    print("Original unchanged: True")
    print("Audit:", audit_path)
    return audit

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--authorization-id", required=True)

    args = parser.parse_args()

    apply_duplicate_removal(
        source=args.source,
        output=args.output,
        authorization_id=args.authorization_id,
    )

if __name__ == "__main__":
    main()
