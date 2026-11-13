
import argparse
import csv
import hashlib
import json
from itertools import zip_longest
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def verify_remediation(
    source: Path,
    output: Path,
    audit_path: Path,
    operation_id: str,
) -> dict:
    if not source.is_file() or not output.is_file():
        raise FileNotFoundError("Source or output dataset missing")

    if source.resolve() == output.resolve():
        raise ValueError("Source and output must be different")

    if not audit_path.is_file():
        raise FileNotFoundError("Audit record missing")

    with audit_path.open(encoding="utf-8") as f:
        records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    source_hash = sha256(source)
    output_hash = sha256(output)

    matches = [
        record for record in records
        if record.get("operation_id") == operation_id
           and record.get("source_fingerprint") == source_hash
           and record.get("output_fingerprint") == output_hash
           and record.get("operation") == "drop_duplicate_column"
           and record.get("execution_status")
           == "copy_created_and_verified"
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one matching audit record"
        )

    audit = matches[0]

    if audit.get("authorization_scope") != "derived_copy_only":
        raise ValueError("Invalid authorization scope")

    keep = audit["keep_position"] - 1
    drop = audit["drop_position"] - 1

    if keep == drop or min(keep, drop) < 0:
        raise ValueError("Invalid column positions")

    row_count = 0
    original_width = None

    with (
        source.open("r", encoding="utf-8-sig", newline="") as src,
        output.open("r", encoding="utf-8-sig", newline="") as dst,
    ):
        source_reader = csv.reader(src)
        output_reader = csv.reader(dst)

        sentinel = object()

        for source_row, output_row in zip_longest(
            source_reader,
            output_reader,
            fillvalue=sentinel,
        ):
            if source_row is sentinel or output_row is sentinel:
                raise ValueError("Row count mismatch")

            if original_width is None:
                original_width = len(source_row)

                if max(keep, drop) >= original_width:
                    raise ValueError("Column position out of range")

            if len(source_row) != original_width:
                raise ValueError("Inconsistent source row width")

            if source_row[keep] != source_row[drop]:
                raise ValueError(
                    "Source columns are not identical"
                )

            expected = (
                source_row[:drop] + source_row[drop + 1:]
            )

            if output_row != expected:
                raise ValueError(
                    f"Unexpected change at CSV row {row_count + 1}"
                )

            row_count += 1

    if row_count < 1:
        raise ValueError("Empty dataset")

    return {
        "operation_id": audit["operation_id"],
        "source_hash_verified": True,
        "output_hash_verified": True,
        "all_retained_values_preserved": True,
        "records": row_count - 1,
        "original_columns": original_width,
        "output_columns": original_width - 1,
        "technical_verification": "passed",
        "semantic_verification": "not_established",
    }


def record_verified_outcome(
    verification: dict,
    source: Path,
    output: Path,
    log_path: Path,
) -> bool:
    if (
        verification["technical_verification"] != "passed"
        or verification["all_retained_values_preserved"] is not True
    ):
        raise ValueError("Technical verification has not passed")

    event = {
        "outcome_id": str(uuid4()),
        "operation_id": verification["operation_id"],
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256(source),
        "output_sha256": sha256(output),
        "records": verification["records"],
        "columns_before": verification["original_columns"],
        "columns_after": verification["output_columns"],
        "retained_values_preserved": True,
        "technical_verification": "passed",
        "semantic_verification": "not_established",
        "autonomous_execution_eligible": False,
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)

    if log_path.exists():
        with log_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                existing = json.loads(line)

                if existing.get("operation_id") == event["operation_id"]:
                    if (
                        existing.get("source_sha256") == event["source_sha256"]
                        and existing.get("output_sha256") == event["output_sha256"]
                    ):
                        return False

                    raise ValueError(
                        "Conflicting outcome for the same operation"
                    )

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--outcome-log", type=Path, default=None)
    parser.add_argument("--operation-id", required=True)

    args = parser.parse_args()

    result = verify_remediation(
        args.source,
        args.output,
        args.audit,
        operation_id=args.operation_id,
    )


    if args.outcome_log is not None:
        recorded = record_verified_outcome(
            result,
            args.source,
            args.output,
            args.outcome_log,
        )
        print("Outcome recorded:", recorded)


    print(json.dumps(result, indent=2))
    print("POST-REMEDIATION VERIFICATION: PASS")


if __name__ == "__main__":
    main()
