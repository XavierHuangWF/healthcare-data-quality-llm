
import csv
import hashlib
import json
from pathlib import Path


def inspect_provenance(path: Path) -> list[dict]:
    """Inspect dataset-bound synthetic provenance without modifying data."""

    provenance_path = path.with_suffix(".provenance.json")

    if not provenance_path.is_file():
        return [{"status": "not_available"}]

    provenance = json.loads(
        provenance_path.read_text(encoding="utf-8")
    )

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    if provenance.get("dataset_sha256") != digest.hexdigest():
        return [{
            "status": "rejected",
            "reason": "dataset_fingerprint_mismatch",
        }]

    if (
        provenance.get("schema_version") != 1
        or provenance.get("source_type") != "synthetic_fixture"
        or provenance.get("evidence_type") != "column_lineage"
        or provenance.get("evidence_scope") != "synthetic_poc_only"
    ):
        return [{
            "status": "rejected",
            "reason": "unsupported_provenance",
        }]

    assertion = provenance.get("assertion")

    if not isinstance(assertion, dict):
        return [{
            "status": "rejected",
            "reason": "invalid_assertion",
        }]

    left = assertion.get("source_position")
    right = assertion.get("duplicate_position")
    name = assertion.get("column_name")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        headers = next(csv.reader(f), [])

    if (
        type(left) is not int
        or type(right) is not int
        or left < 1
        or right < 1
        or left == right
        or max(left, right) > len(headers)
        or not isinstance(name, str)
        or headers[left - 1] != name
        or headers[right - 1] != name
        or assertion.get("intentionally_redundant") is not True
    ):
        return [{
            "status": "rejected",
            "reason": "assertion_target_mismatch",
        }]

    return [{
        "status": "dataset_binding_verified",
        "evidence_type": "documented_column_lineage",
        "source_type": "synthetic_fixture",
        "positions": [left, right],
        "column_name": name,
        "documented_redundancy": True,
        "independent_semantic_verification": False,
        "execution_authorized": False,
    }]
