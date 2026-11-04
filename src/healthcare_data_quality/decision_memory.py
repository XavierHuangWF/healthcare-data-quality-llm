
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


DECISIONS_PATH = Path("data/decision_memory/decisions.jsonl")

VALID_DECISIONS = {"accepted", "rejected", "deferred"}


def fingerprint_dataset(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def fingerprint_proposal(proposal: dict) -> str:
    content = {
        "issue": proposal.get("issue"),
        "proposed_action": proposal.get("proposed_action"),
        "target_ref": proposal.get("target_ref"),
        "evidence_ids": proposal.get("evidence_ids"),
    }

    serialized = json.dumps(
        content, sort_keys=True, ensure_ascii=False
    )

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def record_decision(
    dataset_path: Path,
    proposal: dict,
    decision: str,
    reviewer: str,
    note: str = "",
    storage_path: Path = DECISIONS_PATH,
) -> dict:

    if decision not in VALID_DECISIONS:
        raise ValueError(f"Invalid decision: {decision}")

    if not reviewer.strip():
        raise ValueError("A reviewer is required")

    validation = proposal.get("validation")

    if not isinstance(validation, dict):
        raise ValueError("Proposal must be validated first")

    if (
        decision == "accepted"
        and validation.get("status") != "needs_human_review"
    ):
        raise ValueError("Rejected validation cannot be accepted")

    target = proposal.get("target_ref") or {}



    fact_types = sorted({
        fact["fact_type"]
        for fact in validation.get("verified_facts", [])
        if isinstance(fact, dict)
        and isinstance(fact.get("fact_type"), str)
    })


    action_kind = proposal.get("action_kind")
    if action_kind not in {"investigation", "data_change"}:
        action_kind = "unknown"


    record = {
        "decision_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": fingerprint_dataset(dataset_path),
        "proposal_id": fingerprint_proposal(proposal),
        "target_ref": {
            "kind": target.get("kind"),
            "positions": target.get("positions", []),
        },
        "evidence_ids": proposal.get("evidence_ids", []),
        "validation_status": validation.get("status"),
        "target_verified": validation.get("target_verified", False),
        "reviewer": reviewer,
        "human_decision": decision,
        "human_note": note,
        "execution_authorized": False,
        "execution_status": "not_executed",
        "action_kind": action_kind,
        "verified_fact_types": fact_types,
    }

    storage_path.parent.mkdir(parents=True, exist_ok=True)

    with storage_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def find_latest_decision(
    dataset_path: Path,
    proposal: dict,
    storage_path: Path = DECISIONS_PATH,
) -> dict | None:
    if not storage_path.is_file():
        return None

    dataset_id = fingerprint_dataset(dataset_path)
    proposal_id = fingerprint_proposal(proposal)

    latest = None

    with storage_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            if (
                record.get("dataset_fingerprint") == dataset_id
                and record.get("proposal_id") == proposal_id
            ):
                latest = record

    return latest


def get_decision_context(
    dataset_path: Path,
    storage_path: Path = DECISIONS_PATH,
    limit: int = 20,
) -> list[dict]:
    """Retrieve prior review decisions for the exact dataset."""

    if limit < 1:
        raise ValueError("limit must be positive")

    if not storage_path.is_file():
        return []

    dataset_id = fingerprint_dataset(dataset_path)
    latest_by_proposal = {}

    with storage_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("dataset_fingerprint") != dataset_id:
                continue

            if record.get("human_decision") not in VALID_DECISIONS:
                continue

            if record.get("execution_authorized") is not False:
                continue

            proposal_id = record.get("proposal_id")
            if not isinstance(proposal_id, str):
                continue

            latest_by_proposal[proposal_id] = {
                "proposal_id": proposal_id,
                "target_ref": record.get("target_ref"),
                "human_decision": record["human_decision"],
                "validation_status": record.get(
                    "validation_status"
                ),
                "target_verified": (
                    record.get("target_verified") is True
                ),
                "decision_scope": "recommendation_review_only",
                "execution_status": record.get(
                    "execution_status"
                ),
                "action_kind": record.get("action_kind", "unknown"),
                "verified_fact_types": record.get("verified_fact_types", []),
            }

    return list(latest_by_proposal.values())[-limit:]


def get_verified_outcome_context(
    dataset_path: Path,
    outcome_log: Path,
    limit: int = 20,
) -> list[dict]:
    """Retrieve technical outcomes for an exact dataset."""

    if limit < 1:
        raise ValueError("limit must be positive")

    if not outcome_log.is_file():
        return []

    digest = hashlib.sha256()

    with dataset_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    source_hash = digest.hexdigest()
    outcomes = {}

    with outcome_log.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("source_sha256") != source_hash:
                continue

            if record.get("technical_verification") != "passed":
                continue

            if record.get("retained_values_preserved") is not True:
                continue

            if record.get("autonomous_execution_eligible") is not False:
                continue

            operation_id = record.get("operation_id")

            if not isinstance(operation_id, str):
                continue

            outcomes[operation_id] = {
                "operation_id": operation_id,
                "technical_verification": "passed",
                "semantic_verification": record.get(
                    "semantic_verification"
                ),
                "records": record.get("records"),
                "columns_before": record.get("columns_before"),
                "columns_after": record.get("columns_after"),
                "autonomous_execution_eligible": False,
            }

    return list(outcomes.values())[-limit:]
