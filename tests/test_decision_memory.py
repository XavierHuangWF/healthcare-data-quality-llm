import json
from pathlib import Path
from tempfile import TemporaryDirectory

from healthcare_data_quality.decision_memory import (
    record_decision,
    find_latest_decision,
)


proposal = {
    "issue": "Identical columns",
    "proposed_action": "Review duplicate columns",
    "action_kind": "investigation",
    "target_ref": {
        "kind": "column_pair",
        "positions": [105, 106],
    },
    "evidence_ids": ["T1.O1"],
    "validation": {
        "status": "needs_human_review",
        "target_verified": True,
        "verified_facts": [
            {
                "fact_type": "identical_columns",
                "positions": [105, 106],
                "evidence_id": "T1.O1",
            }
        ],
    },
}


with TemporaryDirectory() as tmp:
    root = Path(tmp)
    dataset = root / "sample.csv"
    storage = root / "decisions.jsonl"

    dataset.write_text("a,b\n1,1\n2,2\n")

    first = record_decision(
        dataset, proposal, "accepted",
        reviewer="human",
        storage_path=storage,
    )

    second = record_decision(
        dataset, proposal, "deferred",
        reviewer="human",
        note="Additional verification required",
        storage_path=storage,
    )

    assert first["verified_fact_types"] == ["identical_columns"], (
        f"Actual recorded facts: {first['verified_fact_types']}"
    )
    assert second["verified_fact_types"] == ["identical_columns"]

    print("SEMANTIC DECISION MEMORY: PASS")

    records = [
        json.loads(line)
        for line in storage.read_text().splitlines()
    ]

    assert len(records) == 2
    assert first["proposal_id"] == second["proposal_id"]
    assert first["decision_id"] != second["decision_id"]
    assert first["execution_authorized"] is False
    assert second["execution_authorized"] is False
    assert all(
        r["execution_status"] == "not_executed"
        for r in records
    )

    print("PERSISTENT DECISION MEMORY: PASS")
    print("APPEND-ONLY HISTORY: PASS")
    print("EXECUTION SAFETY: PASS")

    latest = find_latest_decision(dataset, proposal, storage)

    assert latest is not None
    assert latest["decision_id"] == second["decision_id"]
    assert latest["human_decision"] == "deferred"
    assert latest["execution_authorized"] is False

    print("DECISION RECALL: PASS")
