
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from healthcare_data_quality.decision_memory import (
    get_verified_outcome_context,
)
from healthcare_data_quality.decision_memory import (
    get_verified_outcome_context,
    record_decision,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/apply_duplicate_removal.py"



with TemporaryDirectory() as tmp:
    workspace = Path(tmp)
    source = workspace / "synthetic.csv"
    output = workspace / "cleaned.csv"

    headers = [f"column_{i}" for i in range(1, 105)]
    headers += ["ABC99", "ABC99"]

    rows = [
        headers,
        [str(i) for i in range(1, 105)] + ["10", "10"],
        [str(i) for i in range(1, 105)] + ["20", "20"],
        [str(i) for i in range(1, 105)] + ["", ""],
    ]

    with source.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    original_bytes = source.read_bytes()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    # Create an accepted review decision for synthetic test data.
    synthetic_proposal = {
        "issue": "Duplicate synthetic ABC99 columns",
        "proposed_action": "Remove redundant column 106",
        "action_kind": "data_change",
        "target_ref": {
            "kind": "column_pair",
            "positions": [105, 106],
        },
        "evidence_ids": ["T1.O1"],
        "validation": {
            "status": "needs_human_review",
            "target_verified": True,
            "verified_facts": [
                {"fact_type": "identical_columns"}
            ],
        },
    }

    decision_record = record_decision(
        dataset_path=source,
        proposal=synthetic_proposal,
        decision="accepted",
        reviewer="synthetic_test_reviewer",
        note="Synthetic fixture with intentional duplicates",
        storage_path=(
                workspace / "data/decision_memory/decisions.jsonl"
        ),
    )

    decision_id = decision_record["decision_id"]

    assert decision_record["human_decision"] == "accepted"
    assert decision_record["execution_authorized"] is False

    print("SYNTHETIC HUMAN REVIEW: PASS")

    authorize_script = ROOT / "scripts/authorize_duplicate_removal.py"


    # A later rejection must invalidate the earlier acceptance.
    record_decision(
        dataset_path=source,
        proposal=synthetic_proposal,
        decision="rejected",
        reviewer="synthetic_test_reviewer",
        note="Superseding the earlier acceptance",
        storage_path=workspace / "data/decision_memory/decisions.jsonl",
    )

    superseded = subprocess.run(
        [
            sys.executable,
            str(authorize_script),
            str(source),
            str(output),
            "--decision-id",
            decision_id,
        ],
        text=True,
        capture_output=True,
        cwd=workspace,
        env=env,
    )


    print("SUPERSEDED RETURN CODE:", superseded.returncode)
    print("SUPERSEDED STDOUT:", superseded.stdout)
    print("SUPERSEDED STDERR:", superseded.stderr)

    assert superseded.returncode != 0
    assert "Review decision has been superseded" in (
        superseded.stdout + superseded.stderr
    )
    assert not output.exists()
    assert source.read_bytes() == original_bytes

    print("SUPERSEDED REVIEW REJECTED: PASS")

    # Create a fresh acceptance for the remaining integration tests.
    decision_record = record_decision(
        dataset_path=source,
        proposal=synthetic_proposal,
        decision="accepted",
        reviewer="synthetic_test_reviewer",
        note="Fresh synthetic acceptance",
        storage_path=workspace / "data/decision_memory/decisions.jsonl",
    )

    decision_id = decision_record["decision_id"]


    approval = subprocess.run(
        [
            sys.executable,
            str(authorize_script),
            str(source),
            str(output),
            "--decision-id",
            decision_id,
        ],
        input=(
            "synthetic_test_reviewer\n"
            "Synthetic fixture: intentionally identical columns\n"
            "AUTHORIZE DROP 106 TO COPY\n"
        ),
        text=True,
        capture_output=True,
        cwd=workspace,
        env=env,
    )

    assert approval.returncode == 0, (
        approval.stdout + approval.stderr
    )
    assert "AUTHORIZATION RECORDED" in approval.stdout

    authorization_path = (
        workspace
        / "data/decision_memory/operation_authorizations.jsonl"
    )

    authorization = json.loads(
        authorization_path.read_text(encoding="utf-8").strip()
    )

    authorization_id = authorization["authorization_id"]

    print("SEPARATE HUMAN AUTHORIZATION: PASS")


    invalid_execution = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            str(output),
            "--authorization-id",
            "invalid-authorization-id",
        ],
        text=True,
        capture_output=True,
        cwd=workspace,
        env=env,
    )

    assert invalid_execution.returncode != 0
    assert "Authorization missing or duplicated" in (
        invalid_execution.stdout + invalid_execution.stderr
    )
    assert not output.exists()
    assert source.read_bytes() == original_bytes

    print("INVALID AUTHORIZATION REJECTED: PASS")


    redirected_output = workspace / "unauthorized_output.csv"

    redirected_execution = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            str(redirected_output),
            "--authorization-id",
            authorization_id,
        ],
        text=True,
        capture_output=True,
        cwd=workspace,
        env=env,
    )

    assert redirected_execution.returncode != 0
    assert "Authorization mismatch: output_path" in (
        redirected_execution.stdout + redirected_execution.stderr
    )
    assert not redirected_output.exists()
    assert source.read_bytes() == original_bytes

    print("OUTPUT REDIRECTION REJECTED: PASS")


    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            str(output),
            "--authorization-id",
            authorization_id,
        ],
        text=True,
        capture_output=True,
        cwd=workspace,
        env=env,
    )


    assert result.returncode == 0, result.stderr
    assert "REMEDIATION COPY: VERIFIED" in result.stdout
    assert source.read_bytes() == original_bytes
    assert output.exists()

    with output.open(newline="", encoding="utf-8") as f:
        cleaned = list(csv.reader(f))

    assert len(cleaned) == 4
    assert all(len(row) == 105 for row in cleaned)
    assert cleaned == [row[:-1] for row in rows]

    audit_path = workspace / "data/decision_memory/remediation_runs.jsonl"
    audit = json.loads(audit_path.read_text().strip())

    assert audit["execution_status"] == "copy_created_and_verified"
    assert audit["original_modified"] is False


    authorization_path = (
        workspace
        / "data/decision_memory/operation_authorizations.jsonl"
    )

    assert authorization_path.exists()

    authorization = json.loads(
        authorization_path.read_text(encoding="utf-8").strip()
    )

    assert authorization["operation_id"] == audit["operation_id"]
    assert authorization["authorization_id"] == audit["authorization_id"]
    assert authorization["source_fingerprint"] == audit["source_fingerprint"]
    assert authorization["operation"] == "drop_duplicate_column"
    assert authorization["keep_position"] == 105
    assert authorization["drop_position"] == 106
    assert authorization["authorization_scope"] == "derived_copy_only"
    assert authorization["human_confirmation"] is True

    print("AUTHORIZATION-EXECUTION LINKAGE: PASS")


    print("SYNTHETIC REMEDIATION: PASS")
    print("ORIGINAL PRESERVATION: PASS")
    print("OUTPUT VERIFICATION: PASS")
    print("AUDIT RECORD: PASS")


    verify_script = ROOT / "scripts/verify_remediation.py"

    verification = subprocess.run(
        [
            sys.executable,
            str(verify_script),
            str(source),
            str(output),
            str(audit_path),
            "--operation-id",
            audit["operation_id"],
            "--outcome-log",
            str(workspace / "remediation_outcomes.jsonl"),
        ],
        text=True,
        capture_output=True,
        cwd=workspace,
        env=env,
    )

    assert verification.returncode == 0, (
        verification.stdout + verification.stderr
    )

    assert (
        "POST-REMEDIATION VERIFICATION: PASS"
        in verification.stdout
    )

    print("INDEPENDENT REMEDIATION VERIFICATION: PASS")


    outcome_path = workspace / "remediation_outcomes.jsonl"
    assert outcome_path.exists()

    outcomes = [
        json.loads(line)
        for line in outcome_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(outcomes) == 1

    outcome = outcomes[0]
    assert outcome["technical_verification"] == "passed"
    assert outcome["semantic_verification"] == "not_established"
    assert outcome["autonomous_execution_eligible"] is False
    assert outcome["records"] == 3
    assert outcome["columns_before"] == 106
    assert outcome["columns_after"] == 105

    print("VERIFIED OUTCOME MEMORY: PASS")


    recalled = get_verified_outcome_context(
        source,
        outcome_path,
    )

    assert len(recalled) == 1
    assert recalled[0]["operation_id"] == audit["operation_id"]
    assert recalled[0]["technical_verification"] == "passed"
    assert recalled[0]["semantic_verification"] == "not_established"
    assert recalled[0]["autonomous_execution_eligible"] is False

    print("VERIFIED OUTCOME RECALL: PASS")


    # Remove only the synthetic derived output.
    # This ensures the replay reaches the authorization check,
    # rather than failing because the destination already exists.
    output.unlink()

    replay_execution = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            str(output),
            "--authorization-id",
            authorization_id,
        ],
        text=True,
        capture_output=True,
        cwd=workspace,
        env=env,
    )

    assert replay_execution.returncode != 0
    assert "Authorization already used" in (
        replay_execution.stdout + replay_execution.stderr
    )
    assert not output.exists()
    assert source.read_bytes() == original_bytes


    # Create a new accepted review and an unused authorization.
    pending_decision = record_decision(
        dataset_path=source,
        proposal=synthetic_proposal,
        decision="accepted",
        reviewer="synthetic_test_reviewer",
        note="Approval before later revocation",
        storage_path=workspace / "data/decision_memory/decisions.jsonl",
    )

    pending_output = workspace / "revoked_output.csv"

    pending_approval = subprocess.run(
        [
            sys.executable,
            str(authorize_script),
            str(source),
            str(pending_output),
            "--decision-id",
            pending_decision["decision_id"],
        ],
        input=(
            "synthetic_test_reviewer\n"
            "Synthetic fixture evidence\n"
            "AUTHORIZE DROP 106 TO COPY\n"
        ),
        text=True,
        capture_output=True,
        cwd=workspace,
        env=env,
    )

    assert pending_approval.returncode == 0, (
        pending_approval.stdout + pending_approval.stderr
    )

    with authorization_path.open(encoding="utf-8") as f:
        authorizations = [
            json.loads(line) for line in f if line.strip()
        ]

    pending_authorization_id = authorizations[-1]["authorization_id"]

    # The reviewer changes the decision before execution.
    record_decision(
        dataset_path=source,
        proposal=synthetic_proposal,
        decision="rejected",
        reviewer="synthetic_test_reviewer",
        note="Approval revoked before execution",
        storage_path=workspace / "data/decision_memory/decisions.jsonl",
    )

    revoked_execution = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            str(pending_output),
            "--authorization-id",
            pending_authorization_id,
        ],
        text=True,
        capture_output=True,
        cwd=workspace,
        env=env,
    )

    assert revoked_execution.returncode != 0
    assert "Authorization revoked or review superseded" in (
        revoked_execution.stdout + revoked_execution.stderr
    )
    assert not pending_output.exists()
    assert source.read_bytes() == original_bytes

    print("POST-AUTHORIZATION REVOCATION REJECTED: PASS")

    print("AUTHORIZATION REPLAY REJECTED: PASS")


    # Test a different column pair: keep 2, remove 4.
    small_source = workspace / "synthetic_small.csv"
    small_output = workspace / "cleaned_small.csv"

    small_rows = [
        ["subject_id", "score", "site", "score"],
        ["S001", "10", "A", "10"],
        ["S002", "20", "B", "20"],
        ["S003", "", "A", ""],
    ]

    with small_source.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        csv.writer(f).writerows(small_rows)

    small_original = small_source.read_bytes()

    small_proposal = {
        "issue": "Duplicate score columns",
        "proposed_action": "Remove column 4",
        "action_kind": "data_change",
        "target_ref": {
            "kind": "column_pair",
            "positions": [2, 4],
        },
        "evidence_ids": ["T1.O1"],
        "validation": {
            "status": "needs_human_review",
            "target_verified": True,
            "verified_facts": [
                {"fact_type": "identical_columns"}
            ],
        },
    }

    small_decision = record_decision(
        dataset_path=small_source,
        proposal=small_proposal,
        decision="accepted",
        reviewer="synthetic_test_reviewer",
        storage_path=workspace / "data/decision_memory/decisions.jsonl",
    )

    small_approval = subprocess.run(
        [
            sys.executable,
            str(authorize_script),
            str(small_source),
            str(small_output),
            "--keep-position", "2",
            "--drop-position", "4",
            "--decision-id", small_decision["decision_id"],
        ],
        input=(
            "synthetic_test_reviewer\n"
            "Synthetic fixture with known duplicate score columns\n"
            "AUTHORIZE DROP 4 TO COPY\n"
        ),
        text=True,
        capture_output=True,
        cwd=workspace,
        env=env,
    )

    assert small_approval.returncode == 0, (
        small_approval.stdout + small_approval.stderr
    )

    with authorization_path.open(encoding="utf-8") as f:
        small_authorization = [
            json.loads(line) for line in f if line.strip()
        ][-1]

    small_execution = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(small_source),
            str(small_output),
            "--authorization-id",
            small_authorization["authorization_id"],
        ],
        text=True,
        capture_output=True,
        cwd=workspace,
        env=env,
    )

    assert small_execution.returncode == 0, (
        small_execution.stdout + small_execution.stderr
    )

    with small_output.open(
        "r", encoding="utf-8", newline=""
    ) as f:
        small_cleaned = list(csv.reader(f))

    assert small_cleaned == [
        row[:3] for row in small_rows
    ]
    assert small_source.read_bytes() == small_original

    print("ARBITRARY COLUMN PAIR 2/4: PASS")
