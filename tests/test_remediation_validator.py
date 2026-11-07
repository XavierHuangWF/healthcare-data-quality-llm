
from pathlib import Path

from healthcare_data_quality.tools.column_inspection import compare_columns
from healthcare_data_quality.remediation_validator import (
    validate_remediation_proposals,
)


path = Path("data/dirtyL7PVDforXav.csv")

history = [{
    "step": 1,
    "tool": "compare_columns",
    "observations": compare_columns(path, 105, 106),
}]

proposal = {
    "issue": "Potential duplicate columns",
    "proposed_action": "Review before removing either column",
    "target": "Columns 105 and 106",
    "rationale": "Both columns contain identical values",
    "evidence_ids": ["T1.O1"],
    "uncertainty": "Semantic redundancy is unverified",
    "action_kind": "investigation",
    "execution_authorized": True,
}

# Valid evidence must still require review.
valid = validate_remediation_proposals(history, [proposal])[0]


assert valid["validation"]["verified_facts"] == [{
    "fact_type": "identical_columns",
    "positions": [105, 106],
    "evidence_id": "T1.O1",
}]

print("STRUCTURED FACT VERIFICATION: PASS")

assert valid["validation"]["evidence_references_valid"]
assert valid["validation"]["status"] == "needs_human_review"
assert valid["execution_authorized"] is False

print("VALID EVIDENCE: PASS")
print("EXECUTION SAFETY: PASS")

# Fabricated evidence IDs must be rejected.
invalid = {
    **proposal,
    "evidence_ids": ["T99.O1"],
}

rejected = validate_remediation_proposals(history, [invalid])[0]

assert rejected["validation"]["status"] == "rejected"
assert rejected["validation"]["unknown_evidence_ids"] == ["T99.O1"]
assert rejected["execution_authorized"] is False

print("FABRICATED EVIDENCE REJECTION: PASS")
print("REMEDIATION VALIDATOR: ALL TESTS PASSED")


# Structured target matching against verified evidence.
correct_target = {
    **proposal,
    "target_ref": {
        "kind": "column_pair",
        "positions": [105, 106],
    },
}

validated = validate_remediation_proposals(
    history, [correct_target]
)[0]

assert validated["validation"]["target_verified"] is True
assert validated["execution_authorized"] is False

print("CORRECT TARGET: PASS")


# A different column pair must not be verified.
wrong_target = {
    **proposal,
    "target_ref": {
        "kind": "column_pair",
        "positions": [104, 105],
    },
}

validated = validate_remediation_proposals(
    history, [wrong_target]
)[0]

assert validated["validation"]["target_verified"] is False
assert validated["execution_authorized"] is False

print("INCORRECT TARGET REJECTION: PASS")



from pathlib import Path
from healthcare_data_quality.remediation_validator import (
    validate_remediation_proposals,
)

poc_path = Path("/tmp/poc_with_provenance.json")

if poc_path.is_file():
    import json

    poc = json.loads(poc_path.read_text(encoding="utf-8"))

    validated = validate_remediation_proposals(
        history=poc["history"],
        proposals=poc["remediation"]["proposals"],
    )

    assert len(validated) == 1

    proposal = validated[0]
    validation = proposal["validation"]

    facts = {
        fact["fact_type"]: fact
        for fact in validation["verified_facts"]
    }

    assert "identical_columns" in facts
    assert "documented_column_lineage" in facts

    assert facts["identical_columns"]["positions"] == [2, 4]
    assert facts["documented_column_lineage"]["positions"] == [2, 4]

    assert facts["documented_column_lineage"]["source_type"] == (
        "synthetic_fixture"
    )

    assert validation["target_verified"] is True
    assert validation["status"] == "needs_human_review"
    assert validation["claim_support_verified"] is False
    assert proposal["execution_authorized"] is False

    print("GEMMA STRUCTURAL EVIDENCE: PASS")
    print("GEMMA DOCUMENTED LINEAGE: PASS")
    print("HUMAN APPROVAL SAFETY: PASS")


# Reject removal of the source column instead of its documented duplicate.
reverse_history = [
    {
        "step": 1,
        "tool": "compare_columns",
        "observations": [{
            "left_position": 2,
            "right_position": 4,
            "identical": True,
            "one_missing": 0,
            "conflicting_nonmissing": 0,
        }],
    },
    {
        "step": 2,
        "tool": "inspect_provenance",
        "observations": [{
            "status": "dataset_binding_verified",
            "evidence_type": "documented_column_lineage",
            "source_type": "synthetic_fixture",
            "positions": [2, 4],
            "documented_redundancy": True,
            "independent_semantic_verification": False,
            "execution_authorized": False,
        }],
    },
]

reverse_proposal = {
    "action_kind": "data_change",
    "target_ref": {
        "kind": "column_pair",
        "positions": [4, 2],
    },
    "operation": {
        "kind": "drop_duplicate_column",
        "keep_position": 4,
        "drop_position": 2,
    },
    "evidence_ids": ["T1.O1", "T2.O1"],
}

reverse_result = validate_remediation_proposals(
    reverse_history,
    [reverse_proposal],
)[0]

assert reverse_result["validation"]["operation_supported"] is False
assert reverse_result["validation"]["status"] == "rejected"
assert reverse_result["execution_authorized"] is False

print("REVERSED COLUMN REMOVAL REJECTED: PASS")



# An investigation must never contain an executable operation.
from copy import deepcopy

control_proposal = deepcopy(reverse_proposal)
control_proposal["target_ref"]["positions"] = [2, 4]
control_proposal["operation"]["keep_position"] = 2
control_proposal["operation"]["drop_position"] = 4

control = validate_remediation_proposals(
    reverse_history, [control_proposal]
)[0]

assert control["validation"]["operation_supported"] is True

inconsistent = deepcopy(control_proposal)
inconsistent["action_kind"] = "investigation"

result = validate_remediation_proposals(
    reverse_history, [inconsistent]
)[0]

assert result["validation"]["status"] == "rejected", (
    result["validation"]
)
assert result["execution_authorized"] is False

print("ACTION/OPERATION CONSISTENCY: PASS")
