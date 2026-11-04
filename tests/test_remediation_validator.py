
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
