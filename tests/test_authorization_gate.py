from unittest.mock import patch

from run_workflow import eligible_for_authorization

proposal = {
    "action_kind": "data_change",
    "target_ref": {
        "kind": "column_pair",
        "positions": [105, 106],
    },
    "operation": {
        "kind": "drop_duplicate_column",
        "keep_position": 105,
        "drop_position": 106,
    },

    "validation": {
        "target_verified": True,
        "operation_supported": True,
        "status": "needs_human_review",
    },
}

cases = [
    ("Ready and accepted", {"p1"}, {"p1": "synthetic_verified"}, True),
    ("Needs clarification", {"p1"}, {"p1": "needs_clarification"}, False),
    ("Generation failed", {"p1"}, {}, False),
    ("Not accepted", set(), {"p1": "draft_ready"}, False),
    ("Invalid draft status", {"p1"}, {"p1": "invalid"}, False),
    ("Draft not verified", {"p1"}, {"p1": "draft_ready"}, False),
]

with patch("run_workflow.fingerprint_proposal", return_value="p1"):
    for name, accepted, statuses, expected in cases:
        actual = eligible_for_authorization(
            proposal, accepted, statuses
        )
        assert actual is expected, name
        print(f"{name}: PASS")

    reversed_proposal = {
        **proposal,
        "operation": {
            "kind": "drop_duplicate_column",
            "keep_position": 106,
            "drop_position": 105,
        },
    }

    assert eligible_for_authorization(
        reversed_proposal,
        {"p1"},
        {"p1": "draft_ready"},
    ) is False

    print("REVERSED OPERATION REJECTED: PASS")

print("AUTHORIZATION ELIGIBILITY: ALL TESTS PASSED")
