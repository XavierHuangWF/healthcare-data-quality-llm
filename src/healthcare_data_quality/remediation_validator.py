
from .investigation_report import build_evidence_index


def validate_remediation_proposals(
    history: list[dict],
    proposals: list[dict],
) -> list[dict]:
    evidence_index = build_evidence_index(history)
    validated = []

    for proposal in proposals:
        evidence_ids = proposal.get("evidence_ids", [])

        unknown_ids = [
            evidence_id
            for evidence_id in evidence_ids
            if evidence_id not in evidence_index
        ]

        references_valid = (
            bool(evidence_ids) and not unknown_ids
        )

        action_valid = proposal.get("action_kind") in {
            "investigation",
            "data_change",
        }


        verified_facts = []

        if references_valid:
            for evidence_id in evidence_ids:
                entry = evidence_index[evidence_id]

                if entry["tool"] != "compare_columns":
                    continue

                observation = entry["observation"]

                if not isinstance(observation, dict):
                    continue

                left = observation.get("left_position")
                right = observation.get("right_position")

                if (
                    observation.get("identical") is True
                    and type(left) is int
                    and type(right) is int
                    and left != right
                    and observation.get("one_missing") == 0
                    and observation.get("conflicting_nonmissing") == 0
                ):
                    verified_facts.append({
                        "fact_type": "identical_columns",
                        "positions": [left, right],
                        "evidence_id": evidence_id,
                    })


        target_ref = proposal.get("target_ref")

        target_verified = (
            isinstance(target_ref, dict)
            and target_ref.get("kind") == "column_pair"
            and isinstance(target_ref.get("positions"), list)
            and len(target_ref["positions"]) == 2
            and all(
                type(position) is int
                for position in target_ref["positions"]
            )
            and any(
                fact["fact_type"] == "identical_columns"
                and sorted(fact["positions"])
                == sorted(target_ref["positions"])
                for fact in verified_facts
            )
        )


        validated.append({
            **proposal,
            "execution_authorized": False,
            "validation": {
                "evidence_references_valid": references_valid,
                "unknown_evidence_ids": unknown_ids,
                "action_kind_valid": action_valid,
                "claim_support_verified": False,
                "target_verified": target_verified,
                "verified_facts": verified_facts,
                "status": (
                    "needs_human_review"
                    if references_valid and action_valid
                    else "rejected"
                ),
            },
        })


    return validated
