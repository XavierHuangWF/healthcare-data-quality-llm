
from .investigation_report import build_evidence_index


def validate_remediation_proposals(
    history: list[dict],
    proposals: list[dict],
    baseline: dict | None = None,
) -> list[dict]:
    evidence_index = build_evidence_index(
        history,
        baseline=baseline,
    )
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


                if entry["tool"] == "inspect_provenance":
                    observation = entry["observation"]

                    if not isinstance(observation, dict):
                        continue

                    positions = observation.get("positions")

                    if (
                        observation.get("status")
                        == "dataset_binding_verified"
                        and observation.get("evidence_type")
                        == "documented_column_lineage"
                        and observation.get("source_type")
                        == "synthetic_fixture"
                        and observation.get("documented_redundancy") is True
                        and observation.get(
                            "independent_semantic_verification"
                        ) is False
                        and observation.get("execution_authorized") is False
                        and isinstance(positions, list)
                        and len(positions) == 2
                        and all(
                            type(p) is int and p >= 1
                            for p in positions
                        )
                        and positions[0] != positions[1]
                    ):
                        verified_facts.append({
                            "fact_type": "documented_column_lineage",
                            "positions": positions,
                            "evidence_id": evidence_id,
                            "source_type": "synthetic_fixture",
                        })

                    continue


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


        operation = proposal.get("operation")
        operation_supported = False

        if (
            proposal.get("action_kind") == "data_change"
            and isinstance(operation, dict)
            and operation.get("kind") == "drop_duplicate_column"
        ):
            keep = operation.get("keep_position")
            drop = operation.get("drop_position")

            if (
                type(keep) is int
                and type(drop) is int
                and keep >= 1
                and drop >= 1
                and keep != drop
                and target_verified
                and target_ref["positions"] == [keep, drop]
            ):
                comparison_verified = any(
                    fact["fact_type"] == "identical_columns"
                    and sorted(fact["positions"])
                    == sorted([keep, drop])
                    for fact in verified_facts
                )

                lineage_verified = any(
                    fact["fact_type"] == "documented_column_lineage"
                    and fact["positions"] == [keep, drop]
                    and fact["source_type"] == "synthetic_fixture"
                    for fact in verified_facts
                )

                operation_supported = (
                    comparison_verified and lineage_verified
                )


        validated.append({
            **proposal,
            "execution_authorized": False,
            "validation": {
                "evidence_references_valid": references_valid,
                "unknown_evidence_ids": unknown_ids,
                "action_kind_valid": action_valid,
                "claim_support_verified": False,
                "operation_supported": operation_supported,
                "target_verified": target_verified,
                "verified_facts": verified_facts,

                "status": (
                    "needs_human_review"
                    if (
                        references_valid
                        and action_valid

                        and (
                                (
                                        proposal.get("action_kind") == "investigation"

                                        and (proposal.get("operation") or {}).get("kind", "none") == "none"

                                )
                                or (
                                        proposal.get("action_kind") == "data_change"
                                        and operation_supported
                                )
                        )

                    )
                    else "rejected"
                ),

            },
        })


    return validated