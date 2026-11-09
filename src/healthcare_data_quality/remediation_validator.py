from .investigation_report import build_evidence_index
from copy import deepcopy
import re

def has_executable_plan(proposal: dict) -> bool:
    plan = proposal.get("execution_plan")

    if not isinstance(plan, dict):
        return False

    required_fields = (
        "transformation",
        "target_selection",
        "expected_result",
        "verification_rule",
    )

    return (
        plan.get("requires_clarification") is False
        and all(
            isinstance(plan.get(field), str)
            and bool(plan[field].strip())
            for field in required_fields
        )
    )

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
        contract = proposal.get("verification_contract")

        contract_entry = (
            evidence_index.get(contract.get("evidence_id"))
            if isinstance(contract, dict)
            else None
        )

        observation = (
            contract_entry.get("observation")
            if isinstance(contract_entry, dict)
            else None
        )

        records = (
            observation.get("incomplete_taxonomy")
            if isinstance(observation, dict)
            else None
        )

        contract_supported = (
            isinstance(contract, dict)
            and contract.get("kind") == "flag_rows_from_evidence"
            and contract.get("evidence_id") in evidence_ids
            and isinstance(contract_entry, dict)
            and contract_entry.get("tool") == "inspect_taxonomy_structure"
            and contract.get("observation_field") == "incomplete_taxonomy"
            and contract.get("record_field") == "record"
            and isinstance(records, list)
            and bool(records)
            and all(
                isinstance(item, dict)
                and type(item.get("record")) is int
                and item["record"] >= 2
                for item in records
            )
        )

        # Validate numeric QC contracts against baseline observations.
        numeric_supported = False

        if (
            isinstance(contract, dict)
            and contract.get("kind") == "numeric_qc_from_baseline"
        ):
            observations = (
                (baseline or {})
                .get("checks", {})
                .get("find_invalid_counts", {})
                .get("observations")
            )

            conditions = contract.get("conditions")

            if (
                isinstance(observations, list)
                and bool(observations)
                and isinstance(conditions, list)
                and bool(conditions)
            ):
                available = {
                    (item.get("issue_type"), item.get("column"))
                    for item in observations
                    if isinstance(item, dict)
                    and item.get("issue_type") in {
                        "negative", "fractional", "missing"
                    }
                    and type(item.get("column")) is int
                }

                numeric_supported = all(
                    isinstance(condition, dict)
                    and condition.get("issue_type") in {
                        "negative", "fractional", "missing"
                    }
                    and isinstance(condition.get("columns"), list)
                    and bool(condition["columns"])
                    and all(
                        type(column) is int
                        and (condition["issue_type"], column) in available
                        for column in condition["columns"]
                    )
                    for condition in conditions
                )

                numeric_supported = (
                    numeric_supported
                    and isinstance(contract.get("output_column"), str)
                    and bool(contract["output_column"].strip())
                    and isinstance(contract.get("flagged_value"), str)
                    and isinstance(contract.get("other_value"), str)
                    and contract["flagged_value"] != contract["other_value"]
                )

        contract_supported = contract_supported or numeric_supported

        model_intent_conflict = False
        # The verified contract defines the binding cleaning scope.
        # Do not allow separate model-generated prose to define
        # a different operation for human approval or R codegen.
        if contract_supported:
            model_generated_intent = {
                "proposed_action": proposal.get("proposed_action"),
                "execution_plan": deepcopy(proposal.get("execution_plan")),
                "review_question": proposal.get("review_question"),
            }
            original_plan = model_generated_intent["execution_plan"]

            original_texts = [
                model_generated_intent["proposed_action"],
                (
                    original_plan.get("transformation")
                    if isinstance(original_plan, dict)
                    else None
                ),
            ]

            destructive_pattern = (
                r"\b(delete|drop|remove|truncate|erase|overwrite|discard)\b"
            )

            negated_destructive_pattern = (
                r"\b(?:do not|does not|did not|must not|should not|"
                r"will not|never|without|don't|doesn't|won't)\s+"
                r"(?:\w+\s+){0,2}?"
                r"(?:delete|drop|remove|truncate|erase|overwrite|discard)\b"
            )

            def contains_destructive_intent(value):
                if not isinstance(value, str):
                    return False

                remaining = re.sub(
                    negated_destructive_pattern,
                    " ",
                    value,
                    flags=re.IGNORECASE,
                )

                return bool(
                    re.search(destructive_pattern, remaining, re.IGNORECASE)
                )

            model_intent_conflict = any(
                contains_destructive_intent(value)
                for value in original_texts
            )
            if contract["kind"] == "numeric_qc_from_baseline":
                selection = " OR ".join(
                    f"{condition['issue_type']} in CSV columns "
                    + ", ".join(
                        str(column)
                        for column in sorted(set(condition["columns"]))
                    )
                    for condition in contract["conditions"]
                )
            else:
                selection = (
                    f"Physical CSV rows identified by "
                    f"{contract['evidence_id']}, "
                    f"observation field {contract['observation_field']}, "
                    f"record field {contract['record_field']}"
                )

            action = (
                f"Append one QC column {contract['output_column']!r}. "
                f"For records matching ({selection}), assign "
                f"{contract['flagged_value']!r}; otherwise assign "
                f"{contract['other_value']!r}. "
                "Preserve every existing row, column, and cell value."
            )

            proposal = {
                **proposal,
                "model_generated_intent": model_generated_intent,
                "authorized_effects": {
                    "kind": "append_qc_column",
                    "output_column": contract["output_column"],
                    "selection_contract": deepcopy(contract),
                    "preserve_original_rows": True,
                    "preserve_original_columns": True,
                    "preserve_original_cells": True,
                },
                "proposed_action": action,
                "execution_plan": {
                    "transformation": action,
                    "target_selection": selection,
                    "expected_result": (
                        "Exactly one QC column appended; all original "
                        "records, columns, and values preserved."
                    ),
                    "verification_rule": (
                        "Independently derive the matching records from "
                        "baseline evidence and compare every output flag. "
                        "Reject any unexpected modification."
                    ),
                    "requires_clarification": False,
                },
                "review_question": (
                    "Do you approve exactly this QC transformation?"
                ),
            }
        declared_effects_match = None

        if contract_supported:
            declared = proposal.get("declared_effects")
            authorized = proposal["authorized_effects"]

            comparable_fields = (
                "kind",
                "output_column",
                "preserve_original_rows",
                "preserve_original_columns",
                "preserve_original_cells",
            )

            declared_effects_match = (
                isinstance(declared, dict)
                and set(declared) == set(comparable_fields)
                and all(
                    type(declared[field]) is type(authorized[field])
                    and declared[field] == authorized[field]
                    for field in comparable_fields
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
                "operation_supported": operation_supported,
                "target_verified": target_verified,
                "verified_facts": verified_facts,
                "verification_contract_supported": contract_supported,
                "status": (
                    "needs_human_review"
                    if (
                            references_valid
                            and not model_intent_conflict
                            and (
                                    not contract_supported
                                    or declared_effects_match is True
                            )
                            and (contract_supported or operation_supported)
                            and proposal.get("action_kind") == "data_change"
                            and has_executable_plan(proposal)
                            and isinstance(proposal.get("proposed_action"), str)
                            and bool(proposal["proposed_action"].strip())
                            and isinstance(operation, dict)
                            and (
                                    not contract_supported
                                    or (
                                            operation.get("kind") == "none"
                                            and operation.get("keep_position") is None
                                            and operation.get("drop_position") is None
                                    )
                            )
                            and (
                                    operation_supported
                                    or operation.get("kind") == "none"
                                    or (
                                            operation.get("kind") == "drop_duplicate_column"
                                            and target_verified
                                            and target_ref["positions"] == [
                                                operation.get("keep_position"),
                                                operation.get("drop_position"),
                                            ]
                                    )
                            )
                    )
                    else "rejected"
                ),
                "model_intent_conflict": model_intent_conflict,
                "declared_effects_match": declared_effects_match,
            },
        })

    return validated
