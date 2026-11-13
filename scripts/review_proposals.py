import argparse
import json
from pathlib import Path

from healthcare_data_quality.decision_memory import (
    record_decision,
    find_latest_decision,
    record_clarification,
)

from healthcare_data_quality.remediation_validator import (
    validate_remediation_proposals,
)

from healthcare_data_quality.remediation_proposal import (
    generate_remediation_proposals,
)

AMBIGUOUS_CLARIFICATIONS = {
    "a", "c", "d", "q", "r", "s", "y", "n"
}

def revise_proposal_from_answer(
        investigation: dict,
        proposal: dict,
        answer: str,
        dialogue: list[dict],
) -> dict:
    answer = answer.strip()

    if not answer or answer.casefold() in AMBIGUOUS_CLARIFICATIONS:
        raise ValueError(
            "Ambiguous clarification. Explain the scientific "
            "context or explicitly state that it is unknown."
        )

    generated = generate_remediation_proposals(
        history=investigation["history"],
        baseline=investigation.get("baseline"),
        clarification={
            "proposal": proposal,
            "answer": answer,
            "dialogue": dialogue,
        },
    )

    candidates = generated["proposals"]

    if len(candidates) != 1:
        raise ValueError("Expected exactly one revised proposal")

    revised = validate_remediation_proposals(
        history=investigation["history"],
        proposals=candidates,
        baseline=investigation.get("baseline"),
    )[0]

    original_evidence = set(proposal.get("evidence_ids", []))
    revised_evidence = set(revised.get("evidence_ids", []))

    if not original_evidence.intersection(revised_evidence):
        raise ValueError(
            "Revised proposal lost its original evidence grounding"
        )

    return revised


def review_interactively(
        investigation: dict,
        dataset_path: Path,
) -> list[dict]:
    raw_proposals = investigation.get("proposals")

    if raw_proposals is None:
        raw_proposals = investigation["remediation"]["proposals"]

    proposals = validate_remediation_proposals(
        history=investigation["history"],
        proposals=raw_proposals,
        baseline=investigation.get("baseline"),
    )

    print(f"\nProposals requiring review: {len(proposals)}")

    reviewer = input("Reviewer ID: ").strip()
    if not reviewer:
        raise ValueError("Reviewer ID is required")

    choices = {
        "a": "accepted",
        "r": "rejected",
        "d": "deferred",
    }
    records = []
    for index, proposal in enumerate(proposals, 1):
        validation = proposal["validation"]

        print(f"\n{'=' * 60}")
        print(f"PROPOSAL {index}/{len(proposals)}")
        print("Issue:", proposal["issue"])
        print("Action:", proposal["proposed_action"])
        print("Target:", proposal.get("target_ref"))
        print("Evidence:", proposal.get("evidence_ids"))
        print("Verified facts:", validation.get("verified_facts", []))
        print("Target verified:", validation["target_verified"])
        print("Validation:", validation["status"])
        print("Uncertainty:", proposal.get("uncertainty"))
        question = proposal.get("review_question", "").strip()

        if question:
            print("Gemma clarification question:", question)
        print("Execution authorized: False")

        previous = find_latest_decision(dataset_path, proposal)

        if previous is not None:
            print("\nPrevious human decision found:")
            print("Decision:", previous["human_decision"])
            print("Reviewer:", previous["reviewer"])
            print("Timestamp:", previous["timestamp"])
            print("Note:", previous.get("human_note", ""))
            print("Execution:", previous["execution_status"])

            repeat = input(
                "Review this proposal again? [y/N]: "
            ).strip().lower()

            if repeat != "y":
                continue
        dialogue = []
        while True:
            execution_plan = proposal.get("execution_plan") or {}

            print("\n" + "=" * 60)
            print("EXACT CLEANING PLAN FOR APPROVAL")
            print("=" * 60)

            for field in (
                    "transformation",
                    "target_selection",
                    "expected_result",
                    "verification_rule",
            ):
                print(f"{field}:")
                print(f"  {execution_plan.get(field, 'NOT SPECIFIED')}")
                print()

            print(
                "Requires clarification:",
                execution_plan.get("requires_clarification", True),
            )

            contract = proposal.get("verification_contract")

            print("\nINDEPENDENT VERIFICATION CONTRACT")

            if contract is None:
                print("  Not available — automatic verification unsupported.")
            else:
                kind = contract.get("kind")
                print(f"  kind: {kind}")

                if kind == "numeric_qc_from_baseline":
                    print("  NUMERIC QC CONDITIONS:")
                    for condition in contract.get("conditions", []):
                        print(
                            f"    issue_type={condition['issue_type']}, "
                            f"columns={condition['columns']}"
                        )

                elif kind == "flag_rows_from_evidence":
                    for field in (
                            "evidence_id",
                            "observation_field",
                            "record_field",
                    ):
                        print(f"  {field}: {contract.get(field, 'NOT SPECIFIED')}")

                for field in (
                        "output_column",
                        "flagged_value",
                        "other_value",
                ):
                    print(f"  {field}: {contract.get(field, 'NOT SPECIFIED')}")
            print(
                "\nGemma's proposed cleaning strategy:"
                "\n  [Yes] Agree with this cleaning approach."
                "\n  [No]  Explain your preferred approach; Gemma revises "
                "its proposal and asks again."
                "\n  [Q]   Quit without approving."
            )

            response = input(
                "\nDo you agree with Gemma's cleaning proposal? "
                "[Yes/No/Q]: "
            ).strip().lower()

            choice = {
                "yes": "a",
                "y": "a",
                "no": "c",
                "n": "c",
                "q": "q",
            }.get(response, "")

            if choice == "c":


                print("Do not enter patient identifiers or confidential data.")
                answer = input("Your feedback or instruction for Gemma: ").strip()

                if not answer or answer.casefold() in AMBIGUOUS_CLARIFICATIONS:
                    print(
                        "Clarification requires a meaningful answer. "
                        "If uncertain, enter 'I don't know' and "
                        "describe what is missing."
                    )
                    continue

                try:
                    revised = revise_proposal_from_answer(
                        investigation=investigation,
                        proposal=proposal,
                        answer=answer,
                        dialogue=dialogue,
                    )
                    clarification_record = record_clarification(
                        dataset_path=dataset_path,
                        original_proposal=proposal,
                        revised_proposal=revised,
                        question=question,
                        answer=answer,
                        reviewer=reviewer,
                    )
                except (RuntimeError, ValueError, KeyError) as exc:
                    print(f"Revision failed; original proposal retained: {exc}")
                    continue

                dialogue.append({
                    "reviewer_feedback": answer,
                    "gemma_revised_proposal": revised,
                })

                proposal = revised
                validation = revised["validation"]
                proposals[index - 1] = revised
                raw_proposals[index - 1] = revised
                question = revised.get("review_question", "").strip()

                print(
                    "Clarification recorded:",
                    clarification_record["clarification_id"],
                )

                print("\nREVISED PROPOSAL — NOT AUTHORIZED")
                print("Issue:", revised["issue"])
                print("Action:", revised["proposed_action"])
                print("Evidence:", revised.get("evidence_ids"))
                print("Validation:", validation["status"])
                print("Uncertainty:", revised.get("uncertainty"))

                if question:
                    print("Gemma clarification question:", question)

                continue

            if choice in choices or choice in ("s", "q"):
                break

            print("Invalid choice. Try again.")

        if choice == "q":
            break

        if choice == "s":
            continue

        if choice == "a":
            if validation["status"] != "needs_human_review":
                print(
                    "Cannot accept: proposal validation "
                    "has not passed."
                )
                continue

            plan = proposal.get("execution_plan") or {}

            if plan.get("requires_clarification", True):
                print(
                    "Cannot accept: the proposed cleaning strategy "
                    "still requires clarification."
                )
                print(
                    "Select No, explain the missing information, "
                    "and let Gemma revise the proposal."
                )
                continue

        print("Do not enter patient identifiers or confidential data.")
        note = input("Reviewer note (optional): ").strip()

        record = record_decision(
            dataset_path=dataset_path,
            proposal=proposal,
            decision=choices[choice],
            reviewer=reviewer,
            note=note,
        )
        records.append(record)

        print("Decision recorded:", record["human_decision"])
        print("Decision ID:", record["decision_id"])
        print("Execution status:", record["execution_status"])

    print("\nHuman review session completed.")
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Human review of LLM remediation proposals"
    )
    parser.add_argument("investigation", type=Path)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    with args.investigation.open(encoding="utf-8") as f:
        investigation = json.load(f)

    review_interactively(investigation, args.dataset)


if __name__ == "__main__":
    main()