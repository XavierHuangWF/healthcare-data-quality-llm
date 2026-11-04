
import argparse
import json
from pathlib import Path


from healthcare_data_quality.decision_memory import (
    record_decision,
    find_latest_decision,
)

from healthcare_data_quality.remediation_validator import (
    validate_remediation_proposals,
)



def main():
    parser = argparse.ArgumentParser(
        description="Human review of LLM remediation proposals"
    )
    parser.add_argument("investigation", type=Path)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    with args.investigation.open(encoding="utf-8") as f:
        investigation = json.load(f)

    proposals = validate_remediation_proposals(
        investigation["history"],
        investigation["remediation"]["proposals"],
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
        print("Execution authorized: False")

        previous = find_latest_decision(args.dataset, proposal)

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


        while True:
            choice = input(
                "\n[A]ccept / [R]eject / [D]efer / "
                "[S]kip / [Q]uit: "
            ).strip().lower()

            if choice in choices or choice in ("s", "q"):
                break

            print("Invalid choice. Try again.")

        if choice == "q":
            break

        if choice == "s":
            continue

        if (
            choice == "a"
            and validation["status"] != "needs_human_review"
        ):
            print("Cannot accept a rejected proposal.")
            continue

        print("Do not enter patient identifiers or confidential data.")
        note = input("Reviewer note (optional): ").strip()

        record = record_decision(
            dataset_path=args.dataset,
            proposal=proposal,
            decision=choices[choice],
            reviewer=reviewer,
            note=note,
        )

        print("Decision recorded:", record["human_decision"])
        print("Decision ID:", record["decision_id"])
        print("Execution status:", record["execution_status"])

    print("\nHuman review session completed.")


if __name__ == "__main__":
    main()
