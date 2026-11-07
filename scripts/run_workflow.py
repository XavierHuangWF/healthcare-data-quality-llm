import argparse
import sys
from pathlib import Path

from healthcare_data_quality.workflow import run_initial_audit
from review_proposals import review_interactively
from healthcare_data_quality.decision_memory import fingerprint_proposal
from authorize_duplicate_removal import authorize_duplicate_removal
from apply_duplicate_removal import apply_duplicate_removal
from verify_remediation import (
    verify_remediation,
    record_verified_outcome,
)
import json
from datetime import datetime, timezone

from healthcare_data_quality.qc_report import render_qc_summary
from healthcare_data_quality.cleaning_codegen import draft_r_transformation
from healthcare_data_quality.investigation_report import build_evidence_index


from healthcare_data_quality.r_verification import verify_duplicate_column_draft

def terminal_progress(message: str) -> None:
    print(f"[Agent] {message}", file=sys.stderr, flush=True)


def eligible_for_authorization(
        proposal: dict,
        accepted_ids: set[str],
        draft_statuses: dict[str, str],
) -> bool:
    proposal_id = fingerprint_proposal(proposal)

    operation = proposal.get("operation") or {}
    target = proposal.get("target_ref") or {}

    if (
            operation.get("kind") != "drop_duplicate_column"
            or target.get("kind") != "column_pair"
            or target.get("positions") != [
        operation.get("keep_position"),
        operation.get("drop_position"),
    ]
    ):
        return False

    return (
            proposal_id in accepted_ids
            and draft_statuses.get(proposal_id) == "synthetic_verified"
            and proposal.get("action_kind") == "data_change"
            and proposal.get("validation", {}).get("target_verified") is True
            and proposal.get("validation", {}).get("operation_supported") is True
            and proposal.get("validation", {}).get("status") == "needs_human_review"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive healthcare data-quality workflow"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if not args.interactive:
        parser.error("--interactive is required")

    if not args.input.is_file():
        parser.error(f"Dataset not found: {args.input}")

    result = run_initial_audit(
        args.input,
        progress_callback=terminal_progress,
    )

    report_dir = Path("data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    report_path = report_dir / (
        f"{args.input.stem}_{timestamp}_audit.json"
    )

    with report_path.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print(f"\nAudit snapshot saved: {report_path.resolve()}")

    qc_path = report_path.with_suffix(".md")

    qc_path.write_text(
        render_qc_summary(result),
        encoding="utf-8",
    )

    print(f"QC report saved: {qc_path.resolve()}")

    error = result.get("proposal_generation_error")

    if error:
        print(f"\nPROPOSAL GENERATION: INCOMPLETE ({error})")
        print("\nAutomatic baseline results:")

        for name, check in result["baseline"]["checks"].items():
            if check["status"] == "completed":
                count = len(check.get("observations", []))
                print(f"  {name}: {count} observations")

        print("\nNo cleaning proposals were finalized.")
        print("Human authorization and cleaning were not attempted.")
        raise SystemExit(2)

    print(f"\nInvestigation status: {result['status']}")
    print(f"Proposals generated: {len(result['proposals'])}")

    decisions = review_interactively(result, args.input)

    print(f"\nDecisions recorded this session: {len(decisions)}")
    accepted_ids = {
        decision["proposal_id"]
        for decision in decisions
        if decision["human_decision"] == "accepted"
    }

    accepted_reviews = {
        decision["proposal_id"]: decision
        for decision in decisions
        if decision["human_decision"] == "accepted"
    }

    draft_statuses = {}

    if accepted_reviews:
        evidence_index = build_evidence_index(
            result["history"],
            baseline=result["baseline"],
        )

        draft_dir = Path("data/cleaning_drafts")
        draft_dir.mkdir(parents=True, exist_ok=True)

        for proposal in result["proposals"]:
            proposal_id = fingerprint_proposal(proposal)
            operation = proposal.get("operation") or {}

            review = accepted_reviews.get(proposal_id)

            if review is None:
                continue

            print("\n[Agent] Generating R cleaning draft with Gemma...")

            try:
                draft = draft_r_transformation(
                    proposal=proposal,
                    evidence=evidence_index,
                    reviewer_context=(
                            review.get("human_note")
                            or "No scientific confirmation supplied. "
                               "Preserve uncertain data and request "
                               "clarification rather than guessing."
                    ),
                )

            except (RuntimeError, ValueError) as exc:
                print(f"[Agent] R draft generation incomplete: {exc}")
                continue


            verification = {"status": "not_run"}
            draft_statuses[proposal_id] = draft["status"]

            if draft["status"] == "draft_ready":
                try:
                    verification = verify_duplicate_column_draft(
                        r_function=draft["r_function"],
                        keep_position=operation["keep_position"],
                        drop_position=operation["drop_position"],
                    )
                except (ValueError, RuntimeError, KeyError) as exc:
                    verification = {
                        "status": "failed",
                        "reason": str(exc),
                    }
                    draft_statuses[proposal_id] = "verification_failed"
                else:
                    draft_statuses[proposal_id] = "synthetic_verified"
                    print("[Agent] R synthetic verification: PASS")


            timestamp = datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%S%fZ"
            )

            draft_path = draft_dir / (
                f"{args.input.stem}_{proposal_id[:16]}_{timestamp}.json"
            )

            with draft_path.open("x", encoding="utf-8") as file:
                json.dump(
                    {
                        "proposal_id": proposal_id,
                        "source": str(args.input.resolve()),
                        "review_decision_id": review["decision_id"],
                        "draft": draft,
                        "synthetic_verification": verification,
                        "execution_status": "not_executed",
                    },
                    file,
                    indent=2,
                )

            print("[Agent] R draft status:", draft["status"])
            print("[Agent] R draft saved:", draft_path)

            if draft["status"] == "needs_clarification":
                print(
                    "[Agent] Unresolved uncertainty. "
                    "No executable cleaning plan approved."
                )
            elif draft_statuses[proposal_id] == "synthetic_verified":
                print(
                    "[Agent] R draft passed synthetic verification. "
                    "Separate human authorization is still required."
                )
            elif draft["status"] == "draft_ready":
                print(
                    "[Agent] R synthetic verification failed. "
                    "Authorization blocked."
                )

            print("[Agent] No generated R code executed on the source dataset.")

    eligible_proposals = [
        proposal
        for proposal in result["proposals"]
        if eligible_for_authorization(
            proposal, accepted_ids, draft_statuses
        )
    ]

    print(
        "Proposals eligible for separate authorization:",
        len(eligible_proposals),
    )

    if not eligible_proposals:
        print("No validated cleaning operation is eligible for authorization.")
        print("Original dataset remains unchanged.")
    else:

        for proposal in eligible_proposals:
            operation = proposal.get("operation") or {}

            if operation.get("kind") != "drop_duplicate_column":
                print("Unsupported cleaning operation. Skipping.")
                continue

            proposal_id = fingerprint_proposal(proposal)

            decision = next(
                (
                    record
                    for record in decisions
                    if record["proposal_id"] == proposal_id
                       and record["human_decision"] == "accepted"
                ),
                None,
            )

            if decision is None:
                raise RuntimeError("Accepted review decision missing")

            output = Path("data/derived") / (
                f"{args.input.stem}_cleaned.csv"
            )

            authorization = authorize_duplicate_removal(
                source=args.input,
                output=output,
                keep_position=operation["keep_position"],
                drop_position=operation["drop_position"],
                decision_id=decision["decision_id"],
            )

            if authorization is None:
                print("Authorization declined. No data changed.")
            else:
                print("Authorization recorded successfully.")

                audit = apply_duplicate_removal(
                    source=args.input,
                    output=output,
                    authorization_id=authorization["authorization_id"],
                )

                verification = verify_remediation(
                    source=args.input,
                    output=output,
                    audit_path=Path(
                        "data/decision_memory/remediation_runs.jsonl"
                    ),
                )

                recorded = record_verified_outcome(
                    verification=verification,
                    source=args.input,
                    output=output,
                    log_path=Path(
                        "data/decision_memory/remediation_outcomes.jsonl"
                    ),
                )

                print("\nPOST-REMEDIATION VERIFICATION: PASS")
                print("Cleaned copy:", output.resolve())
                print("Technical verification:", verification["technical_verification"])
                print("Semantic verification:", verification["semantic_verification"])
                print("Verified outcome recorded:", recorded)


if __name__ == "__main__":
    main()
