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

from healthcare_data_quality.r_verification import inspect_r_transformation
from hashlib import sha256
from healthcare_data_quality.r_verification import verify_exact_cell_edits
from healthcare_data_quality.r_verification import verify_qc_contract_from_evidence

from healthcare_data_quality.r_verification import verify_numeric_qc_from_baseline
import csv
from io import StringIO

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

def offer_tool_memory(history: list[dict]) -> None:
    """Offer successfully executed R investigation tools for reuse."""

    for step in history:
        if step.get("tool") != "investigate_with_r":
            continue

        observations = step.get("observations", [])

        executed = any(
            isinstance(obs, dict)
            and obs.get("status") == "sandbox_executed"
            for obs in observations
        )

        if not executed:
            continue

        r_code = step.get("arguments", {}).get("r_function")

        if not isinstance(r_code, str) or not r_code.strip():
            continue

        digest = sha256(r_code.encode("utf-8")).hexdigest()
        tool_dir = Path("tools/generated_r")
        tool_path = tool_dir / f"{digest[:16]}.R"
        metadata_path = tool_dir / f"{digest[:16]}.json"

        if tool_path.exists():
            print(f"[Agent] Existing R tool: {tool_path}")
            continue

        print("\n" + "=" * 60)
        print("GEMMA-GENERATED R TOOL")
        print("Purpose:", step.get("reason", "Not specified"))
        print("Sandbox execution: PASS")
        print("Scientific correctness: NOT YET ESTABLISHED")
        print("\nR code:\n")
        print(r_code)

        decision = input(
            "\nSave this R tool as a reusable candidate? [y/N]: "
        ).strip().lower()

        if decision != "y":
            print("[Agent] Tool not saved.")
            continue

        tool_dir.mkdir(parents=True, exist_ok=True)

        tool_path.write_text(r_code, encoding="utf-8")

        metadata = {
            "tool_id": digest[:16],
            "purpose": step.get("reason", ""),
            "code_sha256": digest,
            "execution_status": "sandbox_executed",
            "semantic_verification": "not_established",
            "human_approved_for_storage": True,
            "reuse_requires_fresh_verification": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        print(f"[Agent] Reusable tool candidate saved: {tool_path}")
def merge_verified_qc_column(
        original_csv: str,
        current_csv: str,
        verified_csv: str,
        output_column: str,
) -> str:
    """Add one independently verified QC column to the cumulative result."""

    def parse_csv(content: str) -> list[list[str]]:
        rows = list(csv.reader(StringIO(content)))

        if not rows:
            raise ValueError("Empty CSV")

        width = len(rows[0])

        if any(len(row) != width for row in rows):
            raise ValueError("Inconsistent CSV row widths")

        return rows

    original = parse_csv(original_csv)
    current = parse_csv(current_csv)
    verified = parse_csv(verified_csv)

    width = len(original[0])

    if not (len(original) == len(current) == len(verified)):
        raise ValueError("Record counts differ")

    if [row[:width] for row in current] != original:
        raise ValueError("Cumulative result changed original data")

    if [row[:width] for row in verified] != original:
        raise ValueError("Verified candidate changed original data")

    if len(verified[0]) != width + 1:
        raise ValueError("Candidate must append exactly one column")

    if verified[0][-1] != output_column:
        raise ValueError("QC column does not match approved contract")

    if output_column in current[0]:
        raise ValueError("QC column already exists")

    combined = [
        row + [verified[index][-1]]
        for index, row in enumerate(current)
    ]

    output = StringIO(newline="")
    csv.writer(output).writerows(combined)

    return output.getvalue()

def consolidated_qc_release_status(
        authorized_count: int,
        verified_count: int,
        original_hash: str,
        current_hash: str,
) -> str:
    if authorized_count < 1:
        return "not_authorized"

    if verified_count != authorized_count:
        return "verification_incomplete"

    if current_hash != original_hash:
        return "source_changed"

    return "ready"
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive healthcare data-quality workflow"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--dataset",
        action="append",
        type=Path,
        default=[],
        help="Additional dataset for the agent to investigate",
    )
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if not args.interactive:
        parser.error("--interactive is required")

    if not args.input.is_file():
        parser.error(f"Dataset not found: {args.input}")

    for dataset in args.dataset:
        if not dataset.is_file():
            parser.error(f"Additional dataset not found: {dataset}")

    result = run_initial_audit(
        args.input,
        progress_callback=terminal_progress,
        additional_datasets=args.dataset,
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
    offer_tool_memory(result["history"])
    if args.dataset:
        print("\n[Agent] Multi-dataset investigation completed.")
        print("[Agent] Dataset investigation results saved in audit report.")

        for index, dataset in enumerate(
            [args.input, *args.dataset], start=1
        ):
            print(f"  dataset_{index}: {dataset}")

        print(
            "[Agent] Cleaning is temporarily disabled for multi-dataset "
            "runs until proposals and R execution are bound to the "
            "correct source dataset."
        )
        return

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

    reviewable_proposals = [
        proposal
        for proposal in result["proposals"]
        if proposal.get("validation", {}).get("status")
           == "needs_human_review"
    ]

    rejected_count = len(result["proposals"]) - len(reviewable_proposals)

    print(f"Proposals rejected before review: {rejected_count}")
    print(f"Proposals eligible for review: {len(reviewable_proposals)}")

    if not reviewable_proposals:
        print("[Agent] No validated cleaning proposals available for approval.")
        print("[Agent] No cleaning performed. Original dataset unchanged.")
        return

    review_result = {
        **result,
        "proposals": reviewable_proposals,
    }

    decisions = review_interactively(review_result, args.input)

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
        original_csv = args.input.read_text(encoding="utf-8-sig")
        original_source_hash = sha256(args.input.read_bytes()).hexdigest()
        cumulative_csv = original_csv
        merged_qc_count = 0
        authorized_qc_count = 0
        draft_dir = Path("data/cleaning_drafts")
        draft_dir.mkdir(parents=True, exist_ok=True)

        for proposal in review_result["proposals"]:
            proposal_id = fingerprint_proposal(proposal)
            operation = proposal.get("operation") or {}

            review = accepted_reviews.get(proposal_id)

            if review is None:
                continue

            print("\n[Agent] Generating R cleaning draft with Gemma...")

            reviewer_context = (review.get("human_note") or "").strip()

            try:
                draft = draft_r_transformation(
                    proposal=proposal,
                    evidence=evidence_index,
                    reviewer_context=reviewer_context,
                )
            except (RuntimeError, ValueError) as exc:
                print(f"[Agent] R tool generation failed: {exc}")
                continue

            if draft["status"] == "needs_clarification":
                print(
                    "[Agent] The agreed cleaning strategy is not executable. "
                    "No cleaning performed. Gemma must resolve the strategy "
                    "before requesting Yes."
                )
                continue


            verification = {"status": "not_run"}
            draft_statuses[proposal_id] = draft["status"]

            if (
                    draft["status"] == "draft_ready"
                    and operation.get("kind") == "drop_duplicate_column"
            ):
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
                    if verification.get("status") == "verified":
                        draft_statuses[proposal_id] = "synthetic_verified"
                        print("[Agent] R synthetic verification: PASS")
                    else:
                        draft_statuses[proposal_id] = "verification_failed"
                        print("[Agent] R synthetic verification: FAIL")


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
                        "reviewer_context": reviewer_context,
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
            elif draft_statuses[proposal_id] == "verification_failed":
                print(
                    "[Agent] R synthetic verification failed. "
                    "Authorization blocked."
                )

            elif draft["status"] == "draft_ready":
                print(
                    "\n[Agent] Cleaning proposal accepted, "
                    "but execution is not yet authorized."
                )

                print(
                    "[Agent] R draft is ready for sandbox execution. "
                    "The original dataset will not be overwritten."
                )

                consent = input(
                    "Authorize executing this R transformation and "
                    "saving a cleaned candidate if verification passes? [y/N]: "
                ).strip().lower()

                if consent == "y":
                    authorized_qc_count += 1
                    source_hash = original_source_hash
                    try:
                        preview = inspect_r_transformation(
                            r_function=draft["r_function"],
                            input_csv=args.input.read_text(
                                encoding="utf-8-sig"
                            ),
                        )
                    except (ValueError, RuntimeError, OSError, UnicodeError) as exc:
                        print(f"[Agent] Sandbox preview failed: {exc}")
                    else:
                        print("\n[Agent] R transformation preview:")
                        for field in (
                                "status",
                                "semantic_verification",
                                "input_records",
                                "output_records",
                                "input_columns",
                                "output_columns",
                                "headers_changed",
                                "content_changed",
                        ):
                            print(f"  {field}: {preview[field]}")

                        if sha256(args.input.read_bytes()).hexdigest() != source_hash:
                            raise RuntimeError(
                                "Source dataset changed during cleaning. Output blocked."
                            )

                        # Verify the sandbox output against the approved
                        # contract and independently collected evidence.
                        contract = proposal.get("verification_contract")

                        if not isinstance(contract, dict):
                            print("[Agent] Missing verification contract.")
                            print("[Agent] Candidate rejected. No output saved.")
                            continue

                        if proposal.get("validation", {}).get(
                                "verification_contract_supported"
                        ) is not True:
                            print("[Agent] Unsupported verification contract.")
                            print("[Agent] Candidate rejected. No output saved.")
                            continue

                        source_csv = args.input.read_text(
                            encoding="utf-8-sig"
                        )

                        try:
                            kind = contract.get("kind")

                            if kind == "flag_rows_from_evidence":
                                if contract.get("evidence_id") not in proposal.get(
                                        "evidence_ids", []
                                ):
                                    raise ValueError(
                                        "Contract evidence not cited by proposal"
                                    )

                                checked = verify_qc_contract_from_evidence(
                                    source_csv=source_csv,
                                    output_csv=preview["output_csv"],
                                    contract=contract,
                                    evidence_index=evidence_index,
                                )

                                expected_verification = (
                                    "independent_evidence_match"
                                )

                            elif kind == "numeric_qc_from_baseline":
                                if not any(
                                        evidence_index.get(
                                            evidence_id, {}
                                        ).get("tool") == "find_invalid_counts"
                                        for evidence_id in proposal.get(
                                            "evidence_ids", []
                                        )
                                ):
                                    raise ValueError(
                                        "Numeric baseline evidence not cited"
                                    )

                                checked = verify_numeric_qc_from_baseline(
                                    source_csv=source_csv,
                                    output_csv=preview["output_csv"],
                                    baseline=result["baseline"],
                                    conditions=contract["conditions"],
                                    flag_column=contract["output_column"],
                                    flagged_value=contract["flagged_value"],
                                    other_value=contract["other_value"],
                                )

                                expected_verification = (
                                    "independent_baseline_match"
                                )

                            else:
                                raise ValueError(
                                    f"Unsupported contract kind: {kind}"
                                )

                        except (ValueError, KeyError, TypeError) as exc:
                            print(
                                f"[Agent] Independent verification: FAIL — {exc}"
                            )
                            print("[Agent] Candidate rejected. No output saved.")
                            continue

                        if (
                                checked.get("status") != "qc_flag_contract_matched"
                                or checked.get("verification")
                                != expected_verification
                        ):
                            print("[Agent] Independent verification: FAIL")
                            print("[Agent] Candidate rejected. No output saved.")
                            continue

                        print(
                            "[Agent] Independent QC verification: PASS",
                            f"({checked['flagged_records']} flagged records)",
                        )
                        try:
                            next_cumulative_csv = merge_verified_qc_column(
                                original_csv=original_csv,
                                current_csv=cumulative_csv,
                                verified_csv=preview["output_csv"],
                                output_column=contract["output_column"],
                            )
                        except (ValueError, KeyError, TypeError) as exc:
                            print(
                                "[Agent] Cumulative QC merge rejected:",
                                exc,
                            )
                            continue

                        cumulative_csv = next_cumulative_csv
                        merged_qc_count += 1

                        print(
                            "[Agent] Independently verified QC column "
                            "added to cumulative result:",
                            contract["output_column"],
                        )
                        print(
                            "[Agent] Cumulative verified operations:",
                            merged_qc_count,
                        )
                        print(
                            "[Agent] Verified QC transformation staged "
                            "for consolidated output."
                        )
                        print("[Agent] Original SHA-256:", original_source_hash)
                        print("[Agent] Technical sandbox execution: PASS")
                        print("[Agent] Independent QC contract verification: PASS")
                else:
                    print("[Agent] Sandbox preview skipped.")

            print("[Agent] No generated R code executed on the source dataset.")

    if authorized_qc_count:
        release_status = consolidated_qc_release_status(
            authorized_count=authorized_qc_count,
            verified_count=merged_qc_count,
            original_hash=original_source_hash,
            current_hash=sha256(args.input.read_bytes()).hexdigest(),
        )

        if release_status == "verification_incomplete":
            print(
                "\n[Agent] CONSOLIDATED OUTPUT BLOCKED: "
                f"{merged_qc_count}/{authorized_qc_count} "
                "authorized transformations verified."
            )
            print(
                "[Agent] No consolidated candidate released. "
                "Original dataset unchanged."
            )


        elif release_status == "source_changed":
            raise RuntimeError(
                "Original dataset changed during cleaning. "
                "Consolidated output blocked."
            )

        else:
            output_dir = Path("data/derived")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / (
                f"{args.input.stem}_{timestamp}_"
                "consolidated_candidate.csv"
            )

            with output_path.open(
                    "x", encoding="utf-8", newline=""
            ) as file:
                file.write(cumulative_csv)

            if sha256(args.input.read_bytes()).hexdigest() != original_source_hash:
                output_path.unlink()
                raise RuntimeError(
                    "Original dataset changed. "
                    "Consolidated candidate discarded."
                )

            print(
                "\n[Agent] CONSOLIDATED CANDIDATE SAVED:",
                output_path.resolve(),
            )
            print(
                "[Agent] Independently verified QC operations:",
                merged_qc_count,
            )
            print("[Agent] Original SHA-256:", original_source_hash)
            print("[Agent] Original dataset unchanged.")
    eligible_proposals = [
        proposal
        for proposal in review_result["proposals"]
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
                    operation_id=audit["operation_id"],
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
