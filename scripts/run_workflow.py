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
def _sha256_file(path: Path) -> str:
    """Hash a dataset without loading large files into memory."""
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_scientific_recommendations(audit: dict) -> list[dict]:
    """Create a non-executable, evidence-linked advisory view of proposals.

    This is intentionally NOT an independent scientific correctness verdict.
    Resolved evidence IDs establish provenance, not whether an LLM's prose
    is supported by that evidence. The remediation validator retains sole
    ownership of executable operation eligibility.
    """
    evidence = build_evidence_index(
        audit.get("history") or [],
        baseline=audit.get("baseline"),
    )
    recommendations = []

    for index, proposal in enumerate(audit.get("proposals") or [], start=1):
        evidence_ids = proposal.get("evidence_ids") or []
        resolved = [ref for ref in evidence_ids if ref in evidence]
        unresolved = [ref for ref in evidence_ids if ref not in evidence]
        citations_resolve = bool(evidence_ids) and not unresolved
        issue = (proposal.get("issue") or "").strip()
        method = (proposal.get("proposed_action") or "").strip()
        advisory_status = (
            "provisional_for_client_discussion"
            if citations_resolve and issue and method
            else "insufficient_documentation"
        )

        referenced_tools = {evidence[ref].get("tool") for ref in resolved}
        flags = [
            "Citations identify source observations; scientific interpretation "
            "and suitability of the method are not independently verified."
        ]
        scope = {"status": "not_evaluated"}

        if "find_invalid_counts" in referenced_tools:
            numeric = proposal.get("numeric_coverage")
            if (isinstance(numeric, dict) and
                    type(numeric.get("total_invalid_cells")) is int and
                    type(numeric.get("covered_invalid_cells")) is int and
                    type(numeric.get("uncovered_invalid_cells")) is int):
                total = numeric["total_invalid_cells"]
                covered = numeric["covered_invalid_cells"]
                uncovered = numeric["uncovered_invalid_cells"]
                if total >= 0 and 0 <= covered <= total and uncovered == total - covered:
                    scope = {
                        "status": (
                            "complete_for_invalid_cells"
                            if total > 0 and uncovered == 0 else "partial"
                        ),
                        "covered_invalid_cells": covered,
                        "total_invalid_cells": total,
                        "uncovered_invalid_cells": uncovered,
                    }
                else:
                    scope = {"status": "unverified_numeric_coverage"}
            else:
                scope = {"status": "unverified_numeric_coverage"}

            if scope["status"] == "partial":
                flags.append(
                    "The proposal covers only part of the observed invalid "
                    "count cells; it must not be described as dataset-wide QC."
                )
            elif scope["status"] == "unverified_numeric_coverage":
                flags.append(
                    "The number of invalid count cells covered by this "
                    "method has NOT been independently established."
                )
            flags.append(
                "Missing counts must not silently become zero; negative and "
                "fractional counts require scientific source review before "
                "CLR/rCLR or downstream preprocessing."
            )

        if "compare_columns" in referenced_tools or "inspect_provenance" in referenced_tools:
            flags.append(
                "Matching sample columns do not prove duplicate sample "
                "identity; confirm specimen IDs and collection provenance "
                "before dropping any column."
            )

        if "find_duplicate_taxa" in referenced_tools:
            flags.append(
                "Check whether repeated taxon labels represent identical "
                "features, different assignments, or technical records "
                "before combining or deleting them."
            )

        if ({"find_unclassified_species", "inspect_taxonomy_structure",
             "inspect_taxonomy_naming"} & referenced_tools):
            flags.append(
                "Do not invent missing taxonomic classifications or "
                "silently discard unclassified taxa; require a documented "
                "reference taxonomy and mapping policy."
            )

        if unresolved or not evidence_ids:
            flags.append(
                "Missing or unknown evidence citations prevent even "
                "provisional evidence-linked presentation."
            )

        validation = proposal.get("validation") or {}
        recommendations.append({
            "recommendation_id": f"R{index}",
            "advisory_status": advisory_status,
            "issue": issue,
            "suggested_method": method,
            "rationale": (proposal.get("rationale") or "").strip(),
            "uncertainty": (proposal.get("uncertainty") or "").strip(),
            "evidence_ids": list(evidence_ids),
            "unresolved_evidence_ids": unresolved,
            "all_evidence_ids_resolve": citations_resolve,
            "numeric_scope": scope,
            "scientific_review_flags": flags,
            "requires_client_review": True,
            "execution_eligible": validation.get("status") == "needs_human_review",
            "execution_validator_status": validation.get("status", "unvalidated"),
            "execution_authorized": False,
        })

    return recommendations


def render_scientific_recommendations_markdown(audit: dict) -> str:
    """Append a deliberately non-authorizing advisory section to QC report."""
    recommendations = audit.get("scientific_recommendations") or []
    lines = [
        "\n## Provisional scientific recommendations (not execution approval)",
        "",
        "These are model-generated suggestions. Resolvable citations do NOT",
        "establish that a finding or method is scientifically correct.",
        "The execution validator is a separate safety gate; no cleaning is",
        "authorized by this report.",
        "",
    ]
    if not recommendations:
        lines.extend(["No recommendations generated; this does not imply clean data.", ""])

    for recommendation in recommendations:
        ident = recommendation["recommendation_id"]
        lines.extend([
            f"### {ident}: {recommendation['advisory_status']}",
            "",
            f"**Issue:** {recommendation['issue']}",
            "",
            f"**Method to discuss:** {recommendation['suggested_method']}",
            "",
            f"**Uncertainty:** {recommendation['uncertainty'] or 'Not specified'}",
            "",
            "**Evidence IDs:** " + ", ".join(map(str, recommendation["evidence_ids"])),
            "",
            "**Numeric scope:** `" + json.dumps(recommendation["numeric_scope"], sort_keys=True) + "`",
            "",
            f"**Separate execution-validation result:** `{recommendation['execution_validator_status']}`",
            "",
            "**Required scientific review:**",
        ])
        lines.extend(f"- {flag}" for flag in recommendation["scientific_review_flags"])
        lines.append("")
    return "\n".join(lines) + "\n"


def display_investigation_only_report(
        audit: dict,
        datasets: list[Path],
) -> None:
    """Present evidence and recommendations without requesting any approval.

    This is a *reporting* boundary: no human review, code generation, R
    execution, authorization, or remediation is reached by the default CLI.
    Full evidence, including rejected proposals, remains in the saved JSON.
    """
    print("\n" + "=" * 64)
    print("INVESTIGATION-ONLY CLIENT REHEARSAL")
    print("=" * 64)
    print("No R cleaning code will be generated or executed in this mode.")
    print("No review decisions, execution authorizations, or cleaned outputs")
    print("will be created. These findings are recommendations, not approval.")

    print("\nDatasets:")
    for index, dataset in enumerate(datasets, start=1):
        print(f"  dataset_{index}: {dataset}")

    status = audit.get("status", "unknown")
    print(f"\nGemma investigation status: {status}")
    if status == "no_new_action":
        print("  INCOMPLETE: No further usable investigation action was selected.")
        print("  This is NOT confirmation that all dataset issues are resolved.")
    elif status != "complete":
        print("  Investigation status does not establish full QC coverage.")

    print("\nDeterministic baseline checks (primary dataset only):")
    checks = audit.get("baseline", {}).get("checks", {})
    incomplete_checks = []
    for name, check in checks.items():
        check_status = check.get("status", "unknown")
        count = len(check.get("observations", []))
        print(f"  {name}: {check_status} ({count} observations)")
        if check_status != "completed":
            incomplete_checks.append(name)
        if check.get("downstream_blocked") is True:
            print("    WARNING: Downstream preprocessing is blocked.")
        if isinstance(check.get("counts_by_type"), dict):
            print("    Counts by type:", json.dumps(
                check["counts_by_type"], ensure_ascii=False, sort_keys=True,
            ))

    if incomplete_checks:
        print(f"  COVERAGE GAP: {len(incomplete_checks)} checks not completed.")
    if len(datasets) > 1:
        print("  NOTE: Baseline summary above covers dataset_1 only.")
        print("  Additional dataset profiles/evidence are stored in the audit JSON;")
        print("  full independent cross-dataset QC is not yet demonstrated.")

    history = audit.get("history", [])
    print(f"\nEvidence-bearing investigation steps: {len(history)}")
    print("  Inspect the audit JSON for individual observations and provenance.")

    recommendations = audit.get("scientific_recommendations") or []
    executable_candidates = sum(
        recommendation["execution_eligible"]
        for recommendation in recommendations
    )
    print(f"\nScientific recommendations drafted: {len(recommendations)}")
    print("  These are PROVISIONAL methods for client discussion, not validated cleaning.")
    print(f"  Separate executable-remediation candidates: {executable_candidates}")
    print("  A rejected execution proposal can still reference a genuine QC issue.")

    for recommendation in recommendations:
        print("\n" + "-" * 64)
        print(
            f"RECOMMENDATION {recommendation['recommendation_id']} — "
            f"{recommendation['advisory_status']}"
        )
        print("Issue:", recommendation["issue"] or "Not specified")
        print("Method for discussion:", recommendation["suggested_method"] or "Not specified")
        print("Model uncertainty:", recommendation["uncertainty"] or "Not specified")
        print("Evidence IDs:", ", ".join(map(str, recommendation["evidence_ids"])) or "None")
        print("All cited IDs resolve:", recommendation["all_evidence_ids_resolve"])
        print("  Resolving citations is NOT the same as confirming the claim.")
        scope = recommendation["numeric_scope"]
        if scope.get("status") == "complete_for_invalid_cells":
            print(
                "Count-QC scope: ", scope["covered_invalid_cells"], "/",
                scope["total_invalid_cells"], "observed invalid cells.", sep=""
            )
            print("  This alone does not mean the entire dataset is clean.")
        elif scope.get("status") == "partial":
            print(
                "PARTIAL COUNT-QC SCOPE: ", scope["covered_invalid_cells"], "/",
                scope["total_invalid_cells"], " invalid cells covered; ",
                scope["uncovered_invalid_cells"], " excluded.", sep=""
            )
        elif scope.get("status") == "unverified_numeric_coverage":
            print("Count-QC scope: NOT independently quantified.")

        print("Client review needed:")
        for flag in recommendation["scientific_review_flags"]:
            print("  -", flag)
        print(
            "Execution validator (separate):",
            recommendation["execution_validator_status"],
        )
        print("Execution authorized: False")

    error = audit.get("proposal_generation_error")
    if error:
        print("\nPROPOSAL GENERATION INCOMPLETE:", error)
    elif not recommendations:
        print("\nNo recommendations generated. This does NOT mean the data are clean.")

    print("\nCLIENT DEMO STOP POINT: investigation report and suggestions only.")
    print("No cleaning attempted and no derived dataset created by this mode.")
    print("Full baseline details, evidence, and proposal validation: audit JSON.")


def _assert_read_only_agent_available() -> None:
    """Fail closed if File 2 has not been installed in this project."""
    from inspect import signature
    from healthcare_data_quality.agent_loop import run_investigation

    parameter = signature(run_investigation).parameters.get(
        "allow_r_investigation"
    )
    if parameter is None or parameter.default is not False:
        raise RuntimeError(
            "Investigation-only mode requires the updated "
            "src/healthcare_data_quality/agent_loop.py (File 2), "
            "with allow_r_investigation=False by default. "
            "No investigation was started."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Investigation-first healthcare data-quality workflow"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--dataset",
        action="append",
        type=Path,
        default=[],
        help="Additional dataset for the agent to investigate",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Retained for compatibility; investigation-only mode needs no input",
    )
    parser.add_argument(
        "--enable-remediation",
        action="store_true",
        help=(
            "Explicitly enable the legacy human-review/R-draft/execution "
            "workflow. NOT for investigation-only client rehearsals."
        ),
    )
    args = parser.parse_args()

    if args.enable_remediation and not args.interactive:
        parser.error("--interactive is required with --enable-remediation")

    if not args.input.is_file():
        parser.error(f"Dataset not found: {args.input}")

    for dataset in args.dataset:
        if not dataset.is_file():
            parser.error(f"Additional dataset not found: {dataset}")

    if not args.enable_remediation:
        _assert_read_only_agent_available()

    datasets = [args.input, *args.dataset]
    source_hashes = (
        {p.resolve(): _sha256_file(p) for p in datasets}
        if not args.enable_remediation else {}
    )

    if args.enable_remediation:
        print(
            "[Agent] WARNING: Legacy remediation enabled explicitly. "
            "The workflow may generate R and request execution authorization."
        )
    else:
        print("[Agent] Investigation-only mode: execution is disabled.")

    result = run_initial_audit(
        args.input,
        progress_callback=terminal_progress,
        additional_datasets=args.dataset,
    )

    if source_hashes:
        for path, expected_hash in source_hashes.items():
            if _sha256_file(path) != expected_hash:
                raise RuntimeError(
                    f"Dataset changed during investigation: {path}"
                )

    # Derive a non-executable scientific advisory view before persistence.
    # Legacy execution proposals and their independent validation are unchanged.
    if not args.enable_remediation:
        result["scientific_recommendations"] = (
            build_scientific_recommendations(result)
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

    qc_content = render_qc_summary(result)
    if not args.enable_remediation:
        qc_content += render_scientific_recommendations_markdown(result)

    qc_path.write_text(qc_content, encoding="utf-8")

    print(f"QC report saved: {qc_path.resolve()}")

    # Fail closed before ALL legacy review, draft-generation, authorization,
    # sandbox, and data-changing paths. This is the default client demo mode.
    if not args.enable_remediation:
        display_investigation_only_report(result, datasets)
        if result.get("proposal_generation_error"):
            raise SystemExit(2)
        return

    # The older execution path is retained but must be explicitly requested.
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
