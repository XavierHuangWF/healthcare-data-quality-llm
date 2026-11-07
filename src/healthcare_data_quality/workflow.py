from pathlib import Path

from .agent_loop import run_investigation
from .decision_memory import (
    get_decision_context,
    get_verified_outcome_context,
)
from .ingestion import inspect_headers
from .remediation_proposal import generate_remediation_proposals
from .remediation_validator import validate_remediation_proposals
from collections.abc import Callable
from .baseline_audit import run_baseline_audit
import json

OUTCOME_LOG = Path(
    "data/decision_memory/remediation_outcomes.jsonl"
)


def run_initial_audit(
        dataset_path: Path,
        progress_callback: Callable[[str], None] | None = None,
) -> dict:
    dataset_path = Path(dataset_path)

    ingestion = inspect_headers(dataset_path)

    if progress_callback is not None:
        progress_callback("Running automatic baseline checks...")

    baseline = run_baseline_audit(dataset_path)
    contexts = set(baseline["verified_contexts"])
    baseline_summary = {}

    for name, check in baseline["checks"].items():
        summary = {
            "status": check["status"],
            "observation_count": len(
                check.get("observations", [])
            ),
        }

        for field in (
                "counts_by_type",
                "downstream_blocked",
                "incomplete_count",
                "alternate_delimiter_count",
                "unclassified_count",
                "counts_by_style",
        ):
            if field in check:
                summary[field] = check[field]

        baseline_summary[name] = summary

    if progress_callback is not None:
        progress_callback("Automatic baseline checks completed.")

    prior_decisions = get_decision_context(dataset_path)

    prior_outcomes = get_verified_outcome_context(
        dataset_path,
        OUTCOME_LOG,
    )

    investigation = run_investigation(
        path=dataset_path,
        max_steps=4,
        verified_contexts=contexts,
        user_request=(
            "The automatic baseline audit has completed. "
            "Use its coverage summary to prioritize further "
            "investigation and unresolved issues. "
            "Do not repeat completed checks unnecessarily. "
            "The summary is context, not independently citable "
            "evidence; factual proposals still require verified "
            "tool observations. "
            "Do not modify data. "
            "Baseline coverage summary: "
            f"{json.dumps(baseline_summary)}"
        ),
        progress_callback=progress_callback,
    )

    history = investigation["history"]

    proposal_generation_error = None

    try:
        generated = (
            generate_remediation_proposals(
                history,
                decision_memory=prior_decisions,
                baseline=baseline,
            )
            if history
            else {"proposals": []}
        )
    except RuntimeError as exc:
        if "Incomplete generation: length" not in str(exc):
            raise

        proposal_generation_error = str(exc)
        generated = {"proposals": []}

        if progress_callback is not None:
            progress_callback(
                "Proposal generation incomplete. "
                "Baseline audit results retained."
            )

    proposals = validate_remediation_proposals(
        history=history,
        proposals=generated["proposals"],
        baseline=baseline,
    )

    return {
        "status": investigation["status"],
        "ingestion": ingestion,
        "baseline": baseline,
        "proposal_generation_error": proposal_generation_error,
        "report": investigation["report"],
        "history": history,
        "proposals": proposals,
        "memory": {
            "decisions": prior_decisions,
            "verified_outcomes": prior_outcomes,
        },

    }
