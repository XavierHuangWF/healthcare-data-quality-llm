
from pathlib import Path

from .agent_loop import run_investigation
from .decision_memory import (
    get_decision_context,
    get_verified_outcome_context,
)
from .ingestion import inspect_headers
from .remediation_proposal import generate_remediation_proposals
from .remediation_validator import validate_remediation_proposals
from .tools.context import verify_contexts


OUTCOME_LOG = Path(
    "data/decision_memory/remediation_outcomes.jsonl"
)


def run_initial_audit(dataset_path: Path) -> dict:
    dataset_path = Path(dataset_path)

    ingestion = inspect_headers(dataset_path)
    contexts = verify_contexts(dataset_path)

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
            "Investigate potential data-quality problems. "
            "Gather evidence, identify uncertainty, "
            "and recommend next steps. Do not modify data."
        ),
    )

    history = investigation["history"]

    generated = (
        generate_remediation_proposals(
            history,
            decision_memory=prior_decisions,
        )
        if history
        else {"proposals": []}
    )

    proposals = validate_remediation_proposals(
        history=history,
        proposals=generated["proposals"],
    )

    return {
        "status": investigation["status"],
        "ingestion": ingestion,
        "report": investigation["report"],
        "history": history,
        "proposals": proposals,
        "memory": {
            "decisions": prior_decisions,
            "verified_outcomes": prior_outcomes,
        },
    }
