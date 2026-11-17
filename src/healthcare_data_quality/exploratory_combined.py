"""Run Gemma exploration after baseline QC within ONE explicitly authorized UI run.

Only executable analyses accepted by the existing AST validator and isolated,
read-only, no-network Bubblewrap runner are allowed. Nothing in this module
modifies an input CSV or executes proposed code on the host. A UI caller must
obtain clear approval for an automated *investigation-scoped* run beforehand.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .exploratory_agent import (
    MAX_ROUNDS, execute_with_auto_repair, propose_exploration,
    validate_generated_code,
)
from .exploratory_dedup import known_issue_catalog, filter_redundant_findings
from .exploratory_sandbox import SandboxUnavailable, sandbox_available


def run_exploratory_series(
    datasets: dict[str, Path],
    baseline_audit: dict,
    *,
    previous: list[dict] | None = None,
    force_first: bool = False,
    on_round: Callable[[list[dict]], None] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Run bounded, verified read-only explorations until Gemma recommends stop.

    Caller obtains user authorization before invocation. Unlike the legacy
    one-proposal UI, the explicit full-run authorization scopes all generated
    read-only sandbox code for this upload, within the MAX_ROUNDS ceiling.
    Only verified runs reach on_round; failure is fatal and preserves earlier
    rounds via the callback. A stop decision is not a proof of data cleanliness.
    """
    if not sandbox_available():
        raise SandboxUnavailable("Bubblewrap sandbox is required for automatic exploratory execution")
    runs = list(previous or [])
    if len(runs) > MAX_ROUNDS:
        raise ValueError("Previous exploratory history exceeds safety ceiling")
    catalog = known_issue_catalog(datasets, baseline_audit)
    if len(runs) == MAX_ROUNDS:
        return {"runs": runs, "stop": None, "safety_ceiling_reached": True}

    while len(runs) < MAX_ROUNDS:
        if on_progress:
            on_progress(f"Gemma is planning exploratory investigation {len(runs) + 1}...")
        proposal = propose_exploration(
            datasets, baseline_audit, runs,
            allow_stop=not (force_first and bool(runs)),
        )
        force_first = False
        if proposal.get("decision") == "stop":
            return {"runs": runs, "stop": proposal, "safety_ceiling_reached": False}
        if on_progress:
            on_progress(f"Running investigation {len(runs) + 1} inside the isolated sandbox...")
        # The runner checks AST/contract again and fails closed if isolation is
        # unavailable. Static-contract errors may be fixed within the same
        # bounded auto-repair window; they do not authorize code on the host.
        initial_failure = None
        try:
            validate_generated_code(proposal["code"])
        except ValueError as exc:
            initial_failure = str(exc)
        verified = execute_with_auto_repair(
            proposal, datasets, initial_failure=initial_failure,
        )
        result, count = filter_redundant_findings(verified["result"], catalog, runs)
        verified["result"] = result
        verified["verification"]["already_reported_findings_suppressed"] = count
        runs.append(verified)
        if on_round:
            on_round(list(runs))
    return {"runs": runs, "stop": None, "safety_ceiling_reached": True}
