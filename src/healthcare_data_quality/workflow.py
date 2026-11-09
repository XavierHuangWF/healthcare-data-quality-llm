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
import csv
from collections import Counter, defaultdict
from .tools.registry import TOOL_REGISTRY
from .tools.context import STRUCTURE_CONTEXT

OUTCOME_LOG = Path(
    "data/decision_memory/remediation_outcomes.jsonl"
)




# Observed results and scientific decisions are different objectives. Completing
# a read-only check does not establish that a proposed data change is correct.
# This ledger is serialized in the existing audit JSON by run_workflow.py.
_OBJECTIVE_VERSION = 1
_NONEXECUTING_TOOLS = frozenset({
    "inspect_column", "inspect_columns", "compare_columns",
    "inspect_provenance", "observe_matrix_structure",
    "find_duplicate_ids", "find_duplicate_conflicts",
    "summarize_invalid_counts",
})


def _tool_capability(tool: str, verified_contexts: set[str]) -> dict:
    """Determine availability from the registry, not from Gemma's assertions."""
    spec = TOOL_REGISTRY.get(tool)
    available = (
        tool in _NONEXECUTING_TOOLS
        and spec is not None
        and spec.requires_context in verified_contexts
    )
    return {
        "tool": tool,
        "available": available,
        "requires_context": spec.requires_context if spec is not None else None,
        "mode": "registered_read_only" if available else "unavailable",
    }


def _duplicate_header_pairs(path: Path, limit: int = 24) -> tuple[list[tuple[int, int]], bool]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        headers = next(csv.reader(handle), [])
    positions: dict[str, list[int]] = defaultdict(list)
    for index, raw_name in enumerate(headers, start=1):
        if raw_name.strip():
            positions[raw_name].append(index)
    pairs = []
    truncated = False
    for indices in positions.values():
        if len(indices) < 2:
            continue
        for i, left in enumerate(indices):
            for right in indices[i + 1:]:
                if len(pairs) >= limit:
                    truncated = True
                    return pairs, truncated
                pairs.append((left, right))
    return pairs, truncated


def build_investigation_objective_ledger(
        dataset_paths: dict[str, Path],
        baselines: dict[str, dict],
        history: list[dict],
        *,
        failed_objective_ids: set[str] | None = None,
) -> dict:
    """Record deterministic coverage, remaining tool checks, and decisions.

    The ledger uses stable per-dataset objective identifiers and evidence
    locations. It never treats a model-selected tool or a model summary as
    proof of scientific correctness. 'resolved' means the *specified fact*
    was measured, not that downstream preprocessing is authorized.

    `failed_objective_ids` originates from actual failed read-only tool calls.
    A failed, unresolved tool objective is marked blocked with a specific
    reason. A successfully completed objective is never downgraded by an
    earlier failure, and scientific client decisions are not automatically
    classified as tool failures.
    """
    objectives: list[dict] = []
    completed_comparisons: dict[tuple[str, int, int], list[str]] = {}
    completed_tools: set[tuple[str, str]] = set()
    for item in history:
        if not item.get("observations"):
            continue
        dataset_id = item.get("dataset_id", "dataset_1")
        tool = item.get("tool", "")
        completed_tools.add((dataset_id, tool))
        if tool == "compare_columns":
            arguments = item.get("arguments") or {}
            left, right = arguments.get("left"), arguments.get("right")
            if type(left) is int and type(right) is int and left != right:
                pair = tuple(sorted((left, right)))
                completed_comparisons[(dataset_id, *pair)] = [
                    f"T{item['step']}.O{n}" for n in range(1, len(item["observations"]) + 1)
                ]

    def add(dataset_id: str, key: str, title: str, status: str, *,
            source: str, finding: str, next_action: str,
            capability: dict | None = None, observation_count: int | None = None,
            evidence_refs: list[str] | None = None) -> None:
        item = {
            "objective_id": f"{dataset_id}:{key}",
            "dataset_id": dataset_id,
            "title": title,
            "status": status,
            "source": source,
            "finding": finding,
            "next_action": next_action,
            "evidence_refs": evidence_refs or [],
        }
        if capability is not None:
            item["tool_capability"] = capability
        if observation_count is not None:
            item["observation_count"] = observation_count
        objectives.append(item)

    for dataset_id, dataset_path in dataset_paths.items():
        baseline = baselines[dataset_id]
        contexts = set(baseline.get("verified_contexts", []))
        checks = baseline["checks"]
        for check_name, check in checks.items():
            # Skipped checks on an inapplicable dataset are not QC failures:
            # a taxonomy matrix need not have SampleID/NPDNum metadata columns.
            if check.get("status") != "completed":
                continue
            n = len(check.get("observations", []))
            add(
                dataset_id,
                f"baseline:{check_name}",
                f"Measure {check_name.replace('_', ' ')}",
                "resolved", source="deterministic_baseline",
                finding=f"Read-only check completed with {n} observations. This is not a cleaning decision.",
                next_action="Use the recorded observations when proposing evidence-linked methods.",
                observation_count=n,
                evidence_refs=[f"dataset_baselines.{dataset_id}.checks.{check_name}"],
            )

        if checks.get("find_invalid_counts", {}).get("status") == "completed":
            numeric = checks["find_invalid_counts"]
            types = dict(numeric.get("counts_by_type", {}))
            total = sum(types.values())
            if total:
                add(
                    dataset_id, "decision:invalid_count_policy",
                    "Determine whether and how invalid counts should be corrected",
                    "requires_client_input", source="scientific_policy",
                    finding=f"{total} invalid cells: {json.dumps(types, sort_keys=True)}.",
                    next_action=(
                        "Request source-of-truth and missing/negative/fractional count policy; "
                        "do not silently replace missing counts with zero or run CLR/rCLR."
                    ),
                    evidence_refs=[f"dataset_baselines.{dataset_id}.checks.find_invalid_counts"],
                )

        if checks.get("find_duplicate_taxa", {}).get("observations"):
            add(
                dataset_id, "decision:duplicate_taxa_policy",
                "Determine whether duplicate taxon labels are equivalent features",
                "requires_client_input", source="scientific_policy",
                finding="Repeated taxon labels identified by the deterministic baseline.",
                next_action="Confirm taxonomy provenance before merging, summing, or removing taxa.",
                evidence_refs=[f"dataset_baselines.{dataset_id}.checks.find_duplicate_taxa"],
            )

        naming_styles = checks.get("inspect_taxonomy_naming", {}).get("counts_by_style", {})
        if sum(int(value > 0) for value in naming_styles.values()) > 1:
            add(
                dataset_id, "decision:taxonomy_naming_convention",
                "Choose a consistent representation for taxonomy labels",
                "requires_client_input", source="scientific_policy",
                finding="Multiple taxonomy naming styles occur in the same dataset.",
                next_action="Agree on the taxonomy naming standard; preserve original labels and provenance in any future mapping.",
                evidence_refs=[f"dataset_baselines.{dataset_id}.checks.inspect_taxonomy_naming"],
            )

        if checks.get("inspect_taxonomy_structure", {}).get("incomplete_count", 0) or checks.get(
                "find_unclassified_species", {}).get("unclassified_count", 0):
            add(
                dataset_id, "decision:taxonomy_mapping_policy",
                "Establish taxonomy mapping and unknown-species handling",
                "requires_client_input", source="scientific_policy",
                finding="Incomplete or unclassified taxonomic assignments were detected.",
                next_action="Obtain accepted taxonomy reference/version and rules for retaining unknown taxa.",
                evidence_refs=[f"dataset_baselines.{dataset_id}.checks.inspect_taxonomy_structure"],
            )

        if checks.get("find_duplicate_ids", {}).get("observations"):
            add(
                dataset_id, "decision:duplicate_metadata_id_policy",
                "Determine which repeated metadata identifiers represent true duplicates",
                "requires_client_input", source="scientific_policy",
                finding="Repeated sample identifiers detected in metadata.",
                next_action="Confirm whether repeated identifiers correspond to technical replicates, separate records, or entry errors.",
                evidence_refs=[f"dataset_baselines.{dataset_id}.checks.find_duplicate_ids"],
            )

        if checks.get("find_duplicate_conflicts", {}).get("observations"):
            add(
                dataset_id, "decision:metadata_conflict_policy",
                "Determine authority of conflicting metadata records",
                "requires_client_input", source="scientific_policy",
                finding="Conflicting records were detected for repeated metadata identifiers.",
                next_action="Request authoritative source and reconciliation rules; do not choose a winner automatically.",
                evidence_refs=[f"dataset_baselines.{dataset_id}.checks.find_duplicate_conflicts"],
            )

        pairs, truncated = _duplicate_header_pairs(dataset_path)
        comparison_tool = _tool_capability("compare_columns", contexts)
        inspected_columns = set()
        for item in history:
            if item.get("dataset_id", "dataset_1") != dataset_id or not item.get("observations"):
                continue
            args = item.get("arguments") or {}
            if item.get("tool") == "inspect_column" and type(args.get("position")) is int:
                inspected_columns.add(args["position"])
            if item.get("tool") == "inspect_columns" and isinstance(args.get("positions"), list):
                inspected_columns.update(pos for pos in args["positions"] if type(pos) is int)
        for left, right in pairs:
            evidence_refs = completed_comparisons.get((dataset_id, left, right), [])
            status = (
                "resolved" if evidence_refs else
                "blocked" if not comparison_tool["available"] else
                "in_progress" if {left, right}.intersection(inspected_columns) else
                "uninvestigated"
            )
            add(
                dataset_id,
                f"compare_columns:{left}:{right}",
                f"Compare duplicate-header columns {left} and {right}",
                status, source="agent_tool_objective",
                finding=(
                    "Column values compared and observations recorded. Sample identity remains unverified."
                    if evidence_refs else "Identically named headers require an independent value comparison."
                ),
                next_action=(
                    "Confirm specimen provenance before deleting either column."
                    if evidence_refs else
                    "Run compare_columns on this exact pair if the tool is available."
                ),
                capability=comparison_tool,
                evidence_refs=evidence_refs,
            )
        if truncated:
            add(
                dataset_id, "capacity:duplicate_header_pairs",
                "Inspect additional repeated-header column pairs",
                "blocked", source="workload_safety_limit",
                finding="More than 24 candidate comparisons exist; only the first 24 were scheduled.",
                next_action="Expand the bounded comparison planner after reviewing dataset structure.",
            )
        if pairs:
            provenance_tool = _tool_capability("inspect_provenance", contexts)
            ran = (dataset_id, "inspect_provenance") in completed_tools
            add(
                dataset_id, "provenance:duplicate_headers",
                "Examine lineage of duplicated column names",
                "resolved" if ran else "uninvestigated" if provenance_tool["available"] else "blocked",
                source="agent_tool_objective",
                finding=(
                    "Provenance tool returned observations; these may still be inconclusive."
                    if ran else "Dataset-bound provenance has not yet been inspected."
                ),
                next_action="Confirm actual sample identity with the client/source system.",
                capability=provenance_tool,
            )
            add(
                dataset_id, "decision:duplicate_sample_identity",
                "Verify whether repeated-header columns represent duplicate samples",
                "requires_client_input", source="scientific_policy",
                finding="Duplicate headers alone do not prove repeated samples.",
                next_action="Request sample collection identifiers and documented lineage before removal.",
            )

    if len(dataset_paths) > 1:
        # The registered investigation tools all accept a single CSV path.
        # There is currently no registered read-only cross-dataset matcher.
        add(
            "workspace", "cross_dataset_id_reconciliation",
            "Reconcile identifiers across datasets",
            "blocked", source="tool_capability_gap",
            finding=(
                "The codebase contains an identifier matching utility, but no registered "
                "read-only agent tool accepts a multi-dataset mapping contract."
            ),
            next_action=(
                "Integrate and verify the existing identifier_matching.compare_sample_ids "
                "utility as a registered read-only multi-dataset tool; test ambiguous "
                "and unmatched IDs and require client confirmation."
            ),
            capability={"tool": None, "available": False, "mode": "missing_cross_dataset_tool"},
        )

    # The live agent excludes failed objectives from further automatic retry
    # attempts. Rebuild that outcome here so the *persisted* ledger agrees
    # with the actual execution trace instead of showing them uninvestigated.
    reported_failed_ids = {
        value for value in (failed_objective_ids or set())
        if isinstance(value, str) and value
    }
    ledger_ids = {item["objective_id"] for item in objectives}
    unresolved_failed_ids = []
    for item in objectives:
        if item["objective_id"] not in reported_failed_ids:
            continue
        if item["status"] == "resolved":
            # A later successful observation takes precedence over a failure.
            continue
        if item["source"] != "agent_tool_objective":
            # An execution failure cannot resolve or reclassify a policy
            # decision or an unrelated baseline/capacity objective.
            continue
        item["status"] = "blocked"
        item["block_reason"] = "read_only_tool_execution_failed"
        item["last_attempt"] = "failed"
        item["next_action"] = (
            "Inspect report.investigation_tool_failures for this objective; "
            "repair the read-only tool or source context, then retry in a "
            "new investigation. Do not claim this check was completed."
        )
        unresolved_failed_ids.append(item["objective_id"])

    # Unknown IDs indicate a mismatch between the agent's execution log and
    # the current ledger schema; report them rather than discard silently.
    unrecognized_failed_ids = sorted(reported_failed_ids - ledger_ids)
    counts = dict(Counter(item["status"] for item in objectives))
    unresolved_statuses = {"uninvestigated", "in_progress", "blocked", "requires_client_input"}
    incomplete = (
        any(item["status"] in unresolved_statuses for item in objectives)
        or bool(unrecognized_failed_ids)
    )
    return {
        "schema_version": _OBJECTIVE_VERSION,
        "completion": "incomplete" if incomplete else "evidence_checks_resolved_only",
        "scientific_validation_established": False,
        "objectives": objectives,
        "counts_by_status": counts,
        "failed_tool_objective_ids": sorted(unresolved_failed_ids),
        "unrecognized_failed_objective_ids": unrecognized_failed_ids,
        "note": (
            "Resolved = the specified observational check completed, not that a "
            "cleaning method is correct or authorized. Skipped inapplicable "
            "baseline checks are not counted as coverage failures."
        ),
    }

def run_initial_audit(
        dataset_path: Path,
        progress_callback: Callable[[str], None] | None = None,
        additional_datasets: list[Path] | None = None,
) -> dict:
    dataset_path = Path(dataset_path)

    ingestion = inspect_headers(dataset_path)

    if progress_callback is not None:
        progress_callback("Running automatic baseline checks...")

    # Audit every supplied dataset, not just the primary file. The legacy
    # `baseline` result is deliberately kept for downstream compatibility.
    dataset_paths = {"dataset_1": dataset_path}
    for position, extra_path in enumerate(additional_datasets or [], start=2):
        dataset_paths[f"dataset_{position}"] = Path(extra_path)

    dataset_baselines = {
        dataset_id: run_baseline_audit(dataset)
        for dataset_id, dataset in dataset_paths.items()
    }
    baseline = dataset_baselines["dataset_1"]
    contexts = set(baseline["verified_contexts"])

    # Check statuses matter: a metadata-only check skipped on a taxonomy
    # matrix is not evidence that a relevant dataset check failed.
    baseline_summary = {}
    for dataset_id, dataset_baseline in dataset_baselines.items():
        check_summary = {}
        for name, check in dataset_baseline["checks"].items():
            summary = {
                "status": check["status"],
                "observation_count": len(check.get("observations", [])),
            }
            if check["status"] == "skipped_unverified_context":
                summary["required_context"] = check.get("required_context")
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
            check_summary[name] = summary

        baseline_summary[dataset_id] = {
            "verified_contexts": dataset_baseline["verified_contexts"],
            "checks": check_summary,
        }

    # Build the initial objective plan before invoking Gemma. The complete
    # ledger is rebuilt afterwards using actual successful tool observations.
    initial_ledger = build_investigation_objective_ledger(
        dataset_paths, dataset_baselines, []
    )
    planned_gaps = [
        {"objective_id": obj["objective_id"], "status": obj["status"],
         "next_action": obj["next_action"]}
        for obj in initial_ledger["objectives"]
        if obj["status"] in {"uninvestigated", "blocked"}
    ]

    # Priorities describe evidence gaps rather than deciding how to clean.
    # They are not a substitute for independently validated observations.
    investigation_priorities = []
    for dataset_id, summary in baseline_summary.items():
        checks = summary["checks"]
        if checks.get("find_invalid_counts", {}).get("downstream_blocked"):
            investigation_priorities.append(
                f"{dataset_id}: examine the distribution and meaning of "
                "invalid count cells; distinguish missing values, negative "
                "values, and nonintegral values; do not infer missing=zero."
            )
        if checks.get("find_duplicate_taxa", {}).get("observation_count", 0):
            investigation_priorities.append(
                f"{dataset_id}: investigate duplicated taxonomy labels and "
                "whether their count profiles agree; do not merge on name alone."
            )
        if checks.get("inspect_taxonomy_structure", {}).get("incomplete_count", 0):
            investigation_priorities.append(
                f"{dataset_id}: examine incomplete taxonomy assignments "
                "and state what external reference would be needed."
            )
        if checks.get("find_unclassified_species", {}).get("unclassified_count", 0):
            investigation_priorities.append(
                f"{dataset_id}: characterize unclassified species; do not "
                "invent species assignments."
            )
        if checks.get("find_duplicate_ids", {}).get("observation_count", 0):
            investigation_priorities.append(
                f"{dataset_id}: inspect repeated sample identifiers and "
                "separate duplicate records from conflicting measurements."
            )
        if checks.get("find_duplicate_conflicts", {}).get("observation_count", 0):
            investigation_priorities.append(
                f"{dataset_id}: investigate conflicting metadata records "
                "without automatically choosing which row is authoritative."
            )

    if len(dataset_paths) > 1:
        investigation_priorities.append(
            "Across datasets: evaluate candidate ID relationships, "
            "unmatched identifiers, and ambiguous mappings; never assume "
            "that identifiers with similar spelling designate one subject."
        )

    if progress_callback is not None:
        progress_callback("Automatic baseline checks completed.")

    prior_decisions = get_decision_context(dataset_path)

    prior_outcomes = get_verified_outcome_context(
        dataset_path,
        OUTCOME_LOG,
    )

    investigation = run_investigation(
        path=dataset_path,
        # Reuse the already-computed deterministic audits. File 12's agent
        # otherwise runs another complete baseline audit per dataset.
        objective_baselines=dataset_baselines,
        # Four tool calls were insufficient for dataset-wide investigation.
        # These are upper bounds, not claims of complete QC coverage.
        max_steps=10 if len(dataset_paths) > 1 else 8,
        verified_contexts=contexts,
        user_request=(
            "The automatic read-only baseline audits have completed for "
            "every supplied dataset. Their outcomes are context, NOT "
            "independently citable tool evidence. Investigate unresolved "
            "questions and relationships rather than rerunning completed "
            "checks. A check skipped for unverified context is not proof of "
            "a defect: it may be inapplicable to that dataset. "
            "Use observed evidence only, acknowledge unknown semantics, "
            "and never modify data or execute R. "
            "Prioritized investigation questions: "
            f"{json.dumps(investigation_priorities, ensure_ascii=False)} "
            "Outstanding deterministic investigation objectives (a missing tool "
            "must be reported rather than replaced with arbitrary column scans): "
            f"{json.dumps(planned_gaps[:30], ensure_ascii=False, separators=(',', ':'))} "
            "Baseline findings by dataset: "
            f"{json.dumps(baseline_summary, ensure_ascii=False, separators=(',', ':'))}"
        ),
        progress_callback=progress_callback,
        additional_datasets=additional_datasets,
    )

    history = investigation["history"]
    objective_ledger = build_investigation_objective_ledger(
        dataset_paths, dataset_baselines, history,
        failed_objective_ids=set(investigation.get("failed_objective_ids") or []),
    )
    # The agent's stop reason is not a scientific completion verdict.
    # Persist the ledger with the full audit so unresolved client decisions,
    # unsupported checks and unvisited tools remain visible on every run.
    investigation["report"]["objective_ledger_summary"] = {
        "completion": objective_ledger["completion"],
        "counts_by_status": objective_ledger["counts_by_status"],
        "failed_read_only_objectives": len(objective_ledger["failed_tool_objective_ids"]),
    }
    if objective_ledger["completion"] == "incomplete":
        investigation["report"].setdefault("unresolved_questions", []).append(
            "The objective ledger includes unresolved investigations or "
            "scientific decisions. Inspect investigation_objectives in the audit JSON."
        )
    if objective_ledger["failed_tool_objective_ids"]:
        investigation["report"].setdefault("unresolved_questions", []).append(
            "Some read-only investigation tools failed. Their objectives "
            "remain blocked; inspect report.investigation_tool_failures "
            "before retrying."
        )
    if objective_ledger["unrecognized_failed_objective_ids"]:
        investigation["report"].setdefault("unresolved_questions", []).append(
            "Some failed tool objective IDs could not be reconciled with "
            "the current ledger schema. Review the audit before relying "
            "on completeness counts."
        )
    if progress_callback is not None:
        counts = objective_ledger["counts_by_status"]
        progress_callback(
            "Investigation objectives: "
            + ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
            + f"; overall={objective_ledger['completion']}. "
            "Review investigation_objectives in the saved audit JSON."
        )

    proposal_generation_error = None

    try:
        generated = (
            generate_remediation_proposals(
                history,
                decision_memory=prior_decisions[-3:],
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
        "dataset_baselines": dataset_baselines,
        "investigation_priorities": investigation_priorities,
        "investigation_objectives": objective_ledger,
        "proposal_generation_error": proposal_generation_error,
        "report": investigation["report"],
        "history": history,
        "proposals": proposals,
        "memory": {
            "decisions": prior_decisions,
            "verified_outcomes": prior_outcomes,
        },

    }
