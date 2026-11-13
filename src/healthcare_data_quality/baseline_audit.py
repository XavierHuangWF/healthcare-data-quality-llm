from pathlib import Path

from .tools.context import verify_contexts
from .tools.registry import TOOL_REGISTRY, execute_tool
from collections import Counter

from .count_validation import find_invalid_counts
from .tools.context import CANDIDATE_TAXONOMY_CONTEXT

from .taxonomy_validation import find_duplicate_taxa
from .taxonomy_validation import inspect_taxonomy_structure
from .taxonomy_validation import find_unclassified_species
from .taxonomy_validation import inspect_taxonomy_naming


BASELINE_TOOLS = (
    "observe_matrix_structure",
    "find_duplicate_ids",
    "find_duplicate_conflicts",
)


def run_baseline_audit(dataset_path: Path) -> dict:
    dataset_path = Path(dataset_path)
    contexts = verify_contexts(dataset_path)
    checks = {}

    for name in BASELINE_TOOLS:
        required_context = TOOL_REGISTRY[name].requires_context

        if required_context not in contexts:
            checks[name] = {
                "status": "skipped_unverified_context",
                "required_context": required_context,
            }
            continue

        observations = execute_tool(
            name=name,
            path=dataset_path,
            verified_contexts=contexts,
        )

        checks[name] = {
            "status": "completed",
            "observations": observations,
        }



    if CANDIDATE_TAXONOMY_CONTEXT in contexts:
        issues = find_invalid_counts(dataset_path)

        counts_by_type = Counter(
            issue["issue_type"] for issue in issues
        )

        checks["find_invalid_counts"] = {
            "status": "completed",
            "observations": issues,
            "counts_by_type": dict(counts_by_type),
            "downstream_blocked": bool(issues),
        }
    else:
        checks["find_invalid_counts"] = {
            "status": "skipped_unverified_context",
            "required_context": CANDIDATE_TAXONOMY_CONTEXT,
        }

    if CANDIDATE_TAXONOMY_CONTEXT in contexts:
        duplicates = find_duplicate_taxa(dataset_path)

        checks["find_duplicate_taxa"] = {
            "status": "completed",
            "observations": duplicates,
        }
    else:
        checks["find_duplicate_taxa"] = {
            "status": "skipped_unverified_context",
            "required_context": CANDIDATE_TAXONOMY_CONTEXT,
        }

    if CANDIDATE_TAXONOMY_CONTEXT in contexts:
        structure = inspect_taxonomy_structure(dataset_path)

        checks["inspect_taxonomy_structure"] = {
            "status": "completed",
            "observations": [structure],
            "incomplete_count": len(
                structure["incomplete_taxonomy"]
            ),
            "alternate_delimiter_count": len(
                structure["alternate_delimiters"]
            ),
        }
    else:
        checks["inspect_taxonomy_structure"] = {
            "status": "skipped_unverified_context",
            "required_context": CANDIDATE_TAXONOMY_CONTEXT,
        }

    if CANDIDATE_TAXONOMY_CONTEXT in contexts:
        unclassified = find_unclassified_species(dataset_path)

        checks["find_unclassified_species"] = {
            "status": "completed",
            "observations": unclassified,
            "unclassified_count": len(unclassified),
        }
    else:
        checks["find_unclassified_species"] = {
            "status": "skipped_unverified_context",
            "required_context": CANDIDATE_TAXONOMY_CONTEXT,
        }

    if CANDIDATE_TAXONOMY_CONTEXT in contexts:
        naming = inspect_taxonomy_naming(dataset_path)

        checks["inspect_taxonomy_naming"] = {
            "status": "completed",
            "observations": [naming],
            "counts_by_style": {
                style: len(records)
                for style, records in naming.items()
            },
        }
    else:
        checks["inspect_taxonomy_naming"] = {
            "status": "skipped_unverified_context",
            "required_context": CANDIDATE_TAXONOMY_CONTEXT,
        }

    return {
        "verified_contexts": sorted(contexts),
        "checks": checks,
    }


if __name__ == "__main__":
    import sys

    result = run_baseline_audit(Path(sys.argv[1]))

    for name, check in result["checks"].items():
        count = len(check.get("observations", []))
        print(f"{name}: {check['status']} ({count} observations)")