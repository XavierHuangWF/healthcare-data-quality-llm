"""Conservative, deterministic boundary between observations and LLM prose.

A valid citation is provenance, not semantic entailment. Only an *exact*
rendered fact from a registered tool is independently claim-verified here.
All other model-written interpretations stay explicitly review-required.
"""


def canonical_observation_fact(tool: str, observation: dict) -> str | None:
    """Produce a narrow factual statement using measured fields only."""
    if not isinstance(observation, dict):
        return None
    if tool == "compare_columns":
        keys = ("left_position", "right_position", "matching_nonmissing",
                "both_missing", "one_missing", "conflicting_nonmissing")
        if all(type(observation.get(k)) is int for k in keys):
            a, b = observation["left_position"], observation["right_position"]
            return (
                f"Columns {a} and {b}: "
                f"{observation['matching_nonmissing']} matching nonempty rows, "
                f"{observation['both_missing']} rows with both cells blank, "
                f"{observation['one_missing']} rows with one blank cell, and "
                f"{observation['conflicting_nonmissing']} conflicting nonempty rows."
            )
    if tool == "observe_matrix_structure":
        if (type(observation.get("rows")) is int
                and type(observation.get("columns")) is int):
            return (f"The dataset contains {observation['rows']} data rows "
                    f"and {observation['columns']} columns.")
    if tool in {"inspect_column", "inspect_columns"}:
        if (type(observation.get("position")) is int
                and type(observation.get("missing")) is int
                and type(observation.get("nonmissing")) is int):
            return (f"Column {observation['position']} contains "
                    f"{observation['nonmissing']} filled values and "
                    f"{observation['missing']} missing values.")
    if tool == "summarize_invalid_counts":
        if (isinstance(observation.get("issue_type"), str)
                and type(observation.get("count")) is int):
            return (f"The count check detected {observation['count']} "
                    f"cells of type {observation['issue_type']}.")
    return None


def verify_report_claims(report: dict, evidence_index: dict) -> dict:
    """Annotate claims without granting semantic verification by citation alone.

    Mutates the report. Raises for unknown evidence refs. A verified_exact_fact
    has identical text to a deterministic tool rendering. Other claims retain
    their original wording and are labeled as requiring scientific review.
    """
    for finding in report.get("findings") or []:
        evidence_ids = finding.get("evidence_ids") or []
        if not evidence_ids:
            raise ValueError("Narrative findings require at least one evidence reference")
        facts = []
        for evidence_id in evidence_ids:
            if evidence_id not in evidence_index:
                raise ValueError(f"Unknown evidence reference: {evidence_id}")
            entry = evidence_index[evidence_id]
            fact = canonical_observation_fact(entry["tool"], entry["observation"])
            if fact and fact not in facts:
                facts.append(fact)
        claim = finding.get("claim")
        exact = isinstance(claim, str) and claim in facts
        finding["claim_verification"] = {
            "status": "verified_exact_fact" if exact else "requires_scientific_review",
            "reason": (
                "Text exactly matches a fact rendered from a recorded tool observation."
                if exact else
                "A citation resolves, but the model's interpretation has not been "
                "independently established by a deterministic fact check."
            ),
            "verified_observation_facts": facts,
            "citation_ids_valid": True,
            "deterministic_fact_verified": bool(exact),
            "scientific_interpretation_verified": False,
        }
    report["claim_verification_policy"] = (
        "Only exact deterministic observation facts are automatically verified; "
        "model-authored interpretations require scientific review. "
        "No data cleaning is authorized by these labels."
    )
    return report
