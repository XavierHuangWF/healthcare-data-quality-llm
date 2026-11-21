"""Concise, grounded provenance for an Agent B suggestion.

This is a transparency view, not proof that a learned method was right. Past
review decisions and generated skills are advisory and cannot approve edits.
"""
from __future__ import annotations


def memory_trace(suggestion: dict) -> dict[str, object]:
    """Produce brief honest memory provenance without raw source-cell values."""
    if not isinstance(suggestion, dict):
        return {"summary": "Memory: not assessed", "details": []}
    if ("method_candidates_considered" not in suggestion and
            "skill_candidates_considered" not in suggestion):
        return {"summary": "Memory trace unavailable for this previously cached suggestion",
                "details": ["Start a new investigation to record exactly which saved methods and skills Gemma consulted."]}
    methods = max(0, int(suggestion.get("method_candidates_considered") or 0))
    skills = max(0, int(suggestion.get("skill_candidates_considered") or 0))
    if not methods and not skills:
        return {"summary": "Memory: no relevant approved methods or learned skills used",
                "details": []}
    summary = f"Memory assessed: {methods} approved method(s), {skills} learned skill(s)"
    outcome = suggestion.get("memory_assessment") or {}
    decisions = {
        "reused": "Gemma reports reusing a previous method",
        "adapted": "Gemma reports adapting a previous method",
        "rejected": "Gemma reports rejecting earlier methods for this finding",
        "not_applicable": "Gemma did not identify an applicable earlier method",
    }
    details = []
    if isinstance(outcome, dict):
        kind = outcome.get("decision")
        if kind in decisions:
            details.append(decisions[kind])
            reason = " ".join(str(outcome.get("reason") or "").split())[:200]
            if reason:
                details.append(reason)
    for prior in (suggestion.get("memory_context") or [])[:3]:
        if not isinstance(prior, dict):
            continue
        types = ", ".join(str(t) for t in (prior.get("operation_types") or [])[:5])
        if types:
            scope = ("same source" if prior.get("same_original_csv") else "previous source")
            details.append(f"Earlier approved action · {types} ({scope})")
    details.append("Historical methods are advisory; only exact source validation and your approval authorize changes.")
    return {"summary": summary, "details": details}
