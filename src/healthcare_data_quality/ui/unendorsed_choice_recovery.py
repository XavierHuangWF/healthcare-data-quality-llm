"""Explain declined Gemma appraisals without pretending they authorize edits.

One shared pathway handles every finding and check type. Read-only candidates
are verified against the immutable source before they are offered for renewed
model review. No button in this module executes a data mutation.
"""
from __future__ import annotations

from pathlib import Path


def review_explanation(reviews: list[dict] | None) -> str:
    """Explain why a model-assessed proposal has no executable choice."""
    if not reviews:
        return ('No current Gemma assessment matches these exact options. '
                'The available options require fresh verification.')
    decisions = []
    for review in reviews:
        if not isinstance(review, dict) or review.get('can_offer') is True:
            continue
        reason = ' '.join(str(review.get('reason') or '').split())[:240]
        status = str(review.get('decision') or 'unverified')
        decisions.append(f'{status}: {reason}' if reason else status)
    if not decisions:
        return 'No currently executable option matches Gemma’s assessment and source state.'
    return ('Gemma did not endorse the current action: ' + '; '.join(dict.fromkeys(decisions))[:520]
            + '. No data have been changed by this review.')


def recovery_candidates(finding: dict, source: Path, suggestion: dict) -> list[dict]:
    """Return only immutable-source-backed, displayable candidate plans."""
    from healthcare_data_quality.services.recommendation_options import source_backed_choices
    from healthcare_data_quality.ui.human_choice_replay import prioritize_verified_human_choices
    from healthcare_data_quality.ui.finding_choice_integrity import scoped_review_choices
    try:
        candidates = source_backed_choices(finding, Path(source), suggestion)
        candidates = prioritize_verified_human_choices(finding, Path(source), candidates)
        candidates = scoped_review_choices(finding, Path(source), candidates)
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return []
    # These are UNAPPROVED previews, so do not produce any approve/execute UI.
    return [option for option in candidates if option.get('rows') and option.get('operations')][:3]


def reassessment_request(option: dict) -> str:
    """Preserve exact reviewer-selected scope for a fresh Agent C assessment."""
    import json
    selected = {
        'label': str(option.get('label') or '')[:160],
        'operations': option.get('operations') or [],
        'before_after': option.get('rows') or [],
    }
    return (
        'Independently reassess ONLY the following source-verified candidate. '
        'The reviewer is requesting review, not authorization to edit. '
        'Explain any scientific uncertainty, check already approved operations '
        'and overlapping deleted source coordinates, and either prepare an '
        'exact Before→After selectable solution or explain why it must remain '
        'unapproved. Never execute an operation automatically. Candidate: '
        + json.dumps(selected, ensure_ascii=False, default=str)[:11000]
    )
