"""A narrow, source-verified fallback when a model recommendation fails.

An Agent B exception does not mean a QC finding is invalid. For a duplicate
physical column, byte-identical data can be independently checked in Python.
No model error is suppressed; only this independently validated suggestion is
made available for the human's single-click approval.
"""
from __future__ import annotations

from pathlib import Path


def recover_identical_column_suggestion(finding: dict, source: Path | None,
                                        failed_entry: dict) -> dict | None:
    if (source is None or failed_entry.get('status') != 'failed'
            or finding.get('check_key') != 'compare_repeated_headers'):
        return None
    from healthcare_data_quality.cleaning.direct_actions import draft_identical_column_removal
    from healthcare_data_quality.services.recommendation_options import source_backed_choices
    candidate = draft_identical_column_removal(finding, Path(source))
    if not candidate:
        return None
    options = source_backed_choices(finding, Path(source), {
        'choices': [{
            'label': 'Remove fully identical duplicate column',
            'operations': candidate['operations'],
        }],
    })
    if not options or len(options) != 1:
        return None
    return {
        'status': 'ready',
        'suggestion': 'Source-verified identical physical columns; model recommendation failed.',
        'choices': [{
            'label': options[0]['label'],
            'operations': options[0]['operations'],
        }],
        'memory_assessed': False,
        'memory_assessment': {},
        'memory_context': [],
        'source_pattern_summary': {},
        'origin': 'source_verified_fallback',
        'agent_b_error': str(failed_entry.get('error') or '')[:400],
    }
