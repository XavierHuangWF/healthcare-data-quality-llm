"""Persist advisory reviewer preference without blocking a UI button click.

This is *not* used for approvals, audit trails, or CSV mutations. Those remain
synchronous and transactionally verified. A declined suggestion is recorded on
one bounded background writer to avoid a synchronous fsync on the UI thread.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

_writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix='hdq-review-feedback')
_log = logging.getLogger(__name__)


def record_decline_without_blocking(finding: dict) -> None:
    from healthcare_data_quality.learning.adaptive_skill_memory import record_review
    snapshot = {'title': str(finding.get('title') or ''),
                'check_key': str(finding.get('check_key') or '')}

    def write() -> None:
        try:
            record_review(snapshot, 'declined_proposal', build_skill=False)
        except (OSError, ValueError, RuntimeError, TypeError):
            _log.exception('Could not persist advisory declined-proposal preference')

    _writer.submit(write)
