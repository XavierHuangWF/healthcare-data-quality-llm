"""Small, non-PHI operational feedback for failed local model suggestions.

Only an enumerated issue category and stage are retained. This does NOT
permit the model to modify application code, security policies, or datasets.
It informs the next suggestion to avoid repeating invalid UI/scoping plans.
"""
from __future__ import annotations

import json
import threading
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

_LOCK = threading.RLock()
_DEFAULT = Path(__file__).resolve().parents[3] / 'memory' / 'agent_failure_patterns.jsonl'
_ISSUE_KEYS = {'find_invalid_counts', 'find_duplicate_taxa', 'inspect_taxonomy_structure',
               'inspect_taxonomy_naming', 'inspect_headers', 'compare_repeated_headers',
               'gemma_check'}
_STAGES = {'recommendation', 'model_option_review', 'agent_c_scope', 'execution_preview'}


def _category(exc: Exception) -> str:
    name = type(exc).__name__.casefold()
    detail = str(exc).casefold()
    if any(x in detail for x in ('column', 'row', 'out of scope', 'extra operation', 'overlapping')):
        return 'scope_conflict'
    if any(x in detail for x in ('source', 'sha256', 'coordinate', 'staged')):
        return 'source_mismatch'
    if any(x in detail for x in ('schema', 'json', 'invalid', 'empty', 'incomplete')):
        return 'invalid_model_output'
    if any(x in name for x in ('urlerror', 'timeouterror', 'connectionerror')):
        return 'model_connection'
    return 'other'


def record_failure(finding: dict, stage: str, exc: Exception, *, path: Path | None = None) -> None:
    key = str(finding.get('check_key') or 'other')
    if stage not in _STAGES:
        return
    data = {'schema': 1, 'issue': key if key in _ISSUE_KEYS else 'other',
            'stage': stage, 'category': _category(exc),
            'timestamp': datetime.now(timezone.utc).isoformat()}
    target = Path(path) if path else _DEFAULT
    with _LOCK:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('a', encoding='utf-8') as f:
            f.write(json.dumps(data) + '\n')


def recent_failure_hints(finding: dict, *, path: Path | None = None) -> list[dict]:
    target = Path(path) if path else _DEFAULT
    if not target.is_file():
        return []
    key = str(finding.get('check_key') or 'other')
    counters = Counter()
    with _LOCK:
        with target.open('r', encoding='utf-8') as stream:
            for raw in deque(stream, maxlen=100):
                try:
                    item = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                if (isinstance(item, dict) and item.get('schema') == 1
                        and item.get('issue') == (key if key in _ISSUE_KEYS else 'other')
                        and item.get('stage') in _STAGES):
                    counters[(item.get('stage'), item.get('category'))] += 1
    return [{'stage': stage, 'failure_type': category, 'recent_count': count}
            for (stage, category), count in counters.most_common(5)]
