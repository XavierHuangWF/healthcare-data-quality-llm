"""Give exact human-approved source-matched choices precedence over model guesses.

This is not a learned scientific rule. It replays previously approved actions
only when the original CSV hash, finding identity, exact affected targets and
original before-values are independently verified again. No edits occur here.
"""
from __future__ import annotations

from collections import deque
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path


def _label(operations: list[dict]) -> str:
    parts: list[str] = []
    for op in operations:
        kind = op['operation']
        changes = op['changes']
        if kind == 'replace_cells':
            parts.extend(f"{c['before']} → {c['after']}" for c in changes[:3])
        elif kind == 'drop_column':
            parts.extend(f"Delete column {c['position']} ({c['before'] or 'unnamed'})" for c in changes[:3])
        elif kind == 'drop_rows':
            parts.append('Delete rows ' + ', '.join(str(c['row']) for c in changes[:6]))
        elif kind == 'merge_rows':
            parts.append('Merge rows ' + ', '.join(f"{c['row']} + {c['source_row']}" for c in changes[:4]))
        elif kind == 'rename_headers':
            parts.extend(f"Rename column {c['position']}" for c in changes[:3])
    return '; '.join(parts)[:90] or 'Reuse approved method'


def _recognized_ratio(operations: list[dict]) -> str | None:
    """Describe a repeated observed mapping, never infer a global rule."""
    factors = []
    for op in operations:
        if op['operation'] != 'replace_cells':
            continue
        for c in op['changes']:
            try:
                before, after = Decimal(c['before']), Decimal(c['after'])
                if not before or not before.is_finite() or not after.is_finite():
                    return None
                factors.append(after / before)
            except (InvalidOperation, ValueError, TypeError, ZeroDivisionError):
                return None
    if len(factors) >= 2 and len(set(factors)) == 1:
        return 'Repeated historical mapping: multiply these specific values by ' + format(factors[0], 'f')
    return None


def verified_approved_choices(finding: dict, source: Path | None, *, limit: int = 2,
                              path: Path | None = None) -> list[dict]:
    """Return distinct, current-source-verifiable completed reviewer decisions.

    Read historical decisions in reverse chronology; revocations invalidate
    older matching decisions. No method from another CSV is automatically
    applied: those remain advisory context for Agent B.
    """
    if source is None or type(limit) is not int or not 1 <= limit <= 3:
        return []
    from healthcare_data_quality.learning.interaction_memory import (
        _identity, _location, _VERSION, _LOCK, review_memory_identity,
    )
    from healthcare_data_quality.services.recommendation_options import _rows, _verified, _display

    try:
        wanted = _identity(finding, Path(source))
        original_rows = _rows(Path(source))
        destination = _location(path)
        if not destination.is_file():
            return []
        with _LOCK:
            with destination.open('r', encoding='utf-8') as stream:
                recent = deque((line for line in stream if len(line) < 1_000_000), maxlen=2500)
    except (OSError, UnicodeError, ValueError):
        return []
    result, seen, revoked = [], set(), set()
    for raw in reversed(recent):
        try:
            record = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(record, dict) or record.get('schema_version') != _VERSION:
            continue
        identity = review_memory_identity(record)
        if record.get('decision') == 'revision_reset':
            revoked.add(identity)
            continue
        if identity in revoked or record.get('decision') != 'approved_edit':
            continue
        if (record.get('source_sha256') != wanted['source_sha256'] or
                record.get('finding_type') != wanted['finding_type']):
            continue
        operations = record.get('operations')
        # _verified supplies exact current source 'Before', rejecting a model's
        # or memory's stale coordinate or mismatched original value.
        verified = _verified(operations, original_rows, finding)
        if not verified:
            continue
        key = json.dumps(verified, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        result.append({
            'label': _label(verified),
            'operations': verified,
            'rows': _display(verified),
            'provenance': 'Previously approved for this exact source and finding',
            'memory_id': record.get('memory_id'),
            'user_request': str(record.get('user_request') or '')[:300],
            'observed_pattern': _recognized_ratio(verified),
        })
        if len(result) >= limit:
            break
    return result


def prioritize_verified_human_choices(finding: dict, source: Path | None,
                                      model_choices: list[dict], *, limit: int = 3) -> list[dict]:
    """Never let rounding override a replayable, contradictory human edit."""
    remembered = verified_approved_choices(finding, source, limit=min(2, limit))
    if not remembered:
        return list(model_choices)[:limit]
    seen = {json.dumps(x['operations'], sort_keys=True) for x in remembered}
    human_targets = {}
    for proposal in remembered:
        for op in proposal['operations']:
            if op['operation'] == 'replace_cells':
                for c in op['changes']:
                    human_targets.setdefault((c['row'], c['column']), set()).add(c['after'])
    combined = list(remembered)
    for choice in model_choices:
        if len(combined) >= limit:
            break
        fingerprint = json.dumps(choice.get('operations'), sort_keys=True)
        if fingerprint in seen:
            continue
        contradiction = any(
            (c['row'], c['column']) in human_targets
            and c['after'] not in human_targets[(c['row'], c['column'])]
            for op in choice.get('operations') or [] if op.get('operation') == 'replace_cells'
            for c in op.get('changes') or []
        )
        if contradiction:
            continue
        seen.add(fingerprint)
        combined.append(choice)
    return combined
