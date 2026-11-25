"""Give exact human-approved source-matched choices precedence over model guesses.

This is not a learned scientific rule. It replays previously approved actions
only when the original CSV hash, finding identity, exact affected targets and
original before-values are independently verified again. No edits occur here.
"""
from __future__ import annotations

from collections import Counter, deque
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


MIN_CHOICES = 2
MAX_CHOICES = 5


def action_key(operations: list[dict]) -> str:
    """Canonical exact action; ignore transient finding IDs and ordering of changes."""
    grouped = {}
    for op in operations or []:
        kind = op.get('operation')
        if not kind:
            continue
        group = (kind, json.dumps(op.get('style'), sort_keys=True), op.get('visual_scope'))
        grouped.setdefault(group, []).extend(json.dumps({k: c[k] for k in
            ('row', 'column', 'position', 'source_row', 'before', 'after', 'style', 'fill') if k in c},
            sort_keys=True, ensure_ascii=False) for c in op.get('changes') or [])
    return json.dumps(sorted((str(k), sorted(v)) for k, v in grouped.items()), ensure_ascii=False)


def _history(path=None):
    from healthcare_data_quality.learning.interaction_memory import _location, _LOCK, _VERSION
    destination = _location(path)
    try:
        with _LOCK, destination.open('r', encoding='utf-8') as stream:
            recent = deque((line for line in stream if len(line) < 1_000_000), maxlen=2500)
    except (OSError, UnicodeError, ValueError):
        return []
    records, seen = [], set()
    for line in recent:
        try:
            record = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(record, dict) or record.get('schema_version') != _VERSION:
            continue
        mid = record.get('memory_id')
        if mid and mid in seen:
            continue
        if mid:
            seen.add(mid)
        records.append(record)
    return records


def approval_counts(finding, source, *, path=None, records=None):
    """Count actual historical approvals, not suggestions or independent users.

    Revocations affect replay eligibility, not the historical event count.
    Only exact source and issue are compared; no cross-CSV coordinate replay.
    """
    from healthcare_data_quality.learning.interaction_memory import _identity
    if source is None:
        return Counter()
    try:
        wanted = _identity(finding, Path(source))
    except (OSError, ValueError):
        return Counter()
    counts = Counter()
    for record in _history(path) if records is None else records:
        if (record.get('decision') == 'approved_edit' and
            record.get('source_sha256') == wanted['source_sha256'] and
            record.get('finding_type') == wanted['finding_type']):
            try:
                counts[action_key(record.get('operations') or [])] += 1
            except (TypeError, AttributeError, ValueError):
                continue
    return counts


def rank_choices(finding, source, choices, *, limit=MAX_CHOICES, path=None):
    """Deduplicate and order eligible candidates by historical approval count."""
    counts = approval_counts(finding, source, path=path)
    output, seen = [], set()
    for choice in choices:
        key = (choice.get('decision') or '', action_key(choice.get('operations') or []))
        if key in seen:
            continue
        seen.add(key)
        item = {**choice}
        item.pop('historical_approval_count', None)
        if counts[key[1]]:
            item['historical_approval_count'] = counts[key[1]]
        output.append(item)
    output.sort(key=lambda c: -c.get('historical_approval_count', 0))
    return output[:limit]


def verified_approved_choices(finding: dict, source: Path | None, *, limit: int = MAX_CHOICES,
                              path: Path | None = None) -> list[dict]:
    """Replay only non-revoked exact-source approvals; rank by past approvals."""
    if source is None or type(limit) is not int or not 1 <= limit <= MAX_CHOICES:
        return []
    from healthcare_data_quality.learning.interaction_memory import _identity, review_memory_identity
    from healthcare_data_quality.services.recommendation_options import _rows, _verified, _display
    try:
        wanted = _identity(finding, Path(source))
        original_rows = _rows(Path(source))
    except (OSError, UnicodeError, ValueError):
        return []
    records = _history(path)
    counts = approval_counts(finding, source, records=records)
    result, seen, revoked = [], set(), set()
    for record in reversed(records):
        identity = review_memory_identity(record)
        if record.get('decision') == 'revision_reset':
            revoked.add(identity)
            continue
        if identity in revoked or record.get('decision') != 'approved_edit':
            continue
        if (record.get('source_sha256') != wanted['source_sha256'] or
                record.get('finding_type') != wanted['finding_type']):
            continue
        ops = record.get('operations') or []
        if isinstance(ops, list) and any(isinstance(op, dict) and op.get('operation') == 'flag_cells' for op in ops):
            from healthcare_data_quality.cleaning.visual_recommendations import verify_visual_option
            visual = verify_visual_option(finding, source, {'operations':ops,
                'visual_request':record.get('user_request', '')})
            if visual:
                key = action_key(visual['operations'])
                if key not in seen:
                    seen.add(key)
                    result.append({**visual, 'provenance':'Previously approved annotation',
                        'memory_id':record.get('memory_id'), 'historical_approval_count':counts[key]})
            continue
        verified = _verified(ops, original_rows, finding)
        if not verified:
            continue
        key = action_key(verified)
        if key in seen:
            continue
        seen.add(key)
        result.append({
            'label': _label(verified), 'operations': verified, 'rows': _display(verified),
            'provenance': 'Previously approved for this exact source and finding',
            'memory_id': record.get('memory_id'),
            'user_request': str(record.get('user_request') or '')[:300],
            'observed_pattern': _recognized_ratio(verified),
            'historical_approval_count': counts[key],
        })
    result.sort(key=lambda c: -c['historical_approval_count'])
    selected = result[:limit]
    # Keep an approved annotation available alongside the highest-frequency
    # value edits. Its frequency is still the real historical count.
    if limit >= 2 and not any(c.get('visual_request') for c in selected):
        visual = next((c for c in result if c.get('visual_request')), None)
        if visual:
            selected = [*selected[:limit-1], visual]
    return selected


def prioritize_verified_human_choices(finding: dict, source: Path | None,
                                      model_choices: list[dict], *, limit: int = MAX_CHOICES) -> list[dict]:
    """Never let rounding override a replayable, contradictory human edit."""
    remembered = verified_approved_choices(finding, source, limit=limit)
    if not remembered:
        return rank_choices(finding, source, model_choices, limit=limit)
    seen = {action_key(x['operations']) for x in remembered}
    human_targets = {}
    for proposal in remembered:
        for op in proposal['operations']:
            if op['operation'] == 'replace_cells':
                for c in op['changes']:
                    human_targets.setdefault((c['row'], c['column']), set()).add(c['after'])
    combined = list(remembered)
    for choice in model_choices:
        fingerprint = action_key(choice.get('operations') or [])
        if fingerprint in seen:
            # Preserve the complete source-derived impact warning on a remembered action.
            for existing in combined:
                if action_key(existing.get('operations') or []) == fingerprint:
                    for field in ('impact', 'memory_note', 'derived_from_source'):
                        if choice.get(field) and not existing.get(field):
                            existing[field] = choice[field]
            continue
        contradiction = any(
            (c['row'], c['column']) in human_targets
            and c['after'] not in human_targets[(c['row'], c['column'])]
            for op in choice.get('operations') or [] if op.get('operation') == 'replace_cells'
            for c in op.get('changes') or []
        )
        # A direct, immutable-source-bound missing-value option is a distinct
        # *reviewer choice*, not an inferred rounding override. Older human
        # decisions remain first, but cannot silently conceal the alternative
        # Gemma has just proposed. Scientific validity is the reviewer's call.
        explicit_na_alternative = (
            choice.get('explicit_model_alternative') is True
            and bool(choice.get('operations'))
            and all(op.get('operation') == 'replace_cells'
                    and all(ch.get('after') == 'NA' for ch in op.get('changes') or [])
                    for op in choice.get('operations') or [])
        )
        if contradiction and not explicit_na_alternative:
            continue
        seen.add(fingerprint)
        combined.append(choice)
    return rank_choices(finding, source, combined, limit=limit)


def memory_first_candidates(finding, source, suggestion=None):
    """Shared candidate pipeline: source-derived and saved approvals before LLM drafts.

    The result is advisory only; it still requires current Gemma assessment,
    cross-finding validation and named approval. Never replay old coordinates
    against a different CSV.
    """
    from healthcare_data_quality.services.recommendation_options import source_backed_choices
    from healthcare_data_quality.ui.finding_choice_integrity import scoped_review_choices
    from healthcare_data_quality.cleaning.visual_recommendations import default_visual_option
    candidates = source_backed_choices(finding, source, suggestion or {})
    # Automatic recommendations must not erase existing count measurements
    # merely because NA is a convenient fallback. Prior approvals can still
    # supply an exact NA replacement; explicit chat requests use Agent C.
    if finding.get('check_key') in {'find_invalid_counts','find_negative_counts'}:
        def erases_measurement(option):
            return any(op.get('operation') == 'replace_cells' and any(
                str(c.get('before','')).strip() and str(c.get('before','')).strip().upper() != 'NA'
                and str(c.get('after','')).strip().upper() == 'NA'
                for c in op.get('changes', [])) for op in option.get('operations', []))
        # Only the eligible exact-memory replay below can restore a prior
        # nonempty-to-NA decision; a revoked record is not a new fallback.
        candidates = [c for c in candidates if not erases_measurement(c)]
    # Offer a non-destructive alternative for complete row-based findings.
    # Retain familiar memory choices; do not crowd out existing valid options.
    remembered = prioritize_verified_human_choices(finding, source, candidates)
    visual = default_visual_option(finding, source)
    if visual and not any(any(op.get('operation') == 'flag_cells' for op in c.get('operations', [])) for c in remembered):
        # For already-empty cells, the standard explicit missing marker is a
        # non-imputing alternative. Still replay any genuinely approved visual
        # method for such findings; do not force a third default option.
        def only_encodes_missing(choice):
            ops = choice.get('operations') or []
            return bool(ops) and all(op.get('operation') == 'replace_cells' and
                all(not str(c.get('before','')).strip() and c.get('after') == 'NA'
                    for c in op.get('changes', [])) for op in ops)
        if len(remembered) < MIN_CHOICES or not any(only_encodes_missing(c) for c in remembered):
            remembered = [*remembered[:MAX_CHOICES-1], visual]
    return scoped_review_choices(finding, source, rank_choices(finding, source, remembered))
