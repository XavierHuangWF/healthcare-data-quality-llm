"""Read-only, source-grounded numeric comparators for a reviewer discussion.

A negative count is not made scientifically valid by applying abs().  Report
same-column distributions as context only; never prepare an edit or permission.
Original 1-based physical CSV coordinates remain the immutable reference.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import median, stdev, mean


def _number(text: object) -> float | None:
    try:
        v = float(str(text).strip())
    except (ValueError, TypeError, OverflowError):
        return None
    return v if math.isfinite(v) else None


def _quantile(sorted_numbers: list[float], proportion: float) -> float | None:
    if not sorted_numbers:
        return None
    rank = proportion * (len(sorted_numbers) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    return sorted_numbers[lo] + (sorted_numbers[hi] - sorted_numbers[lo]) * (rank - lo)


def _targets(finding: dict, *, limit: int = 12) -> list[tuple[int, int]]:
    """Use only exact original cells evidenced by the selected finding."""
    targets: list[tuple[int, int]] = []
    for item in finding.get('evidence') or []:
        if not isinstance(item, dict):
            continue
        row = item.get('CSV row', item.get('csv_row', item.get('row')))
        column = item.get('Column', item.get('column', item.get('column_position')))
        if type(row) is int and row >= 2 and type(column) is int and column >= 2:
            pair = row, column
            if pair not in targets:
                targets.append(pair)
            if len(targets) >= limit:
                break
    return targets


def compare_evidenced_numeric_cells(source: Path, finding: dict, *,
                                    max_source_rows: int = 100000,
                                    candidates: dict[tuple[int, int], object] | None = None,
                                    projected=None) -> list[dict]:
    """Compare anomalous numeric values against their own column's other rows.

    All numeric rows are read for the returned columns, excluding each target's
    own row from its comparison. A bounded, explicitly marked scan prevents an
    unbounded memory allocation. Non-numeric / NA / blank observations are
    counted separately; zeros are not conflated with positive measurements.
    """
    targets = list(candidates)[:32] if candidates is not None else _targets(finding)
    if not targets:
        return []
    if not 1 <= max_source_rows <= 100000:
        raise ValueError('Invalid numeric profile row limit')
    with Path(source).open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, None)
        if not header:
            raise ValueError('Missing CSV header')
        if projected is not None:
            records, row_ids, col_ids = projected
            positions = {c:i for i,c in enumerate(col_ids)}
            reader = iter(records[1:])
            indexed = zip(row_ids[1:], reader)
        else:
            positions = {i:i-1 for i in range(1,len(header)+1)}
            indexed = enumerate(reader,start=2)
        cols = sorted({c for _, c in targets if c in positions})
        if not cols:
            return []
        values = {c: [] for c in cols}
        by_position: dict[tuple[int, int], str] = {}
        observed_rows = 0
        truncated = False
        for row, cells in indexed:
            if len(cells) != len(positions):
                raise ValueError(f'Inconsistent CSV width at row {row}')
            if observed_rows >= max_source_rows:
                truncated = True
                break
            observed_rows += 1
            for c in cols:
                values[c].append((row, cells[positions[c]]))
                if (row, c) in targets:
                    by_position[(row, c)] = cells[positions[c]]
    output = []
    for row, c in targets:
        if (row, c) not in by_position:
            continue
        original = by_position[(row, c)]
        target = _number(original)
        proposed = _number(candidates.get((row, c))) if candidates is not None else (abs(target) if target is not None else None)
        if proposed is None or (candidates is None and target is None):
            continue
        others = [(r, text) for r, text in values[c] if r != row]
        numeric = [x for _, text in others if (x := _number(text)) is not None]
        positive = sorted(x for x in numeric if x > 0)
        negative = [x for x in numeric if x < 0]
        zeros = sum(x == 0 for x in numeric)
        transformed = proposed
        comparable = positive
        nonnegative = sorted(x for x in numeric if x >= 0)
        greater_or_equal = sum(x <= transformed for x in comparable)
        output.append({
            'CSV row': row,
            'Column': c,
            'Column name': header[c - 1],
            'Original': original,
            'Absolute-value candidate (not approved)': f'{transformed:g}',
            'Other rows examined': len(others),
            'Other zeros': zeros,
            'Other positive observations': len(positive),
            'Other negative observations': len(negative),
            'Other missing / nonnumeric': len(others) - len(numeric),
            'Positive median': _quantile(positive, 0.5),
            'Positive sample SD': stdev(positive) if len(positive) > 1 else None,
            'Positive mean': mean(positive) if positive else None,
            'Nonnegative median (including zeros)': _quantile(nonnegative, 0.5),
            'Nonnegative sample SD (including zeros)': stdev(nonnegative) if len(nonnegative) > 1 else None,
            'Positive P95': _quantile(positive, 0.95),
            'Positive maximum': positive[-1] if positive else None,
            'Candidate percentile among positive values': (
                round(100 * greater_or_equal / len(comparable), 1) if comparable else None),
            'Candidate exceeds observed positive maximum': (
                transformed > positive[-1] if positive else None),
            'Source scan truncated': truncated,
            'Interpretation': ('Descriptive comparison only. The negative source value remains invalid '
                               'for a nonnegative count; its absolute value is not evidence of '
                               'the true measurement. Confirm provenance or instrument/raw data '
                               'before replacing it.'),
        })
    return output


def display_numeric_comparisons(comparisons: list[dict]) -> list[dict]:
    """Compact table for the UI, preserving full evidence for the Gemma prompt."""
    return [{
        'Location': f"R{r['CSV row']}:C{r['Column']} ({r['Column name']})",
        'Before': r['Original'],
        'Abs. candidate (unapproved)': r['Absolute-value candidate (not approved)'],
        'Other zeros': r['Other zeros'],
        'Other positive values': r['Other positive observations'],
        'Positive median': r['Positive median'],
        'Positive sample SD': r.get('Positive sample SD'),
        'Median incl. zeros': r.get('Nonnegative median (including zeros)'),
        'SD incl. zeros': r.get('Nonnegative sample SD (including zeros)'),
        'Positive P95': r['Positive P95'],
        'Positive max': r['Positive maximum'],
        'Candidate percentile': r['Candidate percentile among positive values'],
    } for r in comparisons]


def proposal_numeric_comparisons(source: Path, plans: list[dict], *, projected=None) -> list[dict]:
    """Actual proposed numbers, compared with original same-column observations."""
    candidates = {}
    for plan in plans:
        if plan.get('operation') != 'replace_cells':
            continue
        for ch in plan.get('changes') or []:
            r, c = ch.get('row'), ch.get('column')
            if type(r) is int and type(c) is int and r >= 2 and c >= 2 and _number(ch.get('after')) is not None:
                candidates[(r, c)] = ch['after']
    if not candidates:
        return []
    records = compare_evidenced_numeric_cells(source, {}, candidates=candidates, projected=projected)
    rows = display_numeric_comparisons(records)
    return [{'Location': row['Location'], 'Before': row['Before'],
             'Proposed value': row['Abs. candidate (unapproved)'],
             **{k: v for k, v in row.items() if k not in {'Location', 'Before', 'Abs. candidate (unapproved)'}}}
            for row in rows]
