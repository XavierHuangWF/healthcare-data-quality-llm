"""Current review context from the same structural replay as final export.

Coordinates for authorization remain original; the projected CSV is advisory
and never written, uploaded or used as an alternate authorization source.
"""
from collections import Counter
from pathlib import Path
from .in_memory_cleaning import _records, apply_approved_row_operations, apply_approved_column_deletions
from .proposal_conflicts import _row_deletions


def projected_records(state, dataset_id, source):
    raw = Path(source).read_bytes()
    staged = state.get('cleaning_staged_bytes_by_dataset', {}).get(dataset_id, raw)
    rows = state.get('cleaning_approved_rows_by_dataset', {}).get(dataset_id, [])
    columns = state.get('cleaning_approved_columns_by_dataset', {}).get(dataset_id, [])
    after_rows = apply_approved_row_operations(raw, staged, rows)
    final = apply_approved_column_deletions(raw, after_rows, columns,
                approved_row_operations=rows, staged_before_rows=staged)
    records = _records(final)
    original = _records(raw)
    removed_rows = _row_deletions(rows)
    removed_cols = {c['position'] for c in columns}
    row_ids = [r for r in range(1, len(original)+1) if r not in removed_rows]
    col_ids = [c for c in range(1, len(original[0])+1) if c not in removed_cols]
    if len(row_ids) != len(records) or any(len(row) != len(col_ids) for row in records):
        raise PermissionError('Current projection does not match approved coordinate mapping')
    return records, row_ids, col_ids


def projected_review_context(state, finding, source):
    from .current_review_evidence import current_finding_evidence
    from .direct_actions import _majority_style
    import re
    ds = finding['dataset_id']
    rows, row_ids, col_ids = projected_records(state, ds, source)
    original_evidence = current_finding_evidence(state, finding, source)
    row_index = {r:i for i,r in enumerate(row_ids)}
    col_index = {c:i for i,c in enumerate(col_ids)}
    # Include actual merge sums in current evidence, without making a new edit.
    for item in original_evidence:
        match = re.fullmatch(r'R(\d+):C(\d+)', item['Location'])
        if match:
            r,c = map(int, match.groups())
            if r in row_index and c in col_index:
                item['Current'] = rows[row_index[r]][col_index[c]]
    statuses = Counter(item['Status'] for item in original_evidence)
    # Sample survivors first, then already-removed targets. Full evidence is
    # still checked independently by the host, not inferred from this sample.
    ordered = sorted(original_evidence, key=lambda e:e['Status']=='Removal approved')
    taxonomy_style = _majority_style(rows) if col_ids and col_ids[0] == 1 else None
    return {'records': len(rows)-1, 'columns': len(col_ids),
            'removed_original_rows': len(_records(Path(source).read_bytes()))-len(rows),
            'current_taxonomy_majority': taxonomy_style,
            'evidence_status_counts': dict(statuses),
            'evidence_count': len(original_evidence),
            'current_evidence_sample': ordered[:8],
            'original_coordinates_preserved': True}


def no_edit_review_option(state, finding, source):
    """Final no-new-edit choice when current source evidence is available.

    Does NOT assert correction and is assessed by Gemma like other options.
    If there is no attributable source evidence, leave the gate locked.
    """
    from .current_review_evidence import current_finding_evidence
    evidence = current_finding_evidence(state, finding, source)
    if not evidence:
        details = finding.get('gemma_check_details') or {}
        if (finding.get('check_key') != 'gemma_check' or
                (details.get('verification') or {}).get('status') != 'source_cells_verified_and_output_reproduced'):
            return None
        rows, _, _ = projected_records(state, finding['dataset_id'], source)
        evidence = [{'Location': 'Current dataset', 'Current': f'{len(rows)-1} records; {len(rows[0])} columns'}]
    return {'label':'Keep current data — no additional edits', 'decision':'no_change', 'operations':[],
            'rows':[{'Location':e['Location'], 'Before':e['Current'], 'After':e['Current']} for e in evidence],
            'impact':'No additional edit. This does not certify that remaining anomalies are corrected.'}
