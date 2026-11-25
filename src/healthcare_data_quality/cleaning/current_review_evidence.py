"""Read-only projection of source coordinates onto the active approval ledger.

The staged CSV deliberately retains its original physical dimensions. Structural
operations are projected from the named approval ledgers, not inferred from text.
Never equate a value change (or an annotation) with a resolved QC predicate.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .in_memory_cleaning import _records


def approval_finding_label(event: dict, catalog: list[dict], dataset_id: str = '') -> str:
    """Resolve a ledger ID to the same human-facing number used in review cards."""
    from healthcare_data_quality.ui.cleaning_review import indexed_review_catalog
    fid = str(event.get('finding_id') or '')
    for label, item in indexed_review_catalog(catalog or []):
        if item.get('id') == fid:
            title = str(item.get('title') or '').strip()
            return f'{label} · {title}' if title else label
    match = re.fullmatch(r'(dataset_\d+):qc:(\d+)', fid)
    label = f'Finding {match[2]}' if match and match[1] == dataset_id else fid
    title = str(event.get('finding') or '').strip()
    return f'{label} · {title}' if label and title else label or 'Origin not recorded'


def approval_action_label(event: dict, operation: str, row: int, column: int) -> str:
    """Describe only the recorded action, never a model-inferred correction."""
    changes = event.get('changes')
    changes = changes if isinstance(changes, list) else []
    if operation == 'drop_rows':
        rows = sorted({ch['row'] for ch in changes if type(ch.get('row')) is int})
        if not rows:
            return f'Delete row {row}'
        return ('Delete row ' if len(rows) == 1 else 'Delete rows ') + ', '.join(map(str, rows[:9])) + (f' … ({len(rows)} rows)' if len(rows) > 9 else '')
    if operation == 'drop_column':
        return f'Delete column {event.get("position", column)}' + (f' · {event["before"]}' if event.get('before') else '')
    if operation == 'merge_rows':
        for ch in changes:
            if row in (ch.get('row'), ch.get('source_row')):
                return f'Merge row {ch["source_row"]} into row {ch["row"]}'
        return 'Merge rows'
    if operation == 'replace_cells':
        return f'Replace R{row}:C{column}'
    if operation == 'rename_headers':
        return f'Rename column {column}'
    return operation.replace('_', ' ').capitalize() if operation else 'Action not recorded'


def current_finding_evidence(state, finding: dict, source: Path, *, display: bool = False, catalog=None) -> list[dict]:
    from .finding_cleaning import allowed_positions
    from .direct_actions import _standard_taxonomy_value, _majority_style
    ds = finding.get('dataset_id')
    raw = Path(source).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    original = _records(raw)
    staged = _records(state.get('cleaning_staged_bytes_by_dataset', {}).get(ds, raw))
    if len(staged) != len(original) or any(len(a) != len(b) for a,b in zip(staged,original)):
        raise ValueError('Current coordinates cannot be verified')
    def valid(event):
        return event.get('source_sha256') == digest and bool(str(event.get('reviewer') or '').strip())
    removed_rows, removed_cols, merged, edited = {}, {}, {}, {}
    for event in state.get('cleaning_approved_rows_by_dataset', {}).get(ds, []):
        if not valid(event):
            continue
        for ch in event.get('changes', []):
            if event.get('operation') == 'drop_rows':
                removed_rows[ch['row']] = event
            elif event.get('operation') == 'merge_rows':
                removed_rows[ch['source_row']] = event
                merged[ch['row']] = event
    for event in state.get('cleaning_approved_columns_by_dataset', {}).get(ds, []):
        if valid(event):
            removed_cols[event['position']] = event
    for event in state.get('cleaning_history', []):
        if event.get('dataset_id') != ds or not valid(event):
            continue
        for ch in event.get('approved_cell_changes', []):
            edited[(ch.get('row'),ch.get('column'))] = (event,ch)
    cells, headers = allowed_positions(finding, source)
    targets = set(cells) | {(1,c) for c in headers}
    # Row-group evidence and non-editable checks also need a current view.
    for item in finding.get('evidence') or []:
        if not isinstance(item, dict):
            continue
        r = item.get('CSV row', item.get('csv_row'))
        group = item.get('CSV rows', item.get('csv_rows'))
        if type(r) is int and not any(rr == r for rr,_ in targets):
            targets.add((r,1))
        if isinstance(group,str) and re.fullmatch(r'\s*\d+(?:\s*,\s*\d+)*\s*',group):
            targets.update((int(r),1) for r in group.split(','))
    key = finding.get('check_key')
    dominant = (_majority_style([row for i,row in enumerate(staged,1) if i not in removed_rows])
                if key == 'inspect_taxonomy_naming' and 1 not in removed_cols else None)
    out = []
    if catalog is None:
        catalog = state.get('cleaning_review_catalog_snapshot', [])
    finding_labels = {}
    row_deletion_labels = {}
    for r,c in sorted(targets):
        if not 1 <= r <= len(original) or not 1 <= c <= len(original[0]):
            continue
        before, after = original[r-1][c-1], staged[r-1][c-1]
        event = removed_rows.get(r) or removed_cols.get(c)
        if event:
            status, after = 'Removal approved', '(removed on export)'
        elif (r,c) in edited and edited[(r,c)][1].get('after') == after:
            event = edited[(r,c)][0]
            status = 'Changed' if before != after else 'Unchanged'
        elif before != after:
            status = 'Changed — attribution unverified'
        else:
            status = 'Unchanged'
        if not (r in removed_rows or c in removed_cols):
            if r in merged and c > 1:
                # Row merges are applied only at export; do not claim the
                # pre-merge staged count is the final count.
                event = merged[r]
                status = 'Merge approved — final count pending export'
            elif key in {'find_invalid_counts','find_negative_counts'} and c > 1:
                from decimal import Decimal, InvalidOperation
                v = after.strip()
                if not v or v.upper() == 'NA':
                    status = 'Missing (NA)' if v else 'Missing (empty)'
                else:
                    try:
                        n = Decimal(v)
                        if not n.is_finite(): raise InvalidOperation
                        status = 'Negative' if n < 0 else 'Fractional' if n != n.to_integral_value() else 'Nonnegative integer'
                    except InvalidOperation:
                        status = 'Nonnumeric'
            elif key == 'inspect_taxonomy_structure' and c == 1:
                item = next((e for e in finding.get('evidence',[]) if e.get('CSV row') == r), {})
                if item.get('Problem') == 'Uses | instead of ;':
                    status = 'Separator still differs' if '|' in after else 'Separator corrected'
                else:
                    levels = len(re.split(r'[;|]',after))
                    status = f'{levels}/7 levels'
            elif key == 'inspect_taxonomy_naming' and c == 1:
                expected = _standard_taxonomy_value(after, dominant=dominant) if dominant else None
                if expected is not None:
                    # Separator and whitespace are separate from naming;
                    # compare rank tokens, not literal whitespace.
                    parts = lambda value: [v.strip() for v in re.split(r'[;|]',value)]
                    status = 'Naming matches convention' if parts(after) == parts(expected) else 'Naming still differs'
                else:
                    status = 'Naming requires review'
        approved_by = str(event.get('finding_id') or '') if event else ''
        operation = str(event.get('operation') or ('drop_column' if c in removed_cols else '')) if event else ''
        if display and event:
            if approved_by not in finding_labels:
                finding_labels[approved_by] = approval_finding_label(event, catalog, ds)
            approved_by = finding_labels[approved_by]
            if operation == 'drop_rows':
                if id(event) not in row_deletion_labels:
                    row_deletion_labels[id(event)] = approval_action_label(event, operation, r, c)
                operation = row_deletion_labels[id(event)]
            else:
                operation = approval_action_label(event, operation, r, c)
        out.append({'Location': f'R{r}:C{c}', 'Original': before, 'Current': after,
                    'Status': status, 'Approved by': approved_by, 'Action': operation})
    return out


def current_evidence_question(question: str) -> bool:
    """Recognize inspection questions, not requests to edit or approve."""
    text = ' '.join(question.casefold().split())
    return bool(re.search(r'\b(?:what|which|show|how|were|was|did|has|have)\b',text)
                and re.search(r'\b(?:current|now|previous|earlier|actions?|approved|fixed|changed|resolved|revised|taken)\b',text)
                and not re.match(r'^(?:please )?(?:fix|delete|replace|change|remove|apply|approve)\b',text))


def project_current_choices(state, finding: dict, source: Path, choices: list[dict]) -> list[dict]:
    """Remove already satisfied or removed targets; never override a prior edit.

    A narrowed action gets a new fingerprint and must receive a new Gemma
    appraisal. The underlying original-coordinate authorization is unchanged.
    A row merge is indivisible and may not be silently rewritten.
    """
    from .proposal_conflicts import _row_deletions
    from healthcare_data_quality.services.recommendation_options import _display
    from healthcare_data_quality.ui.clear_review_actions import final_action_label
    from healthcare_data_quality.cleaning.compound_approval import validate_compound_targets
    raw = Path(source).read_bytes()
    original = _records(raw)
    ds = finding['dataset_id']
    staged = _records(state.get('cleaning_staged_bytes_by_dataset', {}).get(ds,raw))
    if len(staged) != len(original) or any(len(a)!=len(b) for a,b in zip(staged,original)):
        return []
    rows = _row_deletions(state.get('cleaning_approved_rows_by_dataset',{}).get(ds,[]))
    cols = {a['position'] for a in state.get('cleaning_approved_columns_by_dataset',{}).get(ds,[])}
    out = []
    for choice in choices:
        if any(op.get('operation') == 'flag_cells' for op in choice.get('operations', [])):
            from healthcare_data_quality.cleaning.visual_recommendations import visual_targets_current
            if visual_targets_current(state, finding, source, choice):
                out.append(choice)
            continue
        if choice.get('decision') == 'no_change':
            # A keep-controls option is misleading if one of its columns has
            # already been removed. Let the user close via the review footer.
            if finding.get('check_key') == 'gemma_check' and cols:
                from healthcare_data_quality.services.recommendation_options import _named_blank_positions
                if cols.intersection(_named_blank_positions(finding,original)):
                    continue
            out.append(choice); continue
        plans = []
        for op in choice.get('operations',[]):
            kind = op['operation']; retained = []
            for change in op['changes']:
                if kind in {'replace_cells','rename_headers'}:
                    r = change['row'] if kind=='replace_cells' else 1
                    c = change['column'] if kind=='replace_cells' else change['position']
                    if r in rows or c in cols or staged[r-1][c-1] == change['after']:
                        continue
                elif kind == 'drop_rows' and change['row'] in rows:
                    continue
                elif kind == 'drop_column' and change['position'] in cols:
                    continue
                retained.append(change)
            if retained:
                plans.append({**op,'changes':retained})
        if not plans:
            continue
        try:
            validate_compound_targets(plans)
        except (ValueError,KeyError,TypeError):
            continue
        if plans != choice.get('operations'):
            choice = {**choice,'label':final_action_label(plans,finding),'operations':plans,
                      'rows':_display(plans)}
            choice.pop('gemma_review_status',None)
            choice.pop('gemma_review_reason',None)
        out.append(choice)
    return out
