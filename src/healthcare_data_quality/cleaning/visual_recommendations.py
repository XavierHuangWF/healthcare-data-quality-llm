"""Source-bound visual alternatives. Styling is never a value transformation."""
from pathlib import Path
import re
from healthcare_data_quality.cleaning.visual_flags import requested_visual_flag, visual_scope_label


def _evidenced_rows(finding):
    result = set()
    for e in finding.get('evidence') or []:
        if not isinstance(e, dict):
            continue
        row = e.get('CSV row', e.get('csv_row'))
        if type(row) is int:
            result.add(row)
        group = e.get('CSV rows', e.get('csv_rows'))
        if isinstance(group, str):
            result.update(int(x) for x in re.findall(r'(?<!\d)\d+(?!\d)', group))
        elif isinstance(group, (list, tuple)):
            result.update(x for x in group if type(x) is int)
    return result


def visual_option(finding, source, request, *, require_complete=False):
    """Construct a complete, independently verified annotation candidate."""
    if source is None:
        return None
    try:
        plan = requested_visual_flag(request, finding, Path(source))
    except (OSError, ValueError, PermissionError, TypeError, KeyError):
        return None
    if not plan or not plan['flags']:
        return None
    # Recommendation scope must be in this finding (explicit user chat uses
    # its separate request/approval gate). Never offer an arbitrary row.
    evidence = finding.get('evidence') or []
    rows = _evidenced_rows(finding)
    if not {f['row'] for f in plan['flags']} <= rows:
        return None
    if require_complete and len(evidence) != finding.get('total_evidence_records', len(evidence)):
        return None  # a sampled observation is not the full annotation scope
    flags = plan['flags']
    style = 'bold red and yellow fill' if flags[0].get('fill', True) else 'bold red font only'
    return {'label': 'Highlight ' + visual_scope_label(plan), 'visual_request': request,
            'operations': [{'operation':'flag_cells','dataset_id':finding.get('dataset_id',''),
                            'changes':flags,'visual_scope':plan['scope'],'style':style}],
            'rows':[{'Location':f"R{f['row']} · C{f['column']}", 'Before':f['before'],
                     'After':f['before'],'Style':style} for f in flags],
            'impact':'Annotation only; CSV values unchanged.'}


def verify_visual_option(finding, source, option):
    """Rebuild from the same request; fail closed if any exact flag differs."""
    ops = option.get('operations') or []
    if not isinstance(ops, list) or len(ops) != 1 or not isinstance(ops[0], dict) or ops[0].get('operation') != 'flag_cells':
        return None
    request = option.get('visual_request') or ''
    rebuilt = visual_option(finding, source, request)
    if not rebuilt:
        return None
    # Old memory may include source_header metadata, but the exact flags and
    # style must match, not just the number or row IDs.
    def normalized(op):
        return {**op, 'changes':[{k:v for k,v in c.items() if k != 'source_header'}
                                 for c in op.get('changes', [])]}
    try:
        if normalized(ops[0]) != rebuilt['operations'][0]:
            return None
    except (TypeError, AttributeError):
        return None
    return {**option, **rebuilt}


def default_visual_option(finding, source):
    rows = sorted(_evidenced_rows(finding))
    if not rows or len(rows) > 50:
        return None
    return visual_option(finding, source, 'highlight rows ' + ', '.join(map(str, rows)), require_complete=True)


def visual_targets_current(state, finding, source, option):
    from healthcare_data_quality.cleaning.in_memory_cleaning import _records
    from healthcare_data_quality.cleaning.active_visual_flags import split_active_visual_flags
    verified = verify_visual_option(finding, source, option)
    if not verified:
        return False
    ds = finding['dataset_id']; raw = Path(source).read_bytes()
    flags = verified['operations'][0]['changes']
    _, removed = split_active_visual_flags(flags,
        state.get('cleaning_approved_rows_by_dataset', {}).get(ds, []),
        state.get('cleaning_approved_columns_by_dataset', {}).get(ds, []))
    staged = _records(state.get('cleaning_staged_bytes_by_dataset', {}).get(ds, raw))
    try:
        return not removed and all(staged[f['row']-1][f['column']-1] == f['before'] for f in flags)
    except IndexError:
        return False
