"""Presentation order and conservative invalidation of downstream proposals.

Inference remains parallel; publication is ordered. Never mutate named approvals.
"""
from hashlib import sha256
import json


def approval_snapshot(state):
    return {
        'history': list(state.get('cleaning_history', [])),
        'rows': state.get('cleaning_approved_rows_by_dataset', {}),
        'columns': state.get('cleaning_approved_columns_by_dataset', {}),
        'staged': {k: sha256(v).hexdigest() for k, v in state.get('cleaning_staged_bytes_by_dataset', {}).items()},
    }


def snapshot_digest(snapshot):
    return sha256(json.dumps(snapshot, sort_keys=True, default=str).encode()).hexdigest()


def _impact(event, finding):
    """Conservative dependency test. Unknown/aggregate checks depend on all data.
    Structural row removal changes population denominators and comparisons.
    Known taxonomy predicates depend on column 1, not count-only value edits.
    """
    if event.get('dataset_id') and finding.get('dataset_id') and event['dataset_id'] != finding['dataset_id']:
        return False
    kind = event.get('operation')
    if kind == 'flag_cells':
        return False
    if kind in {'drop_rows','merge_rows','drop_column'}:
        return True
    key = finding.get('check_key')
    changes = event.get('approved_cell_changes') or event.get('modified_positions') or []
    columns = {c.get('column', c.get('position')) for c in changes if isinstance(c,dict)}
    columns.discard(None)
    if not columns:
        return True
    if key in {'inspect_taxonomy_structure','inspect_taxonomy_naming'}:
        return kind != 'rename_headers' and 1 in columns
    if key == 'inspect_headers':
        return kind == 'rename_headers'
    if key in {'find_invalid_counts','find_negative_counts'}:
        # All same-column statistics depend on values in every surviving row.
        targets = {x.get('Column', x.get('column', x.get('column_position')))
                   for x in finding.get('evidence',[]) if isinstance(x,dict)}
        targets.discard(None)
        # Missingness evidence may be sampled. Treat it as whole-matrix scope.
        if 'missing' in str(finding.get('title','')).casefold():
            return bool(columns - {1})
        return not targets or bool(columns & targets)
    # Duplicate groups, repeated columns, metadata and exploratory checks may
    # use global comparisons; never infer independence from sampled evidence.
    return True


def revision_suffix(numbered, previous, current):
    """Start at the earliest later finding whose evidence may have changed.

    Invalidation includes the entire suffix, even if another later finding is
    individually independent. No named approval is revoked here.
    """
    if previous is None or snapshot_digest(previous) == snapshot_digest(current):
        return {}
    before = {snapshot_digest(e): e for e in previous['history'] if e.get('operation') != 'flag_cells'}
    after = {snapshot_digest(e): e for e in current['history'] if e.get('operation') != 'flag_cells'}
    changes = [*(before[k] for k in before.keys() - after.keys()),
               *(after[k] for k in after.keys() - before.keys())]
    data_changed = any(previous.get(k) != current.get(k) for k in ('rows', 'columns', 'staged'))
    if not changes and not data_changed:
        return {}
    ids = [f['id'] for _, f in numbered]
    candidates = []
    for event in changes:
        fid = event.get('finding_id')
        if fid not in ids:
            candidates.append((0, 'earlier approval'))
            continue
        origin_index = ids.index(fid)
        for pos in range(origin_index+1,len(numbered)):
            if _impact(event, numbered[pos][1]):
                candidates.append((pos,numbered[origin_index][0]))
                break
    if not changes and data_changed:
        candidates.append((0,'earlier approval'))
    if not candidates:
        return {}
    first, origin = min(candidates,key=lambda x:x[0])
    return {fid: origin for fid in ids[first:]}

def predecessor_blocker(numbered, fid, entries, state, completed, *, publication=None):
    """A later ready result cannot pass an earlier unprepared result.
    Failed results remain locked; only a published actionable or explicitly
    resolved review can release the following cards.
    """
    pending = state.get('cleaning_pending_gemma_b', {})
    revision = state.get('cleaning_order_revision_pending', {})
    for label, finding in numbered:
        other = finding['id']
        if other == fid:
            break
        # The live UI supplies the actual publication states from this render.
        # A draft marked ready, cached suggestion or failed inference is not
        # itself proof of a usable review surface.
        if publication is not None:
            if publication.get(other) not in {"ready", "resolved", "discussion", "completed"}:
                return label
            continue
        if other in revision:
            return label
        if completed(other):
            continue
        if (pending.get(other) or {}).get('kind') == 'exact_option_recovery':
            return label
        if (entries.get(other) or {}).get('status') not in {'ready', 'failed'}:
            if not state.get('cleaning_recommendations', {}).get(other):
                return label
    return None


def publication_epoch(state, fid, published):
    """Advance widget identity only on a real locked -> published transition.

    The new expander mounts expanded once, even when a cached/failed initial
    job recovered asynchronously. Stable identity on subsequent renders keeps
    a reviewer’s deliberate collapse intact.
    """
    epochs = state.setdefault('cleaning_publication_epochs', {})
    old = epochs.get(fid, {'published': False, 'epoch': 0})
    opening = bool(published and not old['published'])
    epoch = old['epoch'] + int(opening)
    epochs[fid] = {'published': bool(published), 'epoch': epoch}
    return epoch, opening


def recovery_due(state, fid, identity, now):
    """Three delayed recovery rounds, no tight loop or user retry buttons.

    The initial attempt is outside this budget. A new source/revision/control
    generation resets the budget. An exhausted review stays locked.
    """
    recovery = state.setdefault('cleaning_empty_review_recovery', {})
    delays = (30, 60, 120)
    record = recovery.get(fid)
    if not record or record['identity'] != identity:
        recovery[fid] = {'identity': identity, 'round': 0, 'due': now + delays[0], 'exhausted': False}
        return False
    if record['exhausted'] or now < record['due']:
        return False
    if record['round'] >= len(delays):
        record['exhausted'] = True
        return False
    record['round'] += 1
    record['due'] = now + delays[min(record['round'], len(delays)-1)]
    return True


def sequential_active_review(numbered, completed):
    """Exactly one unfinished item owns the interactive review lane.

    A ready draft is NOT a terminal decision. Capture this ID once per render;
    a completion mid-render cannot expose a second review tree in that pass.
    """
    return next(((f['id'], label) for label, f in numbered
                 if not completed(f['id'])), (None, None))


def sequential_status_snapshot(state, numbered, completed):
    """Display only current-session work, never speculative disk queue results."""
    active, label = sequential_active_review(numbered, completed)
    entries = {}
    pending = state.get('cleaning_pending_gemma_b', {})
    for name, finding in numbered:
        fid = finding['id']
        status = 'pending'
        if completed(fid):
            status = 'ready'
        elif fid == active:
            job = pending.get(fid) or {}
            published = state.get('cleaning_publication_epochs', {}).get(fid, {}).get('published')
            if job.get('assigned_agent') == 'B':
                status = 'running'
            elif published:
                status = 'ready'
            else:
                status = 'running'
        entries[fid] = {'status': status, 'worker_role': 'agent_b'}
    private = state.get('cleaning_lookahead_records', {})
    future = [(name, private[f['id']]) for name, f in numbered
              if f['id'] != active and f['id'] in private and not completed(f['id'])]
    branch = state.get('cleaning_lookahead_branch') or {}
    branch_label = next((name for name,f in numbered if f['id'] == (branch.get('finding') or {}).get('id')), None) if branch.get('job_id') else None
    return {'context': state.get('cleaning_context_key'), 'entries': entries,
            'lookahead_running': next((name for name, rec in future if rec.get('job_id')), None) or branch_label,
            'lookahead_ready': sum(bool((rec.get('result') or {}).get('ready_count') or
                                        ((rec.get('result') or {}).get('closure') or {}).get('can_offer')) for _,rec in future),
            'sequential': True, 'active_id': active, 'active_label': label,
            'waiting_for_reviewer': bool(active and entries[active]['status'] == 'ready'),
            'completed_reviews': sum(completed(f['id']) for _, f in numbered)}
