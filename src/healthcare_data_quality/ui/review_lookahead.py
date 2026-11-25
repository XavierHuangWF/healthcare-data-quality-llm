"""Private three-item preparation window; only the active review may publish.

A speculative model assessment is advisory, tied to immutable source, exact
candidate set and captured dependencies. Unknown dependencies fail closed.
Neither prefetch nor promotion authorizes or stages any operation.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

WINDOW_SIZE = 3

_STATE_KEYS = ('cleaning_review_catalog_snapshot', 'cleaning_approved_reviewer',
    'cleaning_history', 'cleaning_staged_bytes_by_dataset',
    'cleaning_approved_rows_by_dataset', 'cleaning_approved_columns_by_dataset',
    'cleaning_visual_flags_by_dataset', 'cleaning_messages_by_finding',
    'cleaning_decisions', 'cleaning_review_statuses')


def window(numbered, completed):
    return [(label, f) for label, f in numbered if not completed(f['id'])][:WINDOW_SIZE]


def snapshot(state):
    return {k: deepcopy(state[k]) for k in _STATE_KEYS if k in state}


def _candidates(state, finding, source, suggestion):
    from healthcare_data_quality.ui.human_choice_replay import memory_first_candidates, rank_choices
    from healthcare_data_quality.cleaning.current_review_evidence import project_current_choices
    from healthcare_data_quality.cleaning.proposal_conflicts import filter_conflicting_choices
    from healthcare_data_quality.cleaning.sequential_context import no_edit_review_option
    choices = memory_first_candidates(finding, source, suggestion)
    choices = project_current_choices(state, finding, source, choices)
    choices, _ = filter_conflicting_choices(choices, state=state, finding=finding, source=source)
    if not choices:
        keep = no_edit_review_option(state, finding, source)
        if keep:
            choices = [keep]
    return rank_choices(finding, source, choices)


def _signatures(finding, source, candidates):
    from healthcare_data_quality.services.model_plan_review import option_fingerprint
    return [(option_fingerprint(finding, source, c), c.get('historical_approval_count', 0),
             c.get('rows'), c.get('impact')) for c in candidates]


def prepare(snapshot_state, finding, source, stop, closure_worker=None):
    """Worker-only planning; no Streamlit state, no writes, no authorization."""
    from healthcare_data_quality.services.llm_client import cancel_speculative_requests
    with cancel_speculative_requests(stop):
        return _prepare(snapshot_state, finding, source, stop, closure_worker)


def _prepare(snapshot_state, finding, source, stop, closure_worker=None):
    from healthcare_data_quality.services.model_plan_review import (
        dataset_review_context, assess_exact_options, draft_recovery_options,
        reviewer_selectable_choices, option_fingerprint)
    from healthcare_data_quality.cleaning.previous_approval_resolution import source_verified_prior_row_resolution
    if stop.is_set():
        return {'cancelled': True}
    source = Path(source)
    context = dataset_review_context(snapshot_state, finding['dataset_id'], finding['id'], source=source)
    resolution = source_verified_prior_row_resolution(finding, source,
        snapshot_state.get('cleaning_approved_rows_by_dataset', {}).get(finding['dataset_id'], []))
    if resolution and closure_worker:
        historical = [{'finding_id':e.get('finding_id'), 'operation':e.get('operation'),
                       'positions':(e.get('modified_positions') or [])[:8]}
                      for e in snapshot_state.get('cleaning_history', [])[-30:]]
        result = closure_worker(finding, source, resolution, '', historical, context)
        return {'closure':result, 'resolution':resolution, 'candidates':[]}
    suggestion = {'suggestion':'', 'choices':[], 'gemma_option_reviews':[],
                  '_source':'exact_option_recovery','_worker_role':'agent_b',
                  '_review_revision':context['revision']}
    reviews = {}
    choices = _candidates(snapshot_state, finding, source, suggestion)

    def assess(items):
        if stop.is_set():
            return
        pending = [option for option in items
                   if option_fingerprint(finding, source, option) not in reviews]
        if not pending:
            return
        expected = {option_fingerprint(finding, source, option) for option in pending}
        for review in assess_exact_options(finding, source, pending, context):
            fp = review.get('fingerprint')
            if fp in expected:
                reviews[fp] = review

    def ready(items):
        return [o for o in reviewer_selectable_choices(finding, source, items, list(reviews.values()))
                if o.get('gemma_review_status') in {'supported','disputed'}]

    assess(choices)
    draft_attempted = False
    if not stop.is_set() and len(ready(choices)) < 2:
        draft_attempted = True
        try:
            draft = draft_recovery_options(finding, source, context)
            if isinstance(draft, dict):
                retained = ready(choices)
                suggestion.update(draft)
                suggestion['choices'] = [*retained, *(draft.get('choices') or [])][:5]
                choices = _candidates(snapshot_state, finding, source, suggestion)
                assess(choices)
        except (OSError, RuntimeError, ValueError, TypeError, KeyError):
            pass  # Retain valid already-assessed options.
    if stop.is_set():
        return {'cancelled':True}
    suggestion['gemma_option_reviews'] = list(reviews.values())
    suggestion['_source'] = 'exact_option_recovery'
    suggestion['_worker_role'] = 'agent_b'
    suggestion['_review_revision'] = context['revision']
    return {'suggestion':suggestion, 'candidates':choices,
            'signatures':_signatures(finding, source, choices), 'ready_count':len(ready(choices)), 'draft_attempted':draft_attempted}


def _user_turns(state):
    # Only exact, standard no-change UI messages are non-semantic. Any actual
    # new instruction, even if it leaves the CSV unchanged, invalidates reuse.
    harmless = {'keep as is', 'keep current data',
                'mark reviewed — accepted unchanged (no correction asserted)'}
    return {fid:[str(t.get('content') or '') for t in turns
                 if t.get('role') == 'user' and not t.get('automatic')
                 and str(t.get('content') or '').strip().casefold() not in harmless]
            for fid, turns in state.get('cleaning_messages_by_finding', {}).items()
            if any(t.get('role') == 'user' and not t.get('automatic')
                   and str(t.get('content') or '').strip().casefold() not in harmless for t in turns)}


def _append_only(before, after):
    return len(after) >= len(before) and after[:len(before)] == before


def is_verified_zero_check(finding):
    return (finding.get('check_key') == 'gemma_check' and not finding.get('evidence')
            and not finding.get('actionable')
            and (finding.get('gemma_check_details', {}).get('verification') or {}).get('status') == 'source_cells_verified_and_output_reproduced')


def dependencies_unchanged(old, current, finding, source, candidates=None, *, allow_pending=False):
    """Conservative, full-data dependency check (never use a sampled table).

    Only known deterministic QC dependencies may survive an unrelated edit.
    Structural changes, unknown exploratory dependencies, novel instructions,
    overlapping annotations, revisions and new catalog evidence force reassessment.
    """
    from healthcare_data_quality.cleaning.sequential_context import projected_records
    from healthcare_data_quality.learning.interaction_memory import _identity
    if old.get('cleaning_approved_reviewer') != current.get('cleaning_approved_reviewer'):
        return False
    before_catalog = old.get('cleaning_review_catalog_snapshot', [])
    after_catalog = current.get('cleaning_review_catalog_snapshot', [])
    if before_catalog != after_catalog:
        # Agent A can append a verified zero-anomaly check while B prepares
        # deterministic QC. It adds no competing anomaly or writable scope.
        # New findings, changed evidence and exploratory targets still fail closed.
        if (finding.get('source') != 'QC' or not _append_only(before_catalog, after_catalog)
                or len({f.get('id') for f in after_catalog}) != len(after_catalog)
                or not all(is_verified_zero_check(f) for f in after_catalog[len(before_catalog):])):
            return False
    old_history, new_history = old.get('cleaning_history', []), current.get('cleaning_history', [])
    old_decisions, new_decisions = old.get('cleaning_decisions', []), current.get('cleaning_decisions', [])
    if not _append_only(old_history, new_history) or not _append_only(old_decisions, new_decisions):
        return False
    ds = finding['dataset_id']
    lookup = {f['id']:f for f in current.get('cleaning_review_catalog_snapshot', [])}
    allowed_decisions = {'source_verified_choice_approved_once','no_change_authorized_or_executed',
                         'resolved_by_prior_approvals','review_finished_no_additional_edits',
                         'visual_flag_applied_by_reviewer_once'}
    for decision in new_decisions[len(old_decisions):]:
        if decision.get('decision') not in allowed_decisions:
            return False
        if decision.get('decision') == 'no_change_authorized_or_executed':
            request = str(decision.get('user_request') or '').strip().casefold()
            if request not in {'keep as is','keep current data','mark reviewed — accepted unchanged (no correction asserted)'}:
                return False
    new_events = new_history[len(old_history):]
    new_flags = [e for e in new_events if e.get('operation') == 'flag_cells']
    # A completed, local annotation instruction is not a new global policy.
    # Other discussion (including rejections not followed by this exact
    # approval) still invalidates speculative advice.
    import re
    from healthcare_data_quality.cleaning.visual_flags import is_visual_flag_request, visual_flags_from_history
    old_turns, current_turns = _user_turns(old), _user_turns(current)
    approved_visual = {}
    for decision in new_decisions[len(old_decisions):]:
        if decision.get('decision') == 'visual_flag_applied_by_reviewer_once':
            fid = decision.get('finding_id'); request = str(decision.get('user_request') or '')
            if (not any(e.get('finding_id') == fid and e.get('reviewer') == current.get('cleaning_approved_reviewer') for e in new_flags)
                    or not is_visual_flag_request(request)
                    or re.search(r'\b(?:always|never|future|every|all findings|from now|prefer|policy|instead of)\b', request, re.I)):
                return False
            approved_visual[fid] = request
    for fid in set(old_turns) | set(current_turns):
        before, after = old_turns.get(fid, []), current_turns.get(fid, [])
        if before == after:
            continue
        if not _append_only(before, after) or fid not in approved_visual:
            return False
        if any(t != approved_visual[fid] and t.strip().casefold() != 'no — discuss another idea' for t in after[len(before):]):
            return False
    for event in new_events:
        if event.get('operation') not in {'replace_cells','rename_headers','flag_cells'}:
            return False  # structural changes are never inferred independent
        if event.get('operation') == 'flag_cells' and event.get('finding_id') not in approved_visual:
            return False
        origin = lookup.get(event.get('finding_id'))
        if origin is None or _identity(origin, source)['finding_type'] == _identity(finding, source)['finding_type']:
            return False
    for key in ('cleaning_approved_rows_by_dataset','cleaning_approved_columns_by_dataset'):
        if old.get(key, {}).get(ds, []) != current.get(key, {}).get(ds, []):
            return False
    old_flags = old.get('cleaning_visual_flags_by_dataset', {}).get(ds, [])
    current_flags = current.get('cleaning_visual_flags_by_dataset', {}).get(ds, [])
    if old_flags != current_flags or new_flags:
        # Only ledger-backed annotation changes may be promoted. Unknown or
        # modified styling needs a fresh assessment just like unknown edits.
        if not new_flags or current_flags != visual_flags_from_history(new_history).get(ds, []):
            return False
        changed_flags = {(f['row'],f['column']) for e in new_flags if e.get('dataset_id') == ds for f in e.get('flags', [])}
        # An in-flight private job can finish; promotion will run the full
        # check again with its exact candidates before any buttons appear.
        if candidates is not None:
            for option in candidates:
                for op in option.get('operations', []):
                    for ch in op.get('changes', []):
                        kind = op.get('operation')
                        if kind in {'replace_cells','flag_cells'} and (ch.get('row'),ch.get('column')) in changed_flags:
                            return False
                        if kind in {'drop_rows','merge_rows'} and any(r in {ch.get('row'),ch.get('source_row')} for r,c in changed_flags):
                            return False
                        if kind in {'drop_column','rename_headers'} and any(c == ch.get('position') for r,c in changed_flags):
                            return False
                        if kind not in {'replace_cells','flag_cells','drop_rows','merge_rows','drop_column','rename_headers'}:
                            return False
        if finding.get('check_key') == 'gemma_check':
            return False  # unknown exploratory dependencies are conservative
    left, lr, lc = projected_records(old, ds, source)
    right, rr, rc = projected_records(current, ds, source)
    if lr != rr or lc != rc:
        return False
    changes = {(r,c):(x,y) for r, a,b in zip(lr,left,right) for c,x,y in zip(lc,a,b) if x != y}
    # A data difference without a matching new approved-cell record is never
    # independent, even if the change happened outside the proposed targets.
    approved = {}
    for event in new_history[len(old_history):]:
        for cell in event.get('approved_cell_changes', []):
            approved[(cell.get('row'), cell.get('column'))] = (cell.get('before'), cell.get('after'))
    if any(approved.get(pos) != values for pos, values in changes.items()):
        return False
    if not changes:
        return True
    key = finding.get('check_key')
    if key in {'inspect_taxonomy_structure','inspect_taxonomy_naming'}:
        if any(c == 1 or r == 1 for r,c in changes):
            return False
    elif key == 'find_duplicate_taxa':
        # Duplicate membership depends on ALL taxonomy labels. Decisions to
        # delete, merge or flag duplicate rows also depend on their counts.
        # A count correction outside every duplicated row changes neither.
        from collections import Counter
        labels = Counter(row[0] for row in left[1:])
        duplicated = {r for r, row in zip(lr[1:], left[1:]) if labels[row[0]] > 1}
        if any(r == 1 or c == 1 or r in duplicated for r,c in changes):
            return False
    elif key == 'compare_repeated_headers':
        # Repeated-header findings compare the entire named physical columns,
        # not taxonomy labels or unrelated measurements. Preserve original IDs
        # and all header values; a header edit could change duplicate groups.
        pairs = set()
        for evidence in finding.get('evidence', []):
            if not isinstance(evidence, dict):
                return False
            a, b = evidence.get('Column A'), evidence.get('Column B')
            if type(a) is not int or type(b) is not int or a == b:
                return False
            pairs.update((a, b))
        if not pairs or not pairs.issubset(lc):
            return False
        if any(r == 1 or c in pairs for r, c in changes):
            return False
    elif key == 'inspect_headers':
        if any(r == 1 for r,c in changes):
            return False
    elif key in {'find_invalid_counts','find_negative_counts'}:
        from decimal import Decimal, InvalidOperation
        title = str(finding.get('title','')).casefold()
        issue = ('missing' if 'missing' in title else 'negative' if 'negative' in title or key == 'find_negative_counts'
                 else 'fractional' if 'fractional' in title else 'non_numeric' if 'non-numeric' in title or 'nonnumeric' in title else None)
        if not issue:
            return False
        def category(value):
            value = value.strip()
            if not value:
                return 'missing'
            try:
                n = Decimal(value)
                if not n.is_finite(): return 'non_numeric'
                if n < 0: return 'negative'
                if n != n.to_integral_value(): return 'fractional'
                return 'valid'
            except InvalidOperation:
                return 'non_numeric'
        # Compare the full affected predicate, not a sample from the finding.
        evidence_rows = {e.get('CSV row',e.get('csv_row')) for e in finding.get('evidence',[]) if isinstance(e,dict)}
        for (r,c),(before,after) in changes.items():
            if r == 1 or (c == 1 and (r in evidence_rows or issue == 'missing')):
                return False
            if c > 1 and (category(before) == issue or category(after) == issue):
                return False
    else:
        # Count, duplicate, aggregate and Gemma checks can depend on the whole
        # matrix, including distributions and previously normal observations.
        return False
    # A pending draft with otherwise unchanged deterministic evidence may
    # finish, but it is NEVER promoted until the exact operations are checked.
    if candidates is None:
        return allow_pending
    for option in candidates:
        for op in option.get('operations', []):
            kind = op.get('operation')
            for ch in op.get('changes', []):
                if kind in {'replace_cells','rename_headers','flag_cells'}:
                    target = (ch.get('row',1), ch.get('column',ch.get('position')))
                    if target in changes:
                        return False
                    if kind == 'replace_cells' and target[1] != 1:
                        from healthcare_data_quality.cleaning.numeric_context import _number
                        if _number(ch.get('after')) is not None and any(c == target[1] for r,c in changes):
                            return False  # same-column distributions changed
                elif kind in {'drop_rows','merge_rows'}:
                    if any(r in {ch.get('row'),ch.get('source_row')} for r,c in changes):
                        return False
                elif kind == 'drop_column':
                    if any(c == ch.get('position') for r,c in changes):
                        return False
                else:
                    return False
    return True


def promote(record, current, finding, source):
    """Return fresh-revision advice only after full current-state revalidation."""
    from healthcare_data_quality.services.model_plan_review import dataset_review_context
    from healthcare_data_quality.cleaning.previous_approval_resolution import source_verified_prior_row_resolution
    result = record.get('result') or {}
    if record.get('source_sha256') != sha256(Path(source).read_bytes()).hexdigest() or record.get('finding') != finding:
        return None
    if not dependencies_unchanged(record['snapshot'], current, finding, source, result.get('candidates')):
        return None
    revision = dataset_review_context(current, finding['dataset_id'], finding['id'])['revision']
    if result.get('closure'):
        resolution = source_verified_prior_row_resolution(finding, source,
            current.get('cleaning_approved_rows_by_dataset', {}).get(finding['dataset_id'], []))
        if resolution == result.get('resolution') and result['closure'].get('can_offer'):
            return {'closure':{**result['closure'], 'review_revision':revision}}
        return None
    suggestion = result.get('suggestion')
    if not suggestion or not result.get('ready_count'):
        return None
    choices = _candidates(current, finding, source, suggestion)
    if _signatures(finding, source, choices) != result.get('signatures'):
        return None
    # Do not modify original assessment fingerprints. Only an independently
    # demonstrated unchanged dependency set allows the revision to advance.
    return {'suggestion':{**suggestion,'_review_revision':revision}}


def preferred_structural_branch(choices):
    """Only use an already-assessed, highest-ranked historical structural choice.

    Never predict a free-text instruction, a disputed action or a value edit.
    The regular two future drafts take precedence over this optional work.
    """
    if not choices:
        return None
    choice = choices[0]
    ops = choice.get('operations') or []
    if (choice.get('gemma_review_status') == 'supported'
            and choice.get('historical_approval_count', 0) > 0
            and ops and all(p.get('operation') in {'drop_rows','drop_column','merge_rows'} for p in ops)):
        return deepcopy(choice)
    return None


def prepare_conditional(base, origin, choice, finding, source, stop, closure_worker=None):
    """Assess the next finding against an isolated hypothetical structural edit."""
    from healthcare_data_quality.ui.one_click_choice import project_structural_choice
    if stop.is_set():
        return {'cancelled': True}
    predicted = snapshot(project_structural_choice(base, origin, source, choice))
    if stop.is_set():
        return {'cancelled': True}
    result = prepare(predicted, finding, source, stop, closure_worker)
    return {'predicted': predicted, 'prepared': result}


def branch_selected(base, current, origin, choice, source):
    """Cheap pending-work check; not enough to authorize or publish a result.

    Once the branch finishes, promote() independently compares its complete
    predicted approval state and exact candidate signatures with reality.
    """
    if base.get('cleaning_approved_reviewer') != current.get('cleaning_approved_reviewer'):
        return False
    if _user_turns(base) != _user_turns(current):
        return False
    old, new = base.get('cleaning_history', []), current.get('cleaning_history', [])
    if not _append_only(old, new):
        return False
    old_d, new_d = base.get('cleaning_decisions', []), current.get('cleaning_decisions', [])
    if not _append_only(old_d, new_d) or len(new_d) != len(old_d)+1:
        return False
    decision = new_d[-1]
    if (decision.get('finding_id') != origin['id'] or decision.get('decision') != 'source_verified_choice_approved_once'
            or decision.get('reviewer') != current.get('cleaning_approved_reviewer')):
        return False
    events, ops = new[len(old):], choice.get('operations') or []
    if not ops or len(events) != len(ops):
        return False
    digest = sha256(Path(source).read_bytes()).hexdigest()
    if decision.get('source_sha256') != digest:
        return False
    for event, op in zip(events, ops):
        kind = op.get('operation')
        if (event.get('operation') != kind or event.get('finding_id') != origin['id']
                or event.get('dataset_id') != origin['dataset_id'] or event.get('source_sha256') != digest
                or event.get('reviewer') != current.get('cleaning_approved_reviewer')):
            return False
        if kind == 'drop_column':
            positions = [{'column': c['position']} for c in op['changes']]
        elif kind in {'drop_rows','merge_rows'}:
            positions = [{k:v for k,v in c.items() if k in {'row','source_row'}} for c in op['changes']]
        else:
            return False
        if positions != event.get('modified_positions'):
            return False
    return True
