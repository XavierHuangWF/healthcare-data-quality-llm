"""Atomic single-click approval for exact source-backed Agent B choices.

The single Yes button is the named human's authorization. All selected actions
are revalidated, tested (including R where applicable), and committed only
on successful independent export verification; the input CSV is never edited.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


def prepare_verified_choice_snapshot(state, finding: dict, source: Path, option: dict) -> dict:
    from healthcare_data_quality.cleaning.compound_approval import (
        evaluate_compound, validate_compound_targets,
    )
    from healthcare_data_quality.cleaning.in_memory_cleaning import sha256_bytes

    fid, ds_id = finding['id'], finding['dataset_id']
    reviewer = str(state.get('cleaning_approved_reviewer') or '').strip()
    if not reviewer or len(reviewer) > 120:
        raise PermissionError('Register your reviewer identifier before approving')
    if (state.get('cleaning_review_statuses', {}).get(fid) in {
        'Applied and verified', 'Reviewed — no change',
        'Approved row changes for final export', 'Approved column deletion for final export',
    } or any(h.get('finding_id') == fid for h in state.get('cleaning_history', []))):
        raise PermissionError('This finding has already been reviewed; reopen it to revise')
    plans = option.get('operations')
    if not isinstance(plans, list) or not 1 <= len(plans) <= 16:
        raise ValueError('No verified actions associated with the selected choice')
    validate_compound_targets(plans)
    from healthcare_data_quality.cleaning.finding_cleaning import validate_scoped_plan
    requested_columns = frozenset(c['position'] for p in plans if p['operation'] == 'drop_column'
                                  for c in p['changes'])
    requested_rows = frozenset(c['row'] for p in plans if p['operation'] == 'drop_rows'
                               for c in p['changes'])
    # Source-backed UI rows are only pre-validation records. The canonical
    # scoped validator seals each plan with original-source provenance.
    plans = [validate_scoped_plan(
        plan, finding, Path(source),
        user_requested_columns=requested_columns or None,
        user_requested_rows=requested_rows or None,
    ) for plan in plans]
    original = Path(source).read_bytes()
    staged = state.get('cleaning_staged_bytes_by_dataset', {}).get(ds_id, original)
    columns = state.get('cleaning_approved_columns_by_dataset', {}).get(ds_id, [])
    rows = state.get('cleaning_approved_rows_by_dataset', {}).get(ds_id, [])
    from healthcare_data_quality.cleaning.proposal_conflicts import proposal_conflicts
    # Repeat the annotation-conflict gate at the named approval boundary.
    conflicts = proposal_conflicts(plans, source, staged, rows, columns,
        approved_flags=state.get('cleaning_visual_flags_by_dataset', {}).get(ds_id, []))
    if conflicts:
        raise PermissionError('; '.join(conflicts[:3]))
    # No preview state or authorisation is read from Agent B or from memory.
    # These validators derive their own original source coordinates/values.
    result = evaluate_compound(
        plans, finding, Path(source), staged, columns, rows, reviewer=reviewer,
        requested_columns=requested_columns or None,
        requested_rows=requested_rows or None,
        allow_single=True,
    )
    if result['original_sha256'] != sha256_bytes(original) or result['staged_sha256'] != sha256_bytes(staged):
        raise PermissionError('Original or staged dataset changed before commit')
    if len(result['events']) != len(plans):
        raise PermissionError('Verified operation count differs from the reviewed proposal')
    return {'plans': plans, 'result': result, 'reviewer': reviewer}


def _stage_verified_choice_snapshot(state, finding: dict, source: Path, option: dict, prepared: dict):
    """Apply a verified snapshot to the supplied mapping, without learning writes."""
    from healthcare_data_quality.cleaning.in_memory_cleaning import sha256_bytes
    from healthcare_data_quality.cleaning.compound_approval import proposal_digest
    fid, ds_id = finding['id'], finding['dataset_id']
    plans, result, reviewer = prepared['plans'], prepared['result'], prepared['reviewer']
    original = Path(source).read_bytes()
    staged = state.get('cleaning_staged_bytes_by_dataset', {}).get(ds_id, original)
    columns = state.get('cleaning_approved_columns_by_dataset', {}).get(ds_id, [])
    rows = state.get('cleaning_approved_rows_by_dataset', {}).get(ds_id, [])
    if (str(state.get('cleaning_approved_reviewer') or '').strip() != reviewer
        or result['original_sha256'] != sha256_bytes(original)
        or result['staged_sha256'] != sha256_bytes(staged)
        or result['digest'] != proposal_digest(plans, original, staged, columns, rows)
        or any(h.get('finding_id') == fid for h in state.get('cleaning_history', []))
        or state.get('cleaning_review_statuses', {}).get(fid) in {'Applied and verified','Reviewed — no change','Approved row changes for final export','Approved column deletion for final export','Resolved by earlier approved edit'}):
        raise PermissionError('Review changed during verification; approve the current proposal again')
    # Only now commit the complete, independent preflight. No partial edits.
    history = []
    for event in result['events']:
        history.append({
            **event, 'dataset_id': ds_id, 'finding_id': fid, 'finding': finding['title'],
            'reviewer': reviewer, 'source_sha256': result['original_sha256'],
            'staged_sha256': sha256_bytes(result['staged_after']),
            'storage': ('session_memory_export_only' if event['operation'] in
                        {'drop_column', 'drop_rows', 'merge_rows'} else 'session_memory'),
            'approval_group': result['digest'][:16],
            'gemma_review_status': option.get('gemma_review_status', 'not_assessed'),
            'gemma_review_reason': str(option.get('gemma_review_reason') or '')[:300],
        })
    state.setdefault('cleaning_staged_bytes_by_dataset', {})[ds_id] = result['staged_after']
    state.setdefault('cleaning_approved_columns_by_dataset', {})[ds_id] = result['columns_after']
    state.setdefault('cleaning_approved_rows_by_dataset', {})[ds_id] = result['rows_after']
    state.setdefault('cleaning_history', []).extend(history)
    state.setdefault('cleaning_review_statuses', {})[fid] = 'Applied and verified'
    state.setdefault('cleaning_decisions', []).append({
        'finding_id': fid, 'decision': 'source_verified_choice_approved_once',
        'operations': len(plans), 'reviewer': reviewer,
        'gemma_review_status': option.get('gemma_review_status', 'not_assessed'),
        'gemma_review_reason': str(option.get('gemma_review_reason') or '')[:300],
        'source_sha256': result['original_sha256'], 'export_sha256': result['export_sha256'],
    })
    proofs = list(result['r_previews'].values())
    state.setdefault('cleaning_completed_audits_by_finding', {})[fid] = {
        'r_function': '\n\n'.join(str(p.get('r_function') or '') for p in proofs),
        'insertion_r_function': '\n\n'.join(str(p.get('insertion_r_function') or '') for p in proofs),
        'changes': [c for plan in plans for c in plan['changes']],
    }
    for name in ('cleaning_plans_by_finding', 'cleaning_batch_queue_by_finding',
                 'cleaning_batch_total_by_finding', 'cleaning_compound_previews_by_finding',
                 'cleaning_r_previews_by_finding', 'cleaning_r_errors_by_finding',
                 'cleaning_explicit_row_deletions_by_finding',
                 'cleaning_explicit_column_deletions_by_finding',
                 'cleaning_pending_gemma_b', 'cleaning_discussion_mode_by_finding'):
        state.setdefault(name, {}).pop(fid, None)
    return plans


def project_structural_choice(state, finding, source, option):
    """A private hypothetical state only. Never writes to memory or real state.

    Structural options use the deterministic replay path, not speculative R
    execution. The real action must still go through normal named approval.
    """
    from copy import deepcopy
    operations = option.get('operations') or []
    if not operations or any(p.get('operation') not in {'drop_rows', 'drop_column', 'merge_rows'} for p in operations):
        raise ValueError('Only structural choices are supported for conditional prefetch')
    private = deepcopy(state)
    prepared = prepare_verified_choice_snapshot(private, finding, source, option)
    _stage_verified_choice_snapshot(private, finding, source, option, prepared)
    return private


def commit_verified_choice_snapshot(state, finding: dict, source: Path, option: dict, prepared: dict):
    """Commit on the UI thread only after the calling revision gate passes."""
    from healthcare_data_quality.learning.interaction_memory import save_reviewed_solution
    from healthcare_data_quality.cleaning.reviewed_rule_replay import annotate_source_headers
    plans = _stage_verified_choice_snapshot(state, finding, source, option, prepared)
    # A completed approval persists a new example, not a general-purpose
    # multiplication rule. Persistence failure never falsifies execution.
    warning = None
    from healthcare_data_quality.ui.clear_review_actions import option_action_label
    try:
        save_reviewed_solution(
            finding, source, operations=annotate_source_headers(plans, Path(source)),
            decision='approved_edit',
            user_request=str(option.get('label') or '')[:500],
            agent_response='Exact source-backed choice accepted and independently verified',
            button_label=option_action_label(option, finding),
        )
    except (OSError, ValueError, TypeError, UnicodeError) as exc:
        warning = f'Edit verified and applied, but learning memory was not saved: {exc}'
    return warning


def execute_verified_choice(state, finding: dict, source: Path, option: dict) -> dict:
    """Synchronous API for non-UI callers; the Streamlit app queues preparation."""
    prepared = prepare_verified_choice_snapshot(state, finding, source, option)
    warning = commit_verified_choice_snapshot(state, finding, source, option, prepared)
    return {'result': prepared['result'], 'warning': warning}
