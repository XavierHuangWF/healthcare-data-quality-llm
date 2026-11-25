"""Regression: allow source-proven cross-finding closures, never unsupported completion."""
from __future__ import annotations

import hashlib
import pytest
from pathlib import Path

from healthcare_data_quality.cleaning import cleaning_stage_checkpoint as stage
from healthcare_data_quality.cleaning.final_export_integrity import verify_final_csv
from healthcare_data_quality.cleaning.in_memory_cleaning import (
    _records, apply_approved_row_operations, apply_approved_column_deletions,
)
from healthcare_data_quality.ui.gemma_check_evidence import check_evidence_rows


def scenario(tmp_path):
    # Four-row defect entirely resolved by another named, approved deletion.
    data = b'Taxonomy,Value\nA,1\nBad,2\nC,3\nBad,4\n'
    source = tmp_path / 'source.csv'
    source.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    affected = [3, 5]
    approved = [{'operation': 'drop_rows', 'source_sha256': digest, 'reviewer': 'R',
                 'finding_id': 'f2', 'changes': [
                     {'row': r, 'before': _records(data)[r - 1][0], 'after': ''}
                     for r in affected]}]
    event = {'dataset_id': 'dataset_1', 'finding_id': 'f2', 'operation': 'drop_rows',
             'source_sha256': digest, 'staged_sha256': digest, 'reviewer': 'R',
             'modified_positions': [{'row': r} for r in affected], 'changes': len(affected)}
    proof = {'finding_id': 'f5', 'dataset_id': 'dataset_1',
             'decision': 'resolved_by_prior_approvals', 'source_sha256': digest,
             'reviewer': 'R', 'original_csv_rows': affected,
             'earlier_approved_findings': ['f2'], 'new_edits': 0}
    statuses = {'f2': 'Approved row changes for final export',
                'f5': 'Resolved by earlier approved edit'}
    titles = {'f2': 'Rows containing blanks', 'f5': '2 rows have an incorrect rank count'}
    finding = {'id': 'f5', 'dataset_id': 'dataset_1', 'title': titles['f5'],
               'evidence': [{'CSV row': r} for r in affected]}
    final = apply_approved_row_operations(data, data, approved)
    return source, data, approved, event, proof, statuses, titles, finding, final


def test_earlier_verified_deletion_resolves_finding_and_enables_final_download(tmp_path, monkeypatch):
    source, raw, rows, event, proof, statuses, titles, finding, final = scenario(tmp_path)
    verified = verify_final_csv(source, raw, final, history=[event],
        review_statuses=statuses, finding_titles=titles, approved_rows=rows,
        approved_columns=[], finding_checks={'f5': finding}, review_decisions=[proof])
    assert verified['removed_rows'] == 2
    monkeypatch.setenv('HDQ_APPROVED_STAGE_DIR', str(tmp_path/'saved'))
    saved = stage.save_approved_stage([source], {}, {}, {'dataset_1': rows},
        [event], statuses, 'R', titles, review_decisions=[proof])
    assert saved.exists()
    restored = stage.verify_and_reconstruct(stage.load_approved_stage([source]),
                                           [source], titles)
    assert restored['cleaning_review_statuses']['f5'] == 'Resolved by earlier approved edit'
    assert restored['cleaning_decisions'][0]['finding_id'] == 'f5'


@pytest.mark.parametrize('tamper', ['missing_proof','wrong_rows','wrong_source',
                                     'wrong_prior_finding','false_completion','missing_deletion'])
def test_unproven_closure_blocks_export(tmp_path, tamper):
    source, raw, rows, event, proof, statuses, titles, finding, final = scenario(tmp_path)
    proof = dict(proof)
    if tamper == 'missing_proof':
        decisions = []
    else:
        decisions = [proof]
    if tamper == 'wrong_rows':
        proof['original_csv_rows'] = [2, 3]
    if tamper == 'wrong_source':
        proof['source_sha256'] = '0'*64
    if tamper == 'wrong_prior_finding':
        proof['earlier_approved_findings'] = ['f99']
    if tamper == 'false_completion':
        statuses['f5'] = 'Applied and verified'
    if tamper == 'missing_deletion':
        rows = []
    with pytest.raises((PermissionError, ValueError)):
        verify_final_csv(source, raw, final, history=[event],
            review_statuses=statuses, finding_titles=titles, approved_rows=rows,
            approved_columns=[], finding_checks={'f5': finding},
            review_decisions=decisions)


def test_gemma_check_shows_real_hypothesis_method_result_not_invented_cells():
    finding = {'source': 'Gemma check', 'check_key': 'gemma_check',
               'gemma_check_details': {
                   'proposal': {'hypothesis': 'Are blank control columns nonzero?',
                                'method': 'Read-only Python counts for designated controls'},
                   'result': {'summary': 'Found zero non-zero entries', 'findings': []},
                   'verification': {'status': 'source_cells_verified_and_output_reproduced'}}}
    rows = check_evidence_rows(finding)
    assert len(rows) >= 5
    rendered = ' '.join(str(x) for x in rows)
    assert 'blank control columns' in rendered
    assert 'Found zero non-zero entries' in rendered
    assert '0' in rendered
    assert 'source cells verified' in rendered.lower()
    assert 'delete columns 2, 3, 4' not in rendered.lower()
    assert 'does not justify deleting' in rendered


def test_app_renders_gemma_evidence_and_passes_user_review_proofs():
    content = (Path(__file__).resolve().parents[1] / 'app.py').read_text()
    assert 'check_evidence_rows(finding)' in content
    assert 'review_decisions=st.session_state.cleaning_decisions' in content


def test_real_microbiome_fixture_prior_deleted_rows_can_be_exported():
    """Exercise real 158-row CSV coordinates, not just toy source records."""
    source = Path(__file__).resolve().parents[1] / 'data' / 'dirtyL7PVDforXav.csv'
    if not source.exists():
        pytest.skip('Project fixture was not installed')
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    positions = [3, 5, 15, 20]
    names = _records(raw)
    approvals = [{'operation': 'drop_rows', 'source_sha256': digest,
                  'reviewer': 'Reviewer', 'finding_id': 'dataset_1:qc:2',
                  'changes': [{'row': r, 'before': names[r-1][0], 'after': ''}
                              for r in positions]}]
    event = {'dataset_id': 'dataset_1', 'finding_id': 'dataset_1:qc:2',
             'operation': 'drop_rows', 'changes': len(positions),
             'reviewer': 'Reviewer', 'source_sha256': digest,
             'staged_sha256': digest,
             'modified_positions': [{'row': r} for r in positions]}
    fid = 'dataset_1:qc:5'
    decision = {'finding_id': fid, 'decision': 'resolved_by_prior_approvals',
                'reviewer': 'Reviewer', 'source_sha256': digest,
                'original_csv_rows': positions,
                'earlier_approved_findings': ['dataset_1:qc:2'], 'new_edits': 0}
    finding = {'id': fid, 'dataset_id': 'dataset_1',
               'title': '4 rows have an incorrect number of taxonomy levels',
               'evidence': [{'CSV row': r} for r in positions]}
    out = apply_approved_row_operations(raw, raw, approvals)
    verification = verify_final_csv(
        source, raw, out, history=[event], review_decisions=[decision],
        review_statuses={'dataset_1:qc:2': 'Approved row changes for final export',
                         fid: 'Resolved by earlier approved edit'},
        finding_titles={'dataset_1:qc:2': 'Missing values', fid: finding['title']},
        approved_rows=approvals, approved_columns=[], finding_checks={fid: finding})
    assert verification['removed_rows'] == 4


def test_complete_review_zip_contains_source_cleaned_excel_and_cross_finding_audit(tmp_path):
    import io
    import zipfile
    from healthcare_data_quality.cleaning.review_zip_bundle import build_review_zip_bundle
    from healthcare_data_quality.cleaning.styled_export import build_styled_excel_download

    source, raw, rows, event, proof, statuses, titles, finding, final = scenario(tmp_path)
    excel_name, xlsx = build_styled_excel_download(
        final, original_csv=raw, original_filename='source.csv',
        approved_flags=[], approved_row_operations=rows, approved_column_deletions=[])
    filename, bundle = build_review_zip_bundle(
        original_csv=raw, cleaned_csv=final, original_filename='source.csv',
        reviewer='R', history=[event], decisions=[proof],
        review_statuses=statuses, conversations={},
        approved_row_operations=rows, approved_column_deletions=[],
        completed_audits={}, finding_titles=titles,
        excel_attachment=(excel_name, xlsx))
    assert filename.endswith('.zip')
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert archive.read('source/original_source.csv') == raw
        assert archive.read('cleaned/cleaned_source.csv') == final
        assert 'cleaned/cleaned_highlighted.xlsx' in archive.namelist()
        trail = archive.read('audit/approved_changes.csv').decode('utf-8')
        assert 'review_resolved_by_prior_approvals' in trail
