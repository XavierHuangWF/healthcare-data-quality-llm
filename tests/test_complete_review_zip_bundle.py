"""Complete source-linked export must contain CSV, reviewer interactions and hashes."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from healthcare_data_quality.cleaning.review_zip_bundle import build_review_zip_bundle


def _build(**overrides):
    original = b"Taxon,ABC1,ABC2\nBacteria;A,0.2,4\nBacteria;B,0.5,5\n"
    cleaned = b"Taxon,ABC1,ABC2\nBacteria;A,2,4\nBacteria;B,0.5,5\n"
    event = {
        "finding_id": "finding_1", "dataset_id": "dataset_1",
        "finding": "fractional counts", "operation": "replace_cells",
        "reviewer": "Reviewer X", "source_sha256": hashlib.sha256(original).hexdigest(),
        "approved_cell_changes": [{"row": 2, "column": 2, "before": "0.2", "after": "2"}],
    }
    args = dict(original_csv=original, cleaned_csv=cleaned,
                original_filename="dirty.csv", reviewer="Reviewer X",
                history=[event], decisions=[{"finding_id": "finding_1", "decision": "approved"}],
                review_statuses={"finding_1": "Applied and verified"},
                conversations={"finding_1": [
                    {"role": "user", "content": "make 0.2 become 2"},
                    {"role": "assistant", "content": "Proposed 0.2 → 2"},
                ]},
                approved_row_operations=[], approved_column_deletions=[],
                completed_audits={}, finding_titles={"finding_1": "fractional counts"},
                agent_b_suggestions={"finding_1": {"suggestion": "multiply by ten"}})
    args.update(overrides)
    return build_review_zip_bundle(**args)


def test_zip_includes_original_cleaned_and_real_interaction_records():
    filename, payload = _build()
    assert filename == "review_package_dirty.zip"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        files = set(archive.namelist())
        assert {"source/original_dirty.csv", "cleaned/cleaned_dirty.csv", "manifest.json",
                "audit/reviewer_audit.json", "audit/reviewer_interactions.csv",
                "audit/approved_changes.csv", "audit/finding_decisions.csv",
                "README.txt"} <= files
        manifest = json.loads(archive.read('manifest.json'))
        for path, info in manifest['files'].items():
            data = archive.read(path)
            assert hashlib.sha256(data).hexdigest() == info['sha256']
            assert len(data) == info['bytes']
        audit = json.loads(archive.read('audit/reviewer_audit.json'))
        assert audit['reviewer'] == 'Reviewer X'
        assert audit['original_sha256'] == hashlib.sha256(archive.read('source/original_dirty.csv')).hexdigest()
        assert audit['cleaned_sha256'] == hashlib.sha256(archive.read('cleaned/cleaned_dirty.csv')).hexdigest()
        assert audit['agent_b_suggestions_at_export']['finding_1']['suggestion'] == 'multiply by ten'
        assert audit['conversation'][0]['message'] == 'make 0.2 become 2'
        assert audit['change_reconciliation'][0]['original_location'] == 'R2:C2'
        assert audit['change_reconciliation'][0]['final_csv_value'] == '2'
        text = archive.read('audit/reviewer_interactions.csv').decode('utf-8-sig')
        records = list(csv.DictReader(io.StringIO(text)))
        assert [x['Exact message'] for x in records] == [
            'make 0.2 become 2', 'Proposed 0.2 → 2'
        ]


def test_incomplete_review_does_not_produce_download():
    with pytest.raises(PermissionError, match='incomplete'):
        _build(review_statuses={'finding_1': 'Discussion open'})
    with pytest.raises(PermissionError, match='incomplete'):
        _build(review_statuses={'finding_1': 'Applied and verified'},
               finding_titles={'finding_1': 'first', 'finding_2': 'not reviewed'})
    with pytest.raises(PermissionError, match='named reviewer'):
        _build(reviewer='')


def test_no_change_decisions_can_be_exported_without_actions():
    filename, payload = _build(
        original_csv=b'Taxon,ABC1\nA,0\n',
        cleaned_csv=b'Taxon,ABC1\nA,0\n',
        history=[],
        decisions=[{'finding_id': 'f', 'decision': 'keep as is'}],
        review_statuses={'f': 'Reviewed — no change'},
        finding_titles={'f': 'blank header accepted'},
        conversations={'f': [{'role': 'user', 'content': 'keep it as is'}]},
    )
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        report = json.loads(archive.read('audit/reviewer_audit.json'))
        assert report['approved_operations'] == []
        assert report['finding_reviews'][0]['status'] == 'Reviewed — no change'
        assert report['conversation'][0]['message'] == 'keep it as is'


def test_export_detects_approved_edit_disagreement():
    with pytest.raises(PermissionError, match='mismatch'):
        _build(cleaned_csv=b'Taxon,ABC1,ABC2\nBacteria;A,3,4\nBacteria;B,0.5,5\n')


def test_installation_is_idempotent_and_changes_one_active_button():
    import importlib.util
    import zipfile
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    script = root / 'tools' / 'apply_complete_review_zip.py'
    spec = importlib.util.spec_from_file_location('zip_installer_for_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = root / 'app.before_review_zip.bak'
    if not old.exists():
        pytest.skip('Before-install app snapshot is not present')
    original = old.read_text(encoding='utf-8')
    revised = module.transform(original)
    assert revised != original
    assert module.transform(revised) == revised
    assert revised.count('from healthcare_data_quality.cleaning.review_zip_bundle import build_review_zip_bundle') == 1
    assert revised.count('Download cleaned CSV + complete reviewer audit (ZIP)') == 2


def test_zip_embeds_audit_in_styled_excel_when_selected():
    from healthcare_data_quality.cleaning.styled_export import build_styled_excel_download
    source = b'Taxon,ABC1,ABC2\nBacteria;A,0.2,4\nBacteria;B,0.5,5\n'
    cleaned = b'Taxon,ABC1,ABC2\nBacteria;A,2,4\nBacteria;B,0.5,5\n'
    _, excel = build_styled_excel_download(
        cleaned, original_csv=source, original_filename='dirty.csv',
        approved_flags=[], approved_row_operations=[], approved_column_deletions=[])
    name, output = _build(excel_attachment=('cleaned.xlsx', excel))
    with zipfile.ZipFile(io.BytesIO(output)) as archive:
        from openpyxl import load_workbook
        workbook = load_workbook(io.BytesIO(archive.read('cleaned/cleaned_highlighted.xlsx')))
        assert 'Cleaned dataset' in workbook.sheetnames
        assert 'Audit trail' in workbook.sheetnames
        assert 'Conversation' in workbook.sheetnames
        assert workbook['Cleaned dataset']['B2'].value == 2
        messages = [r[3] for r in workbook['Conversation'].iter_rows(min_row=2, values_only=True)]
        assert 'make 0.2 become 2' in messages
