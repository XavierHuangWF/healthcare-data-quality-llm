from pathlib import Path
import hashlib
from healthcare_data_quality.cleaning.current_review_evidence import current_finding_evidence, approval_finding_label, approval_action_label
from healthcare_data_quality.cleaning.finding_cleaning import build_cleaning_findings
from healthcare_data_quality.cleaning.in_memory_cleaning import _records
from healthcare_data_quality.investigation.baseline_audit import run_baseline_audit
from test_cleaning_review_cards import StubUI, _app

SRC = Path(__file__).resolve().parents[1] / 'data' / 'dirtyL7PVDforXav.csv'


def fixture():
    catalog = build_cleaning_findings({'baseline': run_baseline_audit(SRC)}, [SRC], [SRC.name], [])
    raw = SRC.read_bytes()
    entry = {'finding_id': catalog[1]['id'], 'operation': 'drop_rows', 'reviewer': 'Reviewer',
             'source_sha256': hashlib.sha256(raw).hexdigest(),
             'changes': [{'row': r, 'before': _records(raw)[r-1][0]} for r in (2,3,5,7,15,20)]}
    return catalog, entry


def test_real_finding_six_displays_finding_two_and_exact_approved_action():
    catalog, entry = fixture()
    state = {'cleaning_review_catalog_snapshot': catalog,
             'cleaning_approved_rows_by_dataset': {'dataset_1': [entry]}}
    row, = current_finding_evidence(state, catalog[5], SRC, display=True)
    assert row['Location'] == 'R7:C1'
    assert row['Approved by'] == 'Finding 2 · ' + catalog[1]['title']
    assert row['Action'] == 'Delete rows 2, 3, 5, 7, 15, 20'
    assert row['Current'] == '(removed on export)'
    # Model/audit consumers retain the original identifiers.
    raw, = current_finding_evidence(state, catalog[5], SRC)
    assert raw['Approved by'] == entry['finding_id'] and raw['Action'] == 'drop_rows'
    entry['source_sha256'] = 'wrong-source'
    stale, = current_finding_evidence(state, catalog[5], SRC, display=True)
    assert stale['Approved by'] == stale['Action'] == ''


def test_display_origin_for_multiple_findings_and_action_types():
    catalog = [{'id':'gemma:2:check','source':'Gemma check','title':'Control counts'}]
    assert approval_finding_label({'finding_id':'gemma:2:check'}, catalog) == 'Gemma check 1 · Control counts'
    assert approval_finding_label({'finding_id':'dataset_1:qc:2'}, [], 'dataset_1') == 'Finding 2'
    assert approval_finding_label({}, [], 'dataset_1') == 'Origin not recorded'
    assert approval_action_label({'position':3,'before':'P7_Blank'},'drop_column',7,3) == 'Delete column 3 · P7_Blank'
    assert approval_action_label({'changes':[{'row':2,'source_row':7}]}, 'merge_rows',7,1) == 'Merge row 7 into row 2'
    assert approval_action_label({}, 'replace_cells',7,1) == 'Replace R7:C1'


def test_resolution_banner_mentions_verified_origin(monkeypatch):
    catalog, entry = fixture()
    ui = StubUI(); app = _app(monkeypatch, ui)
    ui.session_state.update(cleaning_review_catalog_snapshot=catalog,
        cleaning_approved_rows_by_dataset={'dataset_1':[entry]})
    f = catalog[5]
    resolution = app._prior_approval_closure(f,SRC)
    ui.session_state['cleaning_resolution_gemma_review']={f['id']:{'fingerprint':resolution['fingerprint'],'can_offer':True,'label':'Mark reviewed','reason':'Verified'}}
    monkeypatch.setattr(app,'_render_visual_discussion',lambda *a:None)
    messages=[]
    monkeypatch.setattr(ui,'info',lambda msg:messages.append(msg))
    assert app._render_prior_approval_closure(f['id'],f,SRC)
    assert 'Finding 2 · '+catalog[1]['title'] in messages[0]
    assert 'Delete rows 2, 3, 5, 7, 15, 20' in messages[0]
