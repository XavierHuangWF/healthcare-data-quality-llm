"""User-reported regressions: real CSV, exact action review, dynamic attribution."""
from pathlib import Path
import csv
from io import StringIO
import pytest
from healthcare_data_quality.cleaning.current_review_evidence import current_finding_evidence, project_current_choices, current_evidence_question
from healthcare_data_quality.cleaning.direct_actions import propose_explicit_column_deletions, draft_format_correction
from healthcare_data_quality.cleaning.finding_cleaning import build_cleaning_findings, validate_scoped_plan
from healthcare_data_quality.cleaning.in_memory_cleaning import _records, sha256_bytes
from healthcare_data_quality.investigation.baseline_audit import run_baseline_audit
from healthcare_data_quality.services.recommendation_options import source_backed_choices
from healthcare_data_quality.services.model_plan_review import option_fingerprint, reviewer_selectable_choices
from healthcare_data_quality.ui.gemma_check_evidence import observed_check_data
from test_cleaning_review_cards import StubUI, _app

SRC=Path(__file__).resolve().parents[1]/'data'/'dirtyL7PVDforXav.csv'

@pytest.fixture(scope='module')
def catalog():
    return build_cleaning_findings({'baseline':run_baseline_audit(SRC)}, [SRC], [SRC.name], [])

def check():
    return {'id':'gemma:1:check','dataset_id':'dataset_1','check_key':'gemma_check','actionable':False,
            'title':'Blank controls Rep_Blank, P7_Blank and P6_Blank should be zero', 'source':'Gemma check',
            'gemma_check_details':{'proposal':{'hypothesis':'Blank controls Rep_Blank, P7_Blank and P6_Blank should be zero'}}}

def test_missing_sample_recovers_all_coordinates(catalog):
    f={**catalog[1], 'evidence':catalog[1]['evidence'][:32]}
    choices=source_backed_choices(f,SRC,{})
    option=next(o for o in choices if o['label']=='Mark 618 missing cells as NA')
    assert len(validate_scoped_plan(option['operations'][0],f,SRC)['changes'])==618
    assert len(option['rows'])==618
    assert any('Delete 6 rows' in o['label'] for o in choices)

def test_blank_deictic_command_scope_and_no_mutation():
    f=check(); raw=SRC.read_bytes(); p=propose_explicit_column_deletions('delete these 3 columns',f,SRC)
    assert p['explicit_columns']==frozenset({2,3,4})
    assert [x['changes'][0]['before'] for x in p['operations']]==['Rep_Blank','P7_Blank','P6_Blank']
    for op in p['operations']:
        assert validate_scoped_plan(op,f,SRC,user_requested_columns=p['explicit_columns'])['operation']=='drop_column'
        with pytest.raises(ValueError): validate_scoped_plan(op,f,SRC)
    assert propose_explicit_column_deletions('delete these 2 columns',f,SRC) is None
    assert propose_explicit_column_deletions('do not delete these 3 columns',f,SRC) is None
    assert SRC.read_bytes()==raw

def test_blank_check_scope_not_all_other_blanks(tmp_path):
    p=tmp_path/'data.csv'; p.write_text('Taxon,Rep_Blank,P7_Blank,P6_Blank,Other_Blank\nA,0,0,0,0\nB,0,0,0,0\n')
    f=check(); choices=source_backed_choices(f,p,{})
    assert {c['position'] for o in choices for op in o['operations'] for c in op['changes']}=={2,3,4}
    assert observed_check_data(f,p,removed_rows={2},removed_columns={3})[1]['Status']=='Removal approved'
    assert all(r['Rows']==1 for r in observed_check_data(f,p,removed_rows={2}))

def test_taxonomy_projection_attributes_real_actions_only(catalog):
    f=catalog[6]; raw=SRC.read_bytes(); rows=_records(raw)
    p=draft_format_correction(f,SRC); edit=next(c for c in p['changes'] if c['row']==6)
    rows[5][0]=edit['after']; b=StringIO(newline=''); csv.writer(b).writerows(rows)
    state={'cleaning_staged_bytes_by_dataset':{'dataset_1':b.getvalue().encode()},
           'cleaning_history':[{'dataset_id':'dataset_1','source_sha256':sha256_bytes(raw),'reviewer':'reviewer','finding_id':'dataset_1:qc:5','operation':'replace_cells','approved_cell_changes':[edit]}],
           'cleaning_approved_rows_by_dataset':{'dataset_1':[{'operation':'drop_rows','source_sha256':sha256_bytes(raw),'reviewer':'reviewer','finding_id':'dataset_1:qc:2','changes':[{'row':7,'before':rows[6][0]}]}]}}
    projection={r['Location']:r for r in current_finding_evidence(state,f,SRC)}
    assert projection['R2:C1']['Status']=='Naming still differs'
    assert projection['R6:C1']['Current']==edit['after']
    assert projection['R6:C1']['Approved by']=='dataset_1:qc:5'
    assert projection['R7:C1']['Status']=='Removal approved'
    assert projection['R7:C1']['Approved by']=='dataset_1:qc:2'
    options=source_backed_choices(f,SRC,{'choices':[{'label':'Standardize','operations':[p]}]})
    original=next(o for o in options if any(c['row']==2 for op in o['operations'] for c in op['changes']))
    narrowed=project_current_choices(state,f,SRC,[original])[0]
    assert [c['row'] for op in narrowed['operations'] for c in op['changes']]==[2]
    assert option_fingerprint(f,SRC,original)!=option_fingerprint(f,SRC,narrowed)
    assert reviewer_selectable_choices(f,SRC,[narrowed],[{'fingerprint':option_fingerprint(f,SRC,original),'can_offer':True}])[0]['gemma_review_status']=='not_assessed'
    assert current_evidence_question('which actions you fix this??')
    assert current_evidence_question('what is current row of 2,6,7 now?')
    assert not current_evidence_question('fix this')

def test_exact_recovery_is_one_async_job_and_preserves_assessment(monkeypatch,catalog):
    f=catalog[1]; ui=StubUI(); app=_app(monkeypatch,ui); state=ui.session_state
    fid=f['id']; options=source_backed_choices(f,SRC,{})
    class Manager:
        submitted=[]
        def submit(self,*args): self.submitted.append(args); return 'job'
        def take(self,job): return {'status':'done','result':proof}
    m=Manager()
    monkeypatch.setattr('healthcare_data_quality.jobs.interactive_chat_jobs.interactive_chat_manager',lambda:m)
    proof=[{'fingerprint':option_fingerprint(f,SRC,o),'can_offer':True,'scope_matches':True,'has_extra_changes':False} for o in options]
    state.cleaning_recommendations={}; state.cleaning_pending_gemma_b={}; state.cleaning_review_statuses={}
    assert app._queue_exact_option_recovery(fid,f,SRC,{'suggestion':'','choices':options},options)
    assert not app._queue_exact_option_recovery(fid,f,SRC,{},options)
    assert len(m.submitted)==1
    app._sync_gemma_conversations(f,SRC)
    assert state.cleaning_recommendations[fid]['gemma_option_reviews']==proof
    assert not app._queue_exact_option_recovery(fid,f,SRC,{},options)
    assert not state.get('cleaning_history')
