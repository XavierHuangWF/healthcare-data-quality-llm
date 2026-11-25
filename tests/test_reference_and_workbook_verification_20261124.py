import csv
import hashlib
import io
import json
from pathlib import Path
import pytest
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
from healthcare_data_quality.cleaning.proposal_reference import reference_evidence, taxonomy_action_label
from healthcare_data_quality.cleaning.styled_export import build_styled_excel_download
from healthcare_data_quality.cleaning.reviewer_audit_workbook import audit_tables, embed_audit_sheets, verify_audit_sheets
from healthcare_data_quality.cleaning.final_export_integrity import verified_workbook_evidence
from healthcare_data_quality.cleaning.final_model_verification import verify_export_with_gemma, review_batches


def _csv(rows):
    buf=io.StringIO(newline=''); csv.writer(buf).writerows(rows); return buf.getvalue().encode()


def _scenario():
    raw=(Path(__file__).resolve().parents[1]/'data/dirtyL7PVDforXav.csv').read_bytes()
    records=list(csv.reader(io.StringIO(raw.decode('utf-8-sig'))))
    rows=[2,3,5,7,11,12,15,20]; cols=[2,3,4,6,12,105]
    flags=[{'row':11,'column':c,'before':v,'style':'bold_red','fill':True} for c,v in enumerate(records[10],1)]
    base={'dataset_id':'dataset_1','reviewer':'Reviewer','source_sha256':hashlib.sha256(raw).hexdigest()}
    history=[{**base,'finding_id':'f1','operation':'flag_cells','flags':flags}]
    approved_rows=[]; approved_cols=[]
    for fid,positions in [('f2',[2,3,5,7,15,20]),('f4',[11,12])]:
        approved_rows.append({**base,'finding_id':fid,'operation':'drop_rows','changes':[{'row':r,'before':records[r-1][0],'after':''} for r in positions]})
        history.append({**base,'finding_id':fid,'operation':'drop_rows','modified_positions':[{'row':r} for r in positions]})
    for fid,positions in [('f3',[6,12]),('f8',[105]),('g1',[2,3,4])]:
        approved_cols.extend({**base,'finding_id':fid,'position':c} for c in positions)
        history.append({**base,'finding_id':fid,'operation':'drop_column','modified_positions':[{'column':c} for c in positions]})
    correction=records[5][0].removeprefix('k__')
    history.extend([{**base,'finding_id':'f7','operation':'replace_cells','approved_cell_changes':[{'row':6,'column':1,'before':records[5][0],'after':correction}]},
                    {**base,'finding_id':'f9','operation':'rename_headers','approved_cell_changes':[{'row':1,'column':1,'before':'','after':'Taxonomy'}]}])
    final=[[v for c,v in enumerate(row,1) if c not in cols] for r,row in enumerate(records,1) if r not in rows]
    final[0][0]='Taxonomy'; final[2][0]=correction; final=_csv(final)
    statuses={e['finding_id']:'Applied and verified' for e in history}; statuses.update(f5='Resolved by earlier approved edit',f6='Resolved by earlier approved edit')
    findings={f'f{i}':f'Finding {i}' for i in range(1,10)}; findings['g1']='Gemma check 1'
    kwargs=dict(original_csv=raw,final_csv=final,history=history,review_statuses=statuses,findings=findings,approved_rows=approved_rows,approved_columns=approved_cols,reviewer='Reviewer',verified={})
    name,excel=build_styled_excel_download(final,original_csv=raw,original_filename='raw.csv',approved_flags=flags,approved_row_operations=approved_rows,approved_column_deletions=approved_cols)
    return raw,final,flags,kwargs,excel


def _proof(raw,final,flags,kwargs,excel,tables):
    return verified_workbook_evidence(excel,final,original_csv=raw,approved_flags=flags,approved_rows=kwargs['approved_rows'],approved_columns=kwargs['approved_columns'],audit_tables=tables)


def test_real_csv_superseded_highlights_and_resolved_audit():
    raw,final,flags,kw,base=_scenario()
    tables=audit_tables(**kw)
    assert all(row[6]=='Resolved by earlier approvals; no new edits' for row in tables['Review overview'] if row[0] in ('Finding 5','Finding 6'))
    details=[r for r in tables['Source-to-final checks'] if r[2]=='flag_cells']
    assert len(details)==106 and all(r[9]=='Superseded by approved deletion' and 'Finding 4' in r[10] for r in details)
    data=embed_audit_sheets(base,tables); proof=_proof(raw,final,flags,kw,data,tables)
    assert proof['cells_verified']==15100 and all(a['final'] is None for a in proof['annotations'])
    assert len(proof['audit_annotations'])==212
    assert all(a['observed_style']=={'bold':True,'red':True,'yellow':True} for a in proof['audit_annotations'])
    # The final proof must be stable when final model metadata is attached.
    kw['verified']={'gemma_final_review':{'verified':True,'batches':3,'final_sha256':hashlib.sha256(final).hexdigest(),'workbook_evidence_sha256':proof['workbook_evidence_sha256']}}
    after=audit_tables(**kw); delivered=embed_audit_sheets(base,after)
    assert _proof(raw,final,flags,kw,delivered,after)['workbook_evidence_sha256']==proof['workbook_evidence_sha256']


@pytest.mark.parametrize('tamper',['data_value','data_style','audit_style','audit_value'])
def test_actual_workbook_tampering_blocks_export(tamper):
    raw,final,flags,kw,base=_scenario(); tables=audit_tables(**kw)
    data=embed_audit_sheets(base,tables); wb=load_workbook(io.BytesIO(data)); sheet=wb['Cleaned dataset']
    if tamper=='data_value': sheet['B2']=99999999
    elif tamper=='data_style': sheet['B2'].font=Font(bold=True,color='FFDC2626')
    elif tamper=='audit_style': wb['Audit trail']['E2'].font=Font()
    else: wb['Audit trail']['E2']='invented source value'
    b=io.BytesIO(); wb.save(b)
    with pytest.raises(PermissionError): _proof(raw,final,flags,kw,b.getvalue(),tables)


def test_gemma_receives_observed_styles_and_rejects_discrepancy(monkeypatch):
    raw,final,flags,kw,base=_scenario();tables=audit_tables(**kw);data=embed_audit_sheets(base,tables);proof=_proof(raw,final,flags,kw,data,tables)
    audit={'source_sha256':hashlib.sha256(raw).hexdigest(),'final_sha256':hashlib.sha256(final).hexdigest()}
    calls=[]
    from healthcare_data_quality.services import llm_client
    def model(prompt,**kwargs):
        calls.append(prompt); return '{"verified":true,"reason":"Agrees with approval evidence"}'
    monkeypatch.setattr(llm_client,'query_local_model',model)
    result=verify_export_with_gemma(raw,final,kw['history'],kw['approved_rows'],kw['approved_columns'],audit,proof)
    assert result['workbook_evidence_sha256']==proof['workbook_evidence_sha256']
    assert any('workbook_audit_annotation' in p and 'observed_style' in p for p in calls)
    monkeypatch.setattr(llm_client,'query_local_model',lambda *a,**k:'{"verified":false,"reason":"Mismatch"}')
    with pytest.raises(PermissionError): verify_export_with_gemma(raw,final,kw['history'],kw['approved_rows'],kw['approved_columns'],audit,proof)


def test_current_taxonomy_references_and_distinct_labels():
    raw,final,flags,kw,base=_scenario()
    original=list(csv.reader(io.StringIO(raw.decode('utf-8-sig'))))
    removed={2,3,5,7,11,12,15,20}; rows=[r for r in range(1,len(original)+1) if r not in removed]
    projected=([original[r-1] for r in rows],rows,list(range(1,len(original[0])+1)))
    before=original[5][0]; after=before.removeprefix('k__'); plan=[{'operation':'replace_cells','changes':[{'row':6,'column':1,'before':before,'after':after}]}]
    summary,examples=reference_evidence(projected,plan)
    assert summary[0]['Reference']=='C1 · Unprefixed'
    assert examples and all(not any(e['Source'].startswith(f'R{r}:') for r in removed|{6}) for e in examples)
    label=taxonomy_action_label('Standardize naming for 1 taxonomy row',plan)
    assert 'Unprefixed' in label and 'with spaces' in label
    other=[{'operation':'replace_cells','changes':[{'row':6,'column':1,'before':before,'after':after.replace('; ',';')}]}]
    assert taxonomy_action_label('Standardize naming for 1 taxonomy row',other)!=label


def test_surviving_font_only_is_not_yellow():
    raw=b'Taxon,A\na,1\nb,2\n'; flags=[{'row':2,'column':2,'before':'1','style':'bold_red','fill':False}]
    event={'dataset_id':'dataset_1','finding_id':'f1','operation':'flag_cells','reviewer':'R','flags':flags}
    kw=dict(original_csv=raw,final_csv=raw,history=[event],review_statuses={'f1':'Applied and verified'},findings={'f1':'Finding 1'},approved_rows=[],approved_columns=[],reviewer='R')
    _,base=build_styled_excel_download(raw,original_csv=raw,original_filename='a.csv',approved_flags=flags,approved_row_operations=[],approved_column_deletions=[])
    tables=audit_tables(**kw); data=embed_audit_sheets(base,tables); proof=_proof(raw,raw,flags,kw,data,tables)
    assert proof['annotations'][0]['observed_style']=={'bold':True,'red':True,'yellow':False}
    assert all(x['observed_style']['yellow'] is False for x in proof['audit_annotations'])


def test_prepare_and_seal_actual_artifact_worker():
    from healthcare_data_quality.cleaning.final_export_integrity import prepare_verified_workbook, seal_verified_workbook
    raw,final,flags,kw,_=_scenario()
    candidate=prepare_verified_workbook(filename='source.csv',approved_flags=flags,audit_arguments=kw)
    digest=candidate['proof']['workbook_evidence_sha256']
    kw['verified']={'gemma_final_review':{'verified':True,'batches':1,'source_sha256':hashlib.sha256(raw).hexdigest(),'final_sha256':hashlib.sha256(final).hexdigest(),'workbook_evidence_sha256':digest}}
    result=seal_verified_workbook(base=candidate['base'],approved_flags=flags,audit_arguments=kw,reviewed_proof=digest)
    assert result['artifact_sha256']==hashlib.sha256(result['verified_excel']).hexdigest()
    with pytest.raises(PermissionError): seal_verified_workbook(base=candidate['base'],approved_flags=flags,audit_arguments=kw,reviewed_proof='wrong')
