"""Independent end-to-end replay of the precise approved actions in screenshots.

These are synthetic approvals for exercising the exporter, not a claim that the
user's real application has applied them or a replacement for the real ledger.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
from openpyxl import load_workbook

from healthcare_data_quality.cleaning.direct_actions import draft_format_correction
from healthcare_data_quality.cleaning.review_reconciliation import independently_reconcile
from healthcare_data_quality.cleaning.reviewer_audit_workbook import audit_tables, embed_audit_sheets, verify_audit_sheets
from healthcare_data_quality.cleaning.styled_export import build_styled_excel_download
from healthcare_data_quality.cleaning.final_export_integrity import verify_final_excel
from healthcare_data_quality.cleaning.review_zip_bundle import build_review_zip_bundle

SRC = Path(__file__).resolve().parents[1]/'data'/'dirtyL7PVDforXav.csv'


def replay_fixture():
    original = SRC.read_bytes()
    raw = list(csv.reader(io.StringIO(original.decode('utf-8-sig'), newline='')))
    assert len(raw) == 159 and len(raw[0]) == 106
    assert hashlib.sha256(original).hexdigest().startswith('2eb51b35')
    who = 'Example Reviewer'
    history=[]; row_ledger=[]; col_ledger=[]
    names={f'f{i}': f'Finding {i} · ' + {
        1:'fractional counts',2:'missing counts',3:'negative counts',
        4:'duplicate taxonomy rows',5:'taxonomy ranks',6:'separator',
        7:'taxonomy naming',8:'duplicate column',9:'blank header',
        10:'blank controls'}.get(i, '') for i in range(1,11)}
    statuses={fid:'Applied and verified' for fid in names}
    statuses['f6']=statuses['f9']='Reviewed — no change'
    for fid, ids in [('f2',(2,3,5,7,15,20)), ('f3',(16,24)), ('f4',(12,))]:
        row_ledger.append({'finding_id':fid, 'operation':'drop_rows','reviewer':who,
                           'changes':[{'row':i,'before':raw[i-1][0],'after':''} for i in ids]})
        history.append({'dataset_id':'dataset_1','finding_id':fid,'reviewer':who,
                        'operation':'drop_rows','modified_positions':[{'row':i} for i in ids],
                        'approval_group':f'approved-{fid}'})
    for fid, ids in [('f8',(106,)),('f10',(2,3,4))]:
        for col in ids:
            col_ledger.append({'finding_id':fid,'position':col,'before':raw[0][col-1],
                               'reviewer':who})
        history.append({'dataset_id':'dataset_1','finding_id':fid,'reviewer':who,
                        'operation':'drop_column','modified_positions':[{'column':i} for i in ids],
                        'approval_group':f'approved-{fid}'})
    from healthcare_data_quality.cleaning.direct_actions import taxonomy_convention_evidence
    suggested=draft_format_correction({
        'dataset_id':'dataset_1','check_key':'inspect_taxonomy_naming',
        'evidence':[{'CSV row':6,'Original classification (verbatim)':raw[5][0]}]}, SRC)
    assert suggested and suggested['changes'][0]['row']==6
    edit=suggested['changes'][0]
    history.append({'dataset_id':'dataset_1','finding_id':'f7','reviewer':who,
                    'operation':'replace_cells','approved_cell_changes':[edit],
                    'approval_group':'approved-f7'})
    flags=[]
    for fid,row in [('f1',11),('f5',3)]:
        flag={'row':row,'column':1,'before':raw[row-1][0],'style':'bold_red'}
        flags.append(flag)
        history.append({'dataset_id':'dataset_1','finding_id':fid,'reviewer':who,
                        'operation':'flag_cells','flags':[flag],
                        'approval_group':f'approved-{fid}'})
    deleted_rows={2,3,5,7,12,15,16,20,24}
    deleted_cols={2,3,4,106}
    final_rows=[]
    for row_i, fields in enumerate(raw,1):
        if row_i in deleted_rows: continue
        new=list(fields)
        if row_i==6: new[0]=edit['after']
        final_rows.append([v for c,v in enumerate(new,1) if c not in deleted_cols])
    out=io.StringIO(newline='')
    csv.writer(out,lineterminator='\r\n').writerows(final_rows)
    final=out.getvalue().encode('utf-8-sig')
    return original, final, history, row_ledger, col_ledger, flags, names, statuses


def test_replay_raw_to_final_every_surviving_cell_and_superseded_flags():
    original, final, history, row_ledger, col_ledger, flags, names, statuses=replay_fixture()
    result=independently_reconcile(original_csv=original,final_csv=final,history=history,
        review_statuses=statuses,findings=names,approved_rows=row_ledger,approved_columns=col_ledger)
    assert result['original_rows']==158
    assert result['final_rows']==149 and result['final_columns']==102
    assert result['deleted_rows']==9 and result['deleted_columns']==4
    assert result['surviving_changed_cells']==1
    assert result['surviving_unchanged_cells']==150*102-1
    assert any('R6:C1'==d[3] for d in result['details'])
    # Source row 11 maps to final row 7, but row 3 was deleted and must not
    # transfer its highlight to any unrelated surviving taxonomy.
    raw=list(csv.reader(io.StringIO(final.decode('utf-8-sig'))))
    assert raw[6][0]==list(csv.reader(io.StringIO(original.decode('utf-8-sig'))))[10][0]
    assert SRC.read_bytes()==original
    # Tampering with one survivor, even if the source location had a prior
    # authorized edit, must never be treated as an approved outcome.
    row4=raw[3][0]; raw[3][0]='MALICIOUS CHANGE'
    output=io.StringIO(newline='');csv.writer(output).writerows(raw)
    with pytest.raises(PermissionError,match='Unapproved CSV value change'):
        independently_reconcile(original_csv=original,final_csv=output.getvalue().encode(),
            history=history,review_statuses=statuses,findings=names,
            approved_rows=row_ledger,approved_columns=col_ledger)


def test_real_full_audit_excel_and_zip_have_human_traceable_tabs():
    original, final, history, row_ledger, col_ledger, flags, names, statuses=replay_fixture()
    discussions={'f7':[{'role':'user','content':'Normalize row six naming and separator'},
                         {'role':'assistant','content':'Verified preview for row six'}]}
    workbook_name, base_xlsx=build_styled_excel_download(final,original_csv=original,
        original_filename='dirtyL7PVDforXav.csv',approved_flags=flags,
        approved_row_operations=row_ledger,approved_column_deletions=col_ledger)
    tables=audit_tables(original_csv=original,final_csv=final,history=history,
        review_statuses=statuses,findings=names,approved_rows=row_ledger,
        approved_columns=col_ledger,discussions=discussions,
        reviewer='Example Reviewer',verified={'pass':True})
    xlsx=embed_audit_sheets(base_xlsx,tables)
    verify_audit_sheets(xlsx,tables)
    verify_final_excel(xlsx,final,original_csv=original,approved_flags=flags,
        approved_rows=row_ledger,approved_columns=col_ledger)
    wb=load_workbook(io.BytesIO(xlsx))
    assert wb['Cleaned dataset'].cell(7,1).font.bold
    assert wb['Cleaned dataset'].cell(7,1).font.color.rgb=='FFDC2626'
    assert wb['Cleaned dataset'].cell(7,1).fill.fgColor.rgb=='FFFFE699'
    assert wb['Cleaned dataset'].cell(3,1).font.bold is not True
    assert len([r for r in wb['Review overview'].values])==11
    assert wb['Interaction log'].cell(2,5).value=='1'
    v=list(wb['Verification'].values)
    assert any('SOURCE-TO-FINAL DETAILS' in str(row[0]) for row in v)
    assert any('R6:C1'==str(row[3]) and 'Mobiluncus' in str(row[5]) for row in v)
    assert any('REVIEW DECISIONS' in str(row[0]) for row in v)
    assert any('Reviewed — no change' in str(row[1]) for row in v)
    assert any('approved' in str(row[10]).lower() for row in list(wb['Source-to-final checks'].values)[1:])
    wb.close()
    archive_name, archive=build_review_zip_bundle(original_csv=original,cleaned_csv=final,
        original_filename='dirtyL7PVDforXav.csv',reviewer='Example Reviewer',
        history=history,decisions=[],review_statuses=statuses,conversations=discussions,
        approved_row_operations=row_ledger,approved_column_deletions=col_ledger,
        completed_audits={},finding_titles=names,excel_attachment=(workbook_name,xlsx))
    assert archive_name.endswith('.zip')
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        names_in_zip=z.namelist()
        assert 'cleaned/cleaned_highlighted.xlsx' in names_in_zip
        assert 'audit/reviewer_interactions.csv' in names_in_zip
        assert 'audit/approved_changes.csv' in names_in_zip
        assert 'audit/finding_decisions.csv' in names_in_zip
        manifest=json.loads(z.read('manifest.json'))
        assert manifest['finding_count']==10
        for filename, info in manifest['files'].items():
            assert hashlib.sha256(z.read(filename)).hexdigest()==info['sha256']
        assert z.read('cleaned/cleaned_dirtyL7PVDforXav.csv')==final
        assert z.read('source/original_dirtyL7PVDforXav.csv')==original
