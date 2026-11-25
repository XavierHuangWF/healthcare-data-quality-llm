"""System-wide order, dependency invalidation, token budget and export checks."""
import io
import json
from hashlib import sha256
import pytest
from healthcare_data_quality.ui.ordered_review import approval_snapshot, revision_suffix, predecessor_blocker
from healthcare_data_quality.cleaning.final_model_verification import review_batches, verify_export_with_gemma
from healthcare_data_quality.services import llm_client
from healthcare_data_quality.services.prompt_budget import compact_data, context_overflow


def catalog():
    keys = ['find_invalid_counts','find_invalid_counts','inspect_taxonomy_structure','inspect_taxonomy_naming','gemma_check']
    return [(f'Finding {i}', {'id':str(i),'dataset_id':'dataset_1','check_key':k,
        'evidence':[{'Column': 2 if i==1 else 5}], 'title': k}) for i,k in enumerate(keys,1)]


def test_later_ready_waits_for_earlier_and_pending_assessment():
    c = catalog(); entries = {str(i):{'status':'ready'} for i in range(1,6)}
    entries['2']['status']='running'
    state = {}
    assert predecessor_blocker(c,'4',entries,state,lambda _:False)=='Finding 2'
    entries['2']['status']='ready'
    state['cleaning_pending_gemma_b']={'2':{'kind':'exact_option_recovery'}}
    assert predecessor_blocker(c,'4',entries,state,lambda _:False)=='Finding 2'
    state.clear()
    assert predecessor_blocker(c,'4',entries,state,lambda _:False) is None


def test_taxonomy_edit_locks_first_affected_and_entire_suffix():
    c=catalog()
    before=approval_snapshot({})
    after=approval_snapshot({'cleaning_history':[{'finding_id':'1','dataset_id':'dataset_1','operation':'replace_cells','approved_cell_changes':[{'row':2,'column':1}]}]})
    suffix=revision_suffix(c,before,after)
    assert suffix=={'3':'Finding 1','4':'Finding 1','5':'Finding 1'}
    assert predecessor_blocker(c,'5',{}, {'cleaning_order_revision_pending':suffix}, lambda f:f in {'1','2'})=='Finding 3'
    assert revision_suffix(c,after,before)==suffix  # revocation also invalidates


def test_structure_edit_locks_all_later_and_visual_only_does_not():
    c=catalog(); before=approval_snapshot({})
    event={'finding_id':'2','operation':'drop_rows','dataset_id':'dataset_1'}
    assert revision_suffix(c,before,approval_snapshot({'cleaning_history':[event]}))=={'3':'Finding 2','4':'Finding 2','5':'Finding 2'}
    assert revision_suffix(c,before,approval_snapshot({'cleaning_history':[{**event,'operation':'flag_cells'}]}))=={}


def test_prompt_budget_preserves_output_room(monkeypatch):
    calls=[]
    def tokenize(request,timeout):
        calls.append(request.full_url)
        return io.BytesIO(json.dumps({'count':10000,'max_model_len':12288}).encode())
    monkeypatch.setattr(llm_client,'urlopen',tokenize)
    assert llm_client._token_budget('x'*16000,4800)==2160
    assert calls[0].endswith('/tokenize')
    monkeypatch.setattr(llm_client,'urlopen',lambda *a,**k:io.BytesIO(b'{"count":12289,"max_model_len":12288}'))
    with pytest.raises(RuntimeError) as exc:
        llm_client._token_budget('x'*16000,2200)
    assert context_overflow(exc.value)
    compact=compact_data({'memory':['x'*9000]*100},chars=1000)
    assert len(json.dumps(compact))<1000
    assert 'omitted' in json.dumps(compact)


def test_final_gemma_checks_actual_diffs_and_structural_ledger(monkeypatch):
    source=b'Taxon,A,B\na,1,9\nb,2,8\n'
    final=b'Taxon,A\na,12\n'
    history=[{'finding_id':'1','reviewer':'R','operation':'replace_cells','approved_cell_changes':[{'row':2,'column':2,'before':'1','after':'12'}]}]
    rows=[{'finding_id':'2','reviewer':'R','operation':'drop_rows','changes':[{'row':3,'before':'b','after':''}]}]
    columns=[{'finding_id':'3','reviewer':'R','position':3}]
    batches=list(review_batches(source,final,history,rows,columns)); items=[i for b in batches for i in b]
    assert items[0]['original_coordinates']==[[2,2]] and items[0]['after']=='12'
    assert items[0]['approved_edits'][0]['finding']=='1'
    assert any(i.get('removed_original_row')==3 and i.get('finding')=='2' for i in items)
    assert any(i.get('removed_original_column')==3 and i.get('finding')=='3' for i in items)
    audit={'source_sha256':sha256(source).hexdigest(),'final_sha256':sha256(final).hexdigest()}
    calls=[]
    def reply(prompt,**kwargs):
        calls.append(prompt)
        return '{"verified":true,"reason":"Matches approved changes."}'
    monkeypatch.setattr(llm_client,'query_local_model',reply)
    assert verify_export_with_gemma(source,final,history,rows,columns,audit)['verified'] is True
    assert 'original_coordinates' in calls[0]
    monkeypatch.setattr(llm_client,'query_local_model',lambda *a,**kw:'{"verified":false,"reason":"Unexplained edit"}')
    with pytest.raises(PermissionError):
        verify_export_with_gemma(source,final,history,rows,columns,audit)
    with pytest.raises(PermissionError):
        verify_export_with_gemma(source,final+b'\n',history,rows,columns,audit)


def test_homogeneous_changes_grouped_without_losing_any_coordinate():
    n=618
    source=('Taxon,A\n'+'a,\n'*n).encode()
    final=('Taxon,A\n'+'a,NA\n'*n).encode()
    history=[{'finding_id':'2','reviewer':'R','approved_cell_changes':[{'row':r,'column':2,'before':'','after':'NA'} for r in range(2,n+2)]}]
    batches=list(review_batches(source,final,history,[],[]))
    assert sum(len(item['original_coordinates']) for b in batches for item in b)==n
    assert len(batches)<5


def test_agent_c_recovers_context_overflow_without_resubmitting_user_turn(monkeypatch):
    from healthcare_data_quality.cleaning import cleaning_workspace as cw
    prompts=[]
    def model(prompt,**kw):
        prompts.append(prompt)
        if len(prompts)==1:
            raise RuntimeError('vLLM HTTP 400: maximum context length is 12288; input_tokens 12289')
        return json.dumps({'reply':'Compare the classifications first.', 'operation':'none',
            'dataset_id':'','changes':[],'clarification_options':[]})
    monkeypatch.setattr(cw,'query_local_model',model)
    text='Compare with other classifications in this dataset.'
    answer=cw.ask_gemma_about_cleaning(text,[],[],[],selected_finding={'finding_id':'f7','dataset_id':'dataset_1','title':'Naming formats','evidence':[]})
    assert answer['operation']=='none' and len(prompts)==2
    assert text in prompts[-1] and 'ONLY SELECTED FINDING' in prompts[-1]
