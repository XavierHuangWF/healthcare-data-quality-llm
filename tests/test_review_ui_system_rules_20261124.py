"""Regression checks of the screenshot cases against the bundled source CSV."""
from pathlib import Path
import pytest
from healthcare_data_quality.investigation.baseline_audit import run_baseline_audit
from healthcare_data_quality.cleaning.finding_cleaning import build_cleaning_findings, validate_scoped_plan
from healthcare_data_quality.services.recommendation_options import source_backed_choices
from healthcare_data_quality.cleaning.numeric_context import proposal_numeric_comparisons
from healthcare_data_quality.cleaning.direct_actions import propose_explicit_taxonomy_padding, draft_format_correction, surviving_taxonomy_proposal
from healthcare_data_quality.cleaning.proposal_conflicts import proposal_conflicts
from healthcare_data_quality.ui.clear_review_actions import final_action_label
from healthcare_data_quality.ui.review_display import partition_review_thread
from healthcare_data_quality.ui.gemma_check_evidence import observed_check_data
SRC = Path(__file__).resolve().parents[1] / 'data' / 'dirtyL7PVDforXav.csv'

@pytest.fixture(scope='module')
def catalog():
    return build_cleaning_findings({'baseline':run_baseline_audit(SRC)}, [SRC], [SRC.name], [])

def test_complete_missing_cells_from_real_csv(catalog):
    finding=catalog[1]
    options=source_backed_choices(finding,SRC,{'choices':[]})
    choice=next(o for o in options if o['label']=='Mark 618 missing cells as NA')
    assert len(choice['rows'])==618
    assert len(validate_scoped_plan(choice['operations'][0],finding,SRC)['changes'])==618
    assert {r['row'] for r in choice['operations'][0]['changes']}=={2,3,5,7,15,20}

def test_negative_preview_has_actual_proposed_numbers_and_column_sd(catalog):
    choice=next(o for o in source_backed_choices(catalog[2],SRC,{'choices':[]}) if 'absolute values' in o['label'])
    rows=proposal_numeric_comparisons(SRC,choice['operations'])
    assert [r['Proposed value'] for r in rows]==['679','4133']
    assert [r['Positive median'] for r in rows]==[413,29]
    assert [r['Positive sample SD'] for r in rows]==pytest.approx([966.897689,1850.335593])
    assert rows[1]['Positive P95']<4133
    assert final_action_label(choice['operations'],catalog[2])=='Use absolute values for 2 negative counts'

def test_literal_pad_to_level_seven_with_flags(catalog):
    p=propose_explicit_taxonomy_padding('pad them to level 7',catalog[4],SRC)
    assert p and [c['row'] for c in p['changes']]==[3,5,15,20]
    assert all(len(c['after'].split(';'))==7 and c['after'].startswith(c['before']) for c in p['changes'])
    assert final_action_label([p],catalog[4])=='Pad 4 taxonomy rows to 7 levels with NA'
    flags=[{'row':c['row'],'column':1,'before':c['before'],'style':'bold_red'} for c in p['changes']]
    assert not proposal_conflicts([p],SRC,SRC.read_bytes(),approved_flags=flags)

def test_naming_correction_keeps_flag_and_skips_deleted_row_before_review(catalog):
    p=draft_format_correction(catalog[6],SRC)
    assert p and {c['row'] for c in p['changes']}=={2,6,7}
    deleted=[{'operation':'drop_rows','changes':[{'row':7}]}]
    p=surviving_taxonomy_proposal(p,catalog[6],SRC,deleted)
    assert {c['row'] for c in p['changes']}=={2,6}
    assert not proposal_conflicts([p],SRC,SRC.read_bytes(),approved_rows=deleted,
                                  approved_flags=[{'row':2,'column':1}])

def test_discussion_keeps_human_messages_and_folds_automatic_diagnostics():
    thread=[{'role':'assistant','content':'Internal background review','background_diagnostic':True},
            {'role':'user','content':'Fix it'}, {'role':'assistant','content':'Here is the answer'},
            {'role':'assistant','content':'Here is the answer'}, {'role':'user','content':'Use NA'}]
    shown, hidden=partition_review_thread(thread)
    assert [t['content'] for t in shown]==['Fix it','Here is the answer','Use NA']
    assert len(hidden)==2 and len(shown)+len(hidden)==len(thread)

def test_control_check_data_not_model_claims():
    rows=observed_check_data({'gemma_check_details':{'proposal':{'hypothesis':'Blank controls Rep_Blank, P7_Blank and P6_Blank should be zero'}}},SRC)
    assert len(rows)==3
    assert all(r['Zero']+r['Nonzero']+r['Missing']+r['Other invalid']==r['Rows']==158 for r in rows)
    assert all(r['Nonzero']==0 for r in rows)
