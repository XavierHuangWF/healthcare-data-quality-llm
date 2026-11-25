import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from healthcare_data_quality.ui.unendorsed_choice_recovery import review_explanation, reassessment_request

def test_rejected_reason_is_visible():
    reason = review_explanation([{'decision':'uncertain','reason':'Biological evidence missing','can_offer':False}])
    assert 'uncertain' in reason and 'Biological evidence missing' in reason

def test_candidate_does_not_grant_approval():
    request = reassessment_request({'label':'Set 2 cells to NA','operations':[{'operation':'replace_cells'}], 'rows':[{'Before':'0.2','After':'NA'}]})
    assert '0.2' in request and 'NA' in request and 'not authorization to edit' in request

def test_missing_review_is_explicit():
    assert 'No current Gemma assessment' in review_explanation([])
