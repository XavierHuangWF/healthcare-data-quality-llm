"""Review workers must not block widgets or publish stale approvals."""
from copy import deepcopy
from pathlib import Path
from threading import Event
from time import monotonic, sleep
import pytest
from test_cleaning_review_cards import StubUI, _app
from healthcare_data_quality.jobs.interactive_chat_jobs import InteractiveChatJobs
from healthcare_data_quality.ui.one_click_choice import prepare_verified_choice_snapshot, commit_verified_choice_snapshot
from healthcare_data_quality.ui.review_refresh import checkpoint_identity


def take_when_done(manager, job):
    deadline = monotonic() + 3
    while monotonic() < deadline:
        result = manager.take(job)
        if result['status'] != 'running':
            return result
        sleep(.01)
    pytest.fail('Worker never delivered its result')


def test_local_work_and_list_assessments_have_independent_lanes():
    manager = InteractiveChatJobs()
    entered, release = Event(), Event()
    def slow():
        entered.set()
        assert release.wait(4)
        return {'verified': True}
    job = manager.submit_local(slow)
    try:
        assert entered.wait(1)
        assert manager.take(job)['status'] == 'running'
        assert manager.active_count('C') == 0
        inference = manager.submit(lambda: [{'can_offer': True}])
        assert take_when_done(manager, inference)['result'] == [{'can_offer': True}]
    finally:
        release.set()
    assert take_when_done(manager, job)['result'] == {'verified': True}
    assert manager.take(job)['status'] == 'lost'


@pytest.mark.parametrize('stale', [False, True])
def test_explicit_review_is_queued_and_revision_bound(monkeypatch, tmp_path, stale):
    ui = StubUI(); app = _app(monkeypatch, ui); state = ui.session_state
    source = tmp_path / 'data.csv'; source.write_text('Taxon,Count\na,0.2\n')
    f = {'id': 'f', 'dataset_id': 'dataset_1', 'title': 'Fractional counts'}
    state.update(cleaning_pending_gemma_b={}, cleaning_review_statuses={}, cleaning_history=[],
                 cleaning_messages_by_finding={'f':[{'role':'user','content':'Make it 2'}]},
                 cleaning_discussion_mode_by_finding={'f':'alternative'})
    class Manager:
        calls = []
        def submit(self, *args):
            self.calls.append(args); return 'job'
        def take(self, job):
            return {'status':'done','result':{'can_offer':True}}
    manager = Manager()
    monkeypatch.setattr('healthcare_data_quality.jobs.interactive_chat_jobs.interactive_chat_manager', lambda:manager)
    monkeypatch.setattr(app, '_explicit_review_worker', lambda *args: pytest.fail('UI called model worker synchronously'))
    seen = []
    monkeypatch.setattr(app, '_record_gemma_b_response', lambda *args: seen.append(args))
    proposal = {'operation':'replace_cells','changes':[{'row':2,'column':2,'before':'0.2','after':'2'}]}
    app._queue_explicit_review(f, source, 'Make it 2', proposal)
    assert len(manager.calls) == 1
    assert state.cleaning_review_statuses['f'] == 'Review queued'
    assert not seen
    if stale:
        state.cleaning_staged_bytes_by_dataset = {'dataset_1': b'Taxon,Count\na,3\n'}
    app._sync_gemma_conversations(f, source)
    assert not state.cleaning_pending_gemma_b
    assert bool(seen) is not stale
    assert not state.cleaning_history


def test_worker_prepares_without_mutation_and_commit_checks_current_state(monkeypatch, tmp_path):
    source = tmp_path/'data.csv'; source.write_text('Taxon,Count\na,0.2\nb,3\n')
    f = {'id':'f','dataset_id':'dataset_1','source':'QC','title':'Fractional count',
         'check_key':'find_invalid_counts','actionable':True,'evidence':[{'CSV row':2,'Column':2,'Original value':'0.2'}]}
    option = {'label':'Delete row 2','operations':[{'operation':'drop_rows','dataset_id':'dataset_1','changes':[{'row':2,'before':'a','after':''}]}]}
    state = {'cleaning_approved_reviewer':'Reviewer','cleaning_history':[]}
    original = deepcopy(state); raw = source.read_bytes()
    monkeypatch.setattr('healthcare_data_quality.learning.interaction_memory.save_reviewed_solution', lambda *a,**k:None)
    prepared = prepare_verified_choice_snapshot(state, f, source, option)
    assert state == original and source.read_bytes() == raw
    stale = deepcopy(state); stale['cleaning_staged_bytes_by_dataset']={'dataset_1':b'Taxon,Count\na,4\nb,3\n'}
    with pytest.raises(PermissionError): commit_verified_choice_snapshot(stale,f,source,option,prepared)
    commit_verified_choice_snapshot(state,f,source,option,prepared)
    assert state['cleaning_history'][0]['finding_id'] == 'f'
    assert source.read_bytes() == raw
    with pytest.raises(PermissionError): commit_verified_choice_snapshot(state,f,source,option,prepared)


def test_checkpoint_not_rewritten_for_pending_status(tmp_path):
    source = tmp_path/'data.csv'; source.write_text('Taxon,Count\na,0\n')
    state = {'cleaning_review_statuses':{'f':'Awaiting review'}}
    key = checkpoint_identity(state,[source],{'f':'Finding'})
    state['cleaning_review_statuses']['f']='Review queued'
    assert checkpoint_identity(state,[source],{'f':'Finding'}) == key
    state['cleaning_review_statuses']['f']='Reviewed — no change'
    assert checkpoint_identity(state,[source],{'f':'Finding'}) != key
