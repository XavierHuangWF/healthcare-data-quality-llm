"""A late response is not an implicit authorization to reopen a completed finding."""

from types import SimpleNamespace


def test_completed_finding_ignores_late_agent_c_response(monkeypatch, tmp_path):
    from test_cleaning_review_cards import _app, StubUI
    app = _app(monkeypatch, StubUI())

    class State(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    state = State(cleaning_review_statuses={'finding': 'Applied and verified'},
                  cleaning_messages_by_finding={'finding': []},
                  cleaning_plans_by_finding={}, cleaning_compound_previews_by_finding={},
                  cleaning_control_generation_by_finding={})
    monkeypatch.setattr(app, 'st', SimpleNamespace(session_state=state))
    source = tmp_path / 'x.csv'
    source.write_text('id,val\na,1\n')
    app._record_gemma_b_response('finding', {'id': 'finding'}, source, {
        'reply': 'Another proposal', 'operation': 'replace_cells',
        'dataset_id': 'dataset_1',
        'changes': [{'row': 2, 'column': 2, 'before': '1', 'after': '2'}],
    })
    assert state.cleaning_review_statuses['finding'] == 'Applied and verified'
    assert state.cleaning_plans_by_finding == {}
    assert state.cleaning_messages_by_finding['finding'] == []


def test_unrelated_staged_edit_requires_fresh_validation_not_reply_discard(monkeypatch, tmp_path):
    from test_cleaning_review_cards import _app, StubUI
    from healthcare_data_quality.jobs import interactive_chat_jobs
    from healthcare_data_quality.cleaning.in_memory_cleaning import sha256_bytes

    app = _app(monkeypatch, StubUI())

    class State(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    source = tmp_path / 'x.csv'
    source.write_bytes(b'id,val\na,1\n')
    stage = b'id,val\na,2\n'
    state = State(
        cleaning_review_statuses={'finding': 'Agent C replying'},
        cleaning_messages_by_finding={'finding': []},
        cleaning_plans_by_finding={},
        cleaning_staged_bytes_by_dataset={'dataset_1': stage},
        cleaning_control_generation_by_finding={'finding': 1},
        cleaning_pending_gemma_b={'finding': {
            'job_id': 'pending-job',
            'source_sha256': sha256_bytes(source.read_bytes()),
            'staged_sha256': sha256_bytes(source.read_bytes()),
            'ui_generation': 1, 'assigned_agent': 'C',
        }},
    )
    monkeypatch.setattr(app, 'st', SimpleNamespace(session_state=state))

    class Manager:
        def take(self, _):
            return {'status': 'done', 'result': {'reply': 'Outdated', 'operation': 'none'}}

    monkeypatch.setattr(interactive_chat_jobs, 'interactive_chat_manager', lambda: Manager())

    seen = []
    monkeypatch.setattr(app, '_record_gemma_b_response',
                        lambda *args: seen.append(args[-1]))
    app._sync_gemma_conversations({'id': 'finding', 'dataset_id': 'dataset_1'}, source)
    assert state.cleaning_pending_gemma_b == {}
    assert seen == [{'reply': 'Outdated', 'operation': 'none'}]
    # The reply can be considered for a new independently validated draft,
    # but is never a cleaning authorization or an implicit stage mutation.
    assert state.cleaning_plans_by_finding == {}
    assert state.cleaning_staged_bytes_by_dataset['dataset_1'] == stage


def test_completed_state_rejects_matching_pending_response(monkeypatch, tmp_path):
    from test_cleaning_review_cards import _app, StubUI
    from healthcare_data_quality.jobs import interactive_chat_jobs
    from healthcare_data_quality.cleaning.in_memory_cleaning import sha256_bytes

    app = _app(monkeypatch, StubUI())

    class State(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    source = tmp_path / 'x.csv'
    original = b'id,val\na,1\n'
    source.write_bytes(original)
    state = State(
        cleaning_review_statuses={'finding': 'Applied and verified'},
        cleaning_messages_by_finding={'finding': []},
        cleaning_plans_by_finding={},
        cleaning_staged_bytes_by_dataset={},
        cleaning_control_generation_by_finding={'finding': 1},
        cleaning_pending_gemma_b={'finding': {
            'job_id': 'pending-job',
            'source_sha256': sha256_bytes(original),
            'staged_sha256': sha256_bytes(original),
            'ui_generation': 1, 'assigned_agent': 'C',
        }},
    )
    monkeypatch.setattr(app, 'st', SimpleNamespace(session_state=state))

    class Manager:
        def take(self, _):
            return {'status': 'done', 'result': {'reply': 'Outdated', 'operation': 'none'}}

    monkeypatch.setattr(interactive_chat_jobs, 'interactive_chat_manager', lambda: Manager())

    def should_not_be_called(*_args, **_kwargs):
        raise AssertionError('Completed finding was reopened')

    monkeypatch.setattr(app, '_record_gemma_b_response', should_not_be_called)
    app._sync_gemma_conversations({'id': 'finding', 'dataset_id': 'dataset_1'}, source)
    assert state.cleaning_review_statuses['finding'] == 'Applied and verified'
    assert state.cleaning_pending_gemma_b == {}
    assert state.cleaning_plans_by_finding == {}
