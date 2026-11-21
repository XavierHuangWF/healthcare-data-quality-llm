"""Combined, one-click read-only exploration controller regression tests."""

from pathlib import Path
import pytest

from healthcare_data_quality.agents import exploratory_combined as combo
from healthcare_data_quality.agents.exploratory_sandbox import SandboxUnavailable


def _env(monkeypatch, *, proposals, completed=None, max_rounds=30, available=True):
    proposals = iter(proposals)
    completed = completed or {
        'proposal': {'hypothesis': 'new source relationship'},
        'result': {'summary': 'checked', 'findings': [{'title': 'novel', 'evidence': []}]},
        'verification': {},
    }
    called = []
    monkeypatch.setattr(combo, 'sandbox_available', lambda: available)
    monkeypatch.setattr(combo, 'MAX_ROUNDS', max_rounds)
    monkeypatch.setattr(combo, 'known_issue_catalog', lambda paths, audit: {})
    def propose(datasets, audit, previous, *, allow_stop=True):
        called.append(('propose', allow_stop, len(previous)))
        return next(proposals)
    monkeypatch.setattr(combo, 'propose_exploration', propose)
    monkeypatch.setattr(combo, 'validate_generated_code', lambda code: None)
    def execute(proposal, datasets, initial_failure=None):
        called.append(('execute', proposal['hypothesis']))
        return {**completed, 'proposal': proposal, 'verification': {}, 'result': {**completed['result']}}
    monkeypatch.setattr(combo, 'execute_with_auto_repair', execute)
    monkeypatch.setattr(combo, 'filter_redundant_findings', lambda result, catalog, runs: (result, 0))
    return called


def _candidate(name):
    return {'hypothesis': name, 'code': 'def investigate(datasets): pass'}


def test_combined_run_explores_then_stops_without_fixed_target(monkeypatch):
    calls = _env(monkeypatch, proposals=[_candidate('new check'),
                   {'decision': 'stop', 'reason': 'No further hypothesis', 'remaining_questions': []}])
    saved = []
    report = combo.run_exploratory_series({'dataset_1': Path('test.csv')}, {},
                                         on_round=lambda rounds: saved.append(len(rounds)))
    assert len(report['runs']) == 1
    assert report['stop']['decision'] == 'stop'
    assert report['safety_ceiling_reached'] is False
    assert saved == [1]
    assert calls == [('propose', True, 0), ('execute', 'new check'), ('propose', True, 1)]


def test_sandbox_is_required_before_proposal_or_execution(monkeypatch):
    calls = _env(monkeypatch, proposals=[_candidate('new check')], available=False)
    with pytest.raises(SandboxUnavailable):
        combo.run_exploratory_series({'dataset_1': Path('test.csv')}, {})
    assert calls == []


def test_verified_checkpoints_survive_later_model_failure(monkeypatch):
    calls = _env(monkeypatch, proposals=[_candidate('new check')])
    def next_proposal(datasets, audit, previous, allow_stop=True):
        if previous:
            raise RuntimeError('model temporarily unreachable')
        return _candidate('new check')
    monkeypatch.setattr(combo, 'propose_exploration', next_proposal)
    verified = []
    with pytest.raises(RuntimeError, match='unreachable'):
        combo.run_exploratory_series({'dataset_1': Path('test.csv')}, {},
                                     on_round=lambda rounds: verified.extend(rounds))
    assert len(verified) == 1
    assert ('execute', 'new check') in calls


def test_emergency_ceiling_not_a_round_target(monkeypatch):
    _env(monkeypatch, proposals=[_candidate('new check')], max_rounds=1)
    result = combo.run_exploratory_series({'dataset_1': Path('test.csv')}, {})
    assert result['safety_ceiling_reached'] is True
    assert len(result['runs']) == 1
    assert result['stop'] is None


def test_override_first_prior_stop_then_follow_normal_stopping(monkeypatch):
    calls = _env(monkeypatch, proposals=[_candidate('another check'),
                   {'decision': 'stop', 'reason': 'Done', 'remaining_questions': []}])
    prior = [{'proposal': {'hypothesis': 'first'}, 'result': {'findings': []}}]
    result = combo.run_exploratory_series({'dataset_1': Path('test.csv')}, {},
                                         previous=prior, force_first=True)
    assert len(result['runs']) == 2
    assert calls[0] == ('propose', False, 1)
    assert calls[-1] == ('propose', True, 2)


def test_filter_applies_before_showing_verified_round(monkeypatch):
    _env(monkeypatch, proposals=[_candidate('novel'),
                   {'decision': 'stop', 'reason': 'Done', 'remaining_questions': []}])
    monkeypatch.setattr(combo, 'filter_redundant_findings',
                        lambda result, catalog, runs: ({**result, 'findings': []}, 1))
    result = combo.run_exploratory_series({'dataset_1': Path('test.csv')}, {})
    assert result['runs'][0]['result']['findings'] == []
    assert result['runs'][0]['verification']['already_reported_findings_suppressed'] == 1
