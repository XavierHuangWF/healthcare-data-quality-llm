"""Recommendation evidence belongs to an actionable proposal, not a finding."""
from contextlib import contextmanager
from pathlib import Path
import csv

import pytest
from test_cleaning_review_cards import StubUI, _app
from healthcare_data_quality.cleaning.proposal_reference import reference_evidence
from healthcare_data_quality.cleaning.numeric_context import proposal_numeric_comparisons

SRC = Path(__file__).resolve().parents[1] / 'data' / 'dirtyL7PVDforXav.csv'


def numeric_plan():
    return {'operation': 'replace_cells', 'changes': [
        {'row': 11, 'column': 7, 'before': '0.2', 'after': '2'},
        {'row': 11, 'column': 11, 'before': '0.5', 'after': '5'}]}


def test_numeric_proposal_has_one_peer_table_on_actual_csv():
    with SRC.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.reader(stream))
    projected = (rows, list(range(1, len(rows)+1)), list(range(1, len(rows[0])+1)))
    plans = [numeric_plan()]
    assert reference_evidence(projected, plans) == ([], [])
    stats = proposal_numeric_comparisons(SRC, plans, projected=projected)
    assert [r['Proposed value'] for r in stats] == ['2', '5']
    assert all('Positive median' in r and 'Positive sample SD' in r for r in stats)
    # Compound deletion + correction retains only the correction comparison.
    summary, examples = reference_evidence(projected, plans + [{'operation':'drop_rows','changes':[{'row':2}]}])
    assert not summary
    assert not examples


class CardUI(StubUI):
    def __init__(self):
        super().__init__(); self.stack=[]; self.events=[]; self.n=0
    @contextmanager
    def container(self, *args, **kwargs):
        self.n += 1; self.stack.append(self.n)
        try: yield self
        finally: self.stack.pop()
    def dataframe(self, data, **kwargs):
        self.events.append((tuple(self.stack), 'table', list(data.columns)))
    def button(self, label, *, key=None, **kwargs):
        self.events.append((tuple(self.stack), 'button', label)); return False


@pytest.mark.parametrize('phase,discussing', [('ready',False), ('preparing',False), ('ready',True)])
def test_initial_proposal_renderer_keeps_preview_stats_and_button_together(monkeypatch, phase, discussing):
    ui = CardUI(); app = _app(monkeypatch,ui)
    fid = 'dataset_1:qc:1'
    f = {'id':fid,'dataset_id':'dataset_1','title':'Fractional counts','check_key':'find_invalid_counts'}
    ui.session_state.update(cleaning_approved_reviewer='Reviewer', cleaning_sequential_active=fid,
        cleaning_review_catalog_snapshot=[f], cleaning_discussion_mode_by_finding={fid:'alternative'} if discussing else {})
    # Use real reference rendering, but no live model requests.
    choices = [{'label':'Correct fractional values','operations':[numeric_plan()],
        'rows':[{'Location':'R11:C7','Before':'0.2','After':'2'}, {'Location':'R11:C11','Before':'0.5','After':'5'}]}]
    app._render_live_agent_b_finding(fid,f,SRC,[],[],None,'dataset_1',prepared=(phase,{'suggestion':'Correction'},choices))
    if phase != 'ready' or discussing:
        assert ui.events == []
    else:
        tables=[e for e in ui.events if e[1]=='table']; buttons=[e for e in ui.events if e[1]=='button']
        assert len(tables)==2
        assert tables[0][2]==['Location','Before','After']
        assert 'Positive sample SD' in tables[1][2]
        assert tables[0][0] == tables[1][0] == buttons[0][0] != ()
        assert buttons[0][2]=='Apply 2 verified cell changes · 0.2 → 2, 0.5 → 5'
        assert buttons[-1][0]==() and 'Discuss' in buttons[-1][2]


def deletion_plans():
    return [
        {'operation': 'drop_rows', 'changes': [
            {'row': 16, 'before': 'Gardnerella; vaginalis'},
            {'row': 24, 'before': 'Atopobium; vaginae'}]},
        {'operation': 'drop_column', 'changes': [
            {'position': 6, 'before': 'AbC10'}, {'position': 12, 'before': 'ABC105'}]},
    ]


@pytest.mark.parametrize('kind', ['row', 'column', 'combined'])
@pytest.mark.parametrize('fid, check_key', [('dataset_1:qc:3', 'find_invalid_counts'), ('gemma:1:check', 'gemma_check')])
def test_deletion_has_preview_and_no_peer_reference(monkeypatch, kind, fid, check_key):
    from healthcare_data_quality.cleaning import sequential_context
    from healthcare_data_quality.cleaning.reviewed_rule_replay import current_change_rows
    plans = deletion_plans()
    plans = plans[:1] if kind == 'row' else plans[1:] if kind == 'column' else plans
    with SRC.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.reader(stream))
    projected = (rows, list(range(1, len(rows)+1)), list(range(1, len(rows[0])+1)))
    assert reference_evidence(projected, plans) == ([], [])
    ui = CardUI(); app = _app(monkeypatch,ui)
    f = {'id':fid,'dataset_id':'dataset_1','title':'Deletion review','check_key':check_key}
    ui.session_state.update(cleaning_approved_reviewer='Reviewer', cleaning_sequential_active=fid,
        cleaning_review_catalog_snapshot=[f])
    def unexpected(*args, **kwargs):
        pytest.fail('Deletion should not project the dataset for peer references')
    monkeypatch.setattr(sequential_context, 'projected_records', unexpected)
    # Agent B initial cards retain the action with its preview.
    choices = [{'label':'Delete affected targets','operations':plans,
        'rows':[{'Location': r['Target'], 'Before': r['Before'], 'After': r['After']}
                for r in current_change_rows(plans)]}]
    app._render_live_agent_b_finding(fid,f,SRC,[],[],None,'dataset_1',prepared=('ready',{'suggestion':'Deletion'},choices))
    tables = [e for e in ui.events if e[1]=='table']
    buttons = [e for e in ui.events if e[1]=='button']
    assert len(tables) == 1 and tables[0][2] == ['Location','Before','After']
    assert tables[0][0] == buttons[0][0] != ()
    # Agent C shared preview (single and compound) must behave identically.
    ui.events.clear()
    app._render_before_after(plans, SRC)
    assert [e[2] for e in ui.events if e[1]=='table'] == [['Location','Before','After']]


def test_deletion_keeps_approved_annotation_impact(monkeypatch):
    import hashlib
    ui = CardUI(); app = _app(monkeypatch,ui)
    fid = 'dataset_1:qc:3'
    ui.session_state.update(cleaning_sequential_active=fid,
        cleaning_review_catalog_snapshot=[{'id':fid,'dataset_id':'dataset_1'}],
        cleaning_history=[{'operation':'flag_cells','dataset_id':'dataset_1','reviewer':'Reviewer',
            'source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),
            'flags':[{'row':16,'column':6},{'row':20,'column':7}]}])
    captions=[]
    monkeypatch.setattr(ui,'caption',lambda value,*a,**kw:captions.append(value))
    app._render_before_after(deletion_plans()[:1],SRC)
    assert len([e for e in ui.events if e[1]=='table']) == 1
    assert captions == ['1 approved annotations will be removed from cleaned data; preserved in the styled audit.']


def test_mixed_proposal_only_references_changed_values(monkeypatch):
    ui = CardUI(); app = _app(monkeypatch,ui)
    fid = 'dataset_1:qc:3'
    ui.session_state.update(cleaning_sequential_active=fid,
        cleaning_review_catalog_snapshot=[{'id':fid,'dataset_id':'dataset_1'}])
    app._render_before_after([numeric_plan()] + deletion_plans()[:1], SRC)
    tables = [e[2] for e in ui.events if e[1]=='table']
    assert len(tables) == 2
    assert tables[0] == ['Location','Before','After']
    assert 'Positive sample SD' in tables[1]
