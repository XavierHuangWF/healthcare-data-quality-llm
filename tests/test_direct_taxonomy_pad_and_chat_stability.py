"""Direct, source-verified taxonomy proposals and stable Agent C widgets."""
import ast
import csv
from pathlib import Path

from healthcare_data_quality.cleaning.direct_actions import propose_explicit_taxonomy_padding
from healthcare_data_quality.cleaning.finding_cleaning import validate_scoped_plan

PROJECT = Path(__file__).resolve().parents[1]


def _source_and_finding(tmp_path: Path):
    source = tmp_path / 'synthetic_taxonomy.csv'
    short = {3: 'a;b;c;d;e;f', 5: 'a;b;c;d;e',
             15: 'a;b', 20: 'a;b;c;d;e;f'}
    evidence = []
    with source.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Taxonomy', 'Count'])
        for row in range(2, 21):
            original = short.get(row, 'a;b;c;d;e;f;g')
            writer.writerow([original, '1'])
            if row in short:
                evidence.append({
                    'CSV row': row,
                    'Original classification (verbatim)': original,
                    'Levels found': len(original.split(';')),
                    'Expected levels': 7,
                })
    finding = {'id': 'taxonomy_find', 'check_key': 'inspect_taxonomy_structure',
               'dataset_id': 'dataset_1', 'title': '4 incomplete taxonomy classifications',
               'evidence': evidence, 'total_evidence_records': len(evidence),
               'actionable': True}
    return source, finding


def test_explicit_pad_all_four_rows_returns_verifiable_before_after(tmp_path):
    source, finding = _source_and_finding(tmp_path)
    original = source.read_bytes()
    plan = propose_explicit_taxonomy_padding('pad all of them to 7 level', finding, source)
    assert plan is not None
    assert plan['operation'] == 'replace_cells'
    assert [c['row'] for c in plan['changes']] == [3, 5, 15, 20]
    assert all(len(c['after'].split(';')) == 7 for c in plan['changes'])
    assert all(c['before'] != c['after'] for c in plan['changes'])
    assert validate_scoped_plan({key: plan[key] for key in ('operation', 'dataset_id', 'changes')},
                                finding, source)['changes'] == plan['changes']
    assert source.read_bytes() == original  # Draft never edits uploaded source.


def test_no_draft_when_not_explicit_or_evidence_not_complete(tmp_path):
    source, finding = _source_and_finding(tmp_path)
    assert propose_explicit_taxonomy_padding('why is it shorter?', finding, source) is None
    assert propose_explicit_taxonomy_padding('do not pad them to 7 levels', finding, source) is None
    finding['evidence'] = finding['evidence'][:-1]
    assert propose_explicit_taxonomy_padding('pad all to seven levels', finding, source) is None


def test_chat_widget_key_does_not_rotate_after_agent_c_replies():
    code = (PROJECT / 'app.py').read_text()
    ast.parse(code)
    assert 'chat_key = (f"review_message_{fid}_" +' in code
    assert 'chat_key = f"review_message_{fid}_{_active_agent_c_control_generation(fid)}"' not in code


def test_background_ready_recommendation_never_forces_full_app_rerun():
    code = (PROJECT / 'app.py').read_text()
    renderer = code.split('def _render_live_agent_b_finding(', 1)[1].split('def _is_blank_control_check(', 1)[0]
    assert 'if not discussion_active:\n            st.rerun()' not in renderer
    assert '_expand_ready_finding_without_rerun(fid)' in renderer
    assert 'run_every=None if fid in ready_ids else "3s"' in code
