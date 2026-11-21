"""Regression tests for read-only known-finding suppression (no Gemma needed)."""

import json
from pathlib import Path

import pytest

from healthcare_data_quality.agents import exploratory_agent as agent
from healthcare_data_quality.agents.exploratory_dedup import (
    known_issue_catalog, redundant_proposal_reason, filter_redundant_findings,
)


CODE = '''def investigate(datasets):
    return {'summary': 'checked', 'findings': []}
'''


def inputs(tmp_path: Path):
    count = tmp_path / 'counts.csv'
    count.write_text('taxon,ABC99,ABC99,ABC42\nT1,1,2,3\nT2,2,5,0\n', encoding='utf-8')
    meta = tmp_path / 'metadata.csv'
    meta.write_text('SampleID,group\nABC99,case\nABC42,control\n', encoding='utf-8')
    return {'dataset_1': count, 'dataset_2': meta}


def proposal(hypothesis, method='Compare the repeated columns'):
    return {'hypothesis': hypothesis, 'rationale': 'Potential anomaly',
            'method': method, 'assumptions': 'An analysis is possible', 'code': CODE}


def test_catalog_scans_duplicate_headers_for_each_dataset(tmp_path):
    datasets = inputs(tmp_path)
    catalog = known_issue_catalog(datasets, {'dataset_baselines': {}})
    assert catalog['dataset_1']['repeated_column_headers'] == {'ABC99': [2, 3]}
    assert 'duplicate_headers' in catalog['dataset_1']['issue_types']
    assert 'duplicate_headers' not in catalog['dataset_2']['issue_types']


def test_duplicate_abc99_hypothesis_is_rejected_before_approval(tmp_path):
    catalog = known_issue_catalog(inputs(tmp_path), {})
    proposed = proposal(
        'The presence of duplicate column headers (ABC99) at positions 2 and 3 '
        'might indicate a data merging or transcription error.',
        'Compare the distribution of values across both columns to see if they differ.',
    )
    assert 'already shown' in redundant_proposal_reason(proposed, catalog, [])


def test_cross_dataset_identity_analysis_is_not_excluded(tmp_path):
    catalog = known_issue_catalog(inputs(tmp_path), {})
    distinct = proposal(
        'Does sample ABC99 in the count matrix link to the correct clinical metadata record?',
        'Check identifiers across datasets and test contradictions between metadata and measurements.',
    )
    assert redundant_proposal_reason(distinct, catalog, []) is None


def test_novel_batch_relationship_is_not_excluded_by_baseline_negative_count(tmp_path):
    catalog = known_issue_catalog(inputs(tmp_path), {
        'dataset_baselines': {'dataset_1': {'checks': {
            'find_invalid_counts': {'status': 'completed', 'counts_by_type': {'negative': 2}}
        }}}
    })
    distinct = proposal(
        'Is there an association between negative count frequencies and collection batch?',
        'Test cross-variable correlation with the recorded batch metadata.',
    )
    assert redundant_proposal_reason(distinct, catalog, []) is None


def test_proposal_from_previous_round_is_rejected(tmp_path):
    catalog = known_issue_catalog(inputs(tmp_path), {})
    old = proposal('An unusual relationship between sample counts and clinical outcomes',
                   'Compute an evidence-backed association')
    history = [{'proposal': old, 'result': {'summary': '', 'findings': []}}]
    assert 'previously completed' in redundant_proposal_reason(old, catalog, history)


def test_one_duplicate_finding_filtered_but_novel_finding_retained(tmp_path):
    catalog = known_issue_catalog(inputs(tmp_path), {})
    evidence = [{'dataset_id': 'dataset_1', 'csv_row': 2, 'column': 2, 'value': '1'}]
    original = {'summary': 'Duplicate headers and unusual batch association found', 'findings': [
        {'title': 'Duplicate column headers ABC99', 'description': 'Two columns repeat same header', 'evidence': evidence},
        {'title': 'Unexpected batch-specific measurement association',
         'description': 'Relationship with clinical outcome requires validation', 'evidence': evidence},
    ]}
    filtered, removed = filter_redundant_findings(original, catalog, [])
    assert removed == 1
    assert [finding['title'] for finding in filtered['findings']] == [
        'Unexpected batch-specific measurement association'
    ]
    assert 'Duplicate headers' not in filtered['summary']
    assert len(original['findings']) == 2  # no mutation of verified result


def test_only_already_reported_findings_return_no_new_candidates(tmp_path):
    catalog = known_issue_catalog(inputs(tmp_path), {})
    filtered, count = filter_redundant_findings({
        'summary': 'ABC99 is a duplicate header',
        'findings': [{'title': 'ABC99 duplicate column headers',
                      'description': 'Both columns use same ID ABC99', 'evidence': []}],
    }, catalog, [])
    assert count == 1
    assert filtered['findings'] == []
    assert 'No new findings' in filtered['summary']


def test_generation_retries_duplicate_proposal_then_accepts_novel_one(tmp_path, monkeypatch):
    datasets = inputs(tmp_path)
    duplicate = proposal('Duplicate column headers ABC99 might indicate wrong IDs')
    fresh = proposal('Check for cross-dataset clinical/sample ID contradictions',
                     'Compare record linkage between datasets to find metadata mismatches')
    responses = [duplicate, fresh]
    prompts = []

    def model(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps(responses.pop(0))

    monkeypatch.setattr(agent, 'query_local_model', model)
    monkeypatch.setattr(agent, 'run_in_sandbox', lambda *_: pytest.fail('No code execution at proposal time'))
    result = agent.propose_exploration(datasets, {'dataset_baselines': {}}, [])
    assert result['hypothesis'] == fresh['hypothesis']
    assert len(prompts) == 2
    assert 'Already reported' in prompts[1]


def test_repeated_known_issue_proposals_stop_without_approving(tmp_path, monkeypatch):
    repeated = proposal('Duplicate column headers ABC99 might indicate wrong IDs')
    requests = []

    def model(prompt, **kwargs):
        requests.append(prompt)
        return json.dumps(repeated)

    monkeypatch.setattr(agent, 'query_local_model', model)
    result = agent.propose_exploration(inputs(tmp_path), {}, [])
    assert result['decision'] == 'stop'
    assert 'does not certify clean data' in result['reason']
    assert len(requests) == agent.MAX_HYPOTHESIS_RETRIES + 1


def negative_catalog(tmp_path):
    datasets = inputs(tmp_path)
    audit = {'dataset_baselines': {'dataset_1': {'checks': {
        'find_invalid_counts': {
            'status': 'completed',
            'counts_by_type': {'negative': 1, 'missing': 1, 'fractional': 1},
            'observations': [
                {'row': 2, 'column': 2, 'issue_type': 'negative', 'original_value': '-679'},
                {'row': 3, 'column': 2, 'issue_type': 'missing', 'original_value': ''},
                {'row': 3, 'column': 3, 'issue_type': 'fractional', 'original_value': '1.5'},
            ],
        }
    }}}}
    return datasets, known_issue_catalog(datasets, audit)


@pytest.mark.parametrize('hypothesis,method', [
    ('The presence of negative counts in a microbial count matrix, specifically AbC10, suggests a processing artifact',
     'Scan all columns for any values strictly less than zero; identify the affected CSV rows.'),
    ('Find negative counts', 'List rows containing negative counts'),
    ('Some microbial measurements have values below zero', 'Identify their locations'),
    ('There are two non-physical values in this abundance matrix', 'Scan each sample for values < 0'),
    ('ABC105 contains non-integer counts', 'Find the fractional count values'),
    ('Are some measurement values missing?', 'Find missing measurement cells'),
])
def test_reject_existing_count_qc_even_with_rephrased_hypotheses(tmp_path, hypothesis, method):
    _, catalog = negative_catalog(tmp_path)
    assert redundant_proposal_reason(proposal(hypothesis, method), catalog, []) is not None


def test_exact_screenshot_hypothesis_rejected_using_actual_fixture():
    from healthcare_data_quality.investigation.baseline_audit import run_baseline_audit
    path = Path(__file__).resolve().parent.parent / 'data' / 'dirtyL7PVDforXav.csv'
    datasets = {'dataset_1': path}
    catalog = known_issue_catalog(datasets, {'dataset_baselines': {
        'dataset_1': run_baseline_audit(path),
    }})
    screenshot = proposal(
        "The presence of negative counts in a microbial count matrix "
        "(specifically in column 'AbC10') suggests a potential data processing artifact, "
        "such as an incorrect subtraction of background noise or a normalization error, "
        "rather than a biological reality.",
        'Scan all columns for any values that are strictly less than zero. If found, '
        'identify the specific column and row to provide evidence of the non-physical value.',
    )
    assert 'Negative measurement' in redundant_proposal_reason(screenshot, catalog, [])


def test_new_cross_variable_analysis_and_negative_controls_are_allowed(tmp_path):
    _, catalog = negative_catalog(tmp_path)
    assert redundant_proposal_reason(proposal(
        'Are the observed negative count cells associated with a particular collection batch?',
        'Test association between batch metadata and measurement values.'
    ), catalog, []) is None
    assert redundant_proposal_reason(proposal(
        'Could negative control samples contain contamination?',
        'Compare the control samples against study-specific controls.'
    ), catalog, []) is None


def test_finding_with_same_qc_source_cell_rejected_even_without_category_words(tmp_path):
    _, catalog = negative_catalog(tmp_path)
    baseline_cell = {'dataset_id': 'dataset_1', 'csv_row': 2, 'column': 2, 'value': '-679'}
    novel_cell = {'dataset_id': 'dataset_2', 'csv_row': 2, 'column': 2, 'value': 'case'}
    findings = [
        {'title': 'AbC10 possible processing artifact',
         'description': 'Incorrect background subtraction at a previously identified location',
         'evidence': [baseline_cell]},
        {'title': 'AbC10 metadata link may be wrong',
         'description': 'A cross-dataset identity relationship needs review',
         'evidence': [baseline_cell, novel_cell]},
    ]
    filtered, removed = filter_redundant_findings({'summary': 'two discoveries', 'findings': findings}, catalog, [])
    assert removed == 1
    assert filtered['findings'] == [findings[1]]


def test_private_qc_cell_index_not_in_model_prompt(tmp_path):
    from healthcare_data_quality.agents.exploratory_dedup import prompt_issue_catalog
    _, catalog = negative_catalog(tmp_path)
    assert '2:2' in catalog['dataset_1']['_known_count_cells']
    public = prompt_issue_catalog(catalog)
    assert '_known_count_cells' not in public['dataset_1']
    assert 'count_negative' in public['dataset_1']['issue_types']


def test_skipped_count_check_does_not_suppress_hypothesis(tmp_path):
    datasets = inputs(tmp_path)
    catalog = known_issue_catalog(datasets, {'dataset_baselines': {'dataset_1': {'checks': {
        'find_invalid_counts': {'status': 'skipped_unverified_context', 'counts_by_type': {'negative': 2}}
    }}}})
    assert redundant_proposal_reason(proposal('Find negative counts', 'Scan for negative counts'), catalog, []) is None


def test_generator_retries_screenshot_negative_count_before_human_approval(tmp_path, monkeypatch):
    datasets, _ = negative_catalog(tmp_path)
    duplicate = proposal('The presence of negative counts in a microbial count matrix suggests an error',
                         'Scan for values less than zero')
    novel = proposal('Does sample ABC99 link to conflicting clinical metadata records?',
                     'Compare identities across datasets and test metadata consistency')
    responses = [duplicate, novel]
    monkeypatch.setattr(agent, 'query_local_model', lambda *_args, **_kwargs: json.dumps(responses.pop(0)))
    audit = {'dataset_baselines': {'dataset_1': {'checks': {'find_invalid_counts': {
        'status': 'completed', 'counts_by_type': {'negative': 1}
    }}}}}
    selected = agent.propose_exploration(datasets, audit, [])
    assert selected['hypothesis'] == novel['hypothesis']
    assert not responses
