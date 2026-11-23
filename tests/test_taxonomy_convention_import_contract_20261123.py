"""Regression: discussion/recommendation imports match the installed module.

No model requests and no writes to the source CSV.  The original test dataset
is intentionally used so the row-number and source-evidence contract is checked.
"""
from __future__ import annotations

import ast
import csv
from pathlib import Path

from healthcare_data_quality.cleaning import direct_actions
from healthcare_data_quality.cleaning.direct_actions import (
    taxonomy_convention_evidence,
    draft_format_correction,
    draft_identical_column_removal,
    same_taxonomy_identity,
    already_resolved_format_issue,
    propose_explicit_taxonomy_padding,
    propose_explicit_column_deletions,
)

ROOT = Path(direct_actions.__file__).resolve().parents[3]
SOURCE = ROOT / 'data' / 'dirtyL7PVDforXav.csv'


def test_all_project_named_imports_from_direct_actions_exist():
    """Catch app/service drift even when imports happen inside functions."""
    exported = set(dir(direct_actions))
    for file in [ROOT / 'app.py', *(ROOT / 'src').rglob('*.py')]:
        if file == Path(direct_actions.__file__).resolve():
            continue
        for node in ast.walk(ast.parse(file.read_text(encoding='utf-8'))):
            if (isinstance(node, ast.ImportFrom)
                    and node.module == 'healthcare_data_quality.cleaning.direct_actions'):
                missing = {symbol.name for symbol in node.names if symbol.name not in exported}
                assert not missing, f'{file}: direct_actions exports missing {sorted(missing)}'


def test_taxonomy_evidence_matches_uploaded_source_without_changing_it():
    before = SOURCE.read_bytes()
    stats = taxonomy_convention_evidence(SOURCE)
    assert stats['data_rows'] == 158
    assert stats['semicolon_rows'] == 157
    assert stats['majority_style'] == 'unprefixed'
    assert stats['seven_rank_semicolon_rows'] == 153
    assert SOURCE.read_bytes() == before


def test_row_seven_combines_separator_and_naming_safely():
    with SOURCE.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.reader(handle))
    original = rows[6][0]
    finding = {
        'dataset_id': 'dataset_1',
        'check_key': 'inspect_taxonomy_structure',
        'evidence': [{'CSV row': 7, 'Original classification': original,
                      'Problem': 'Uses | instead of ;'}],
    }
    proposal = draft_format_correction(finding, SOURCE)
    assert proposal is not None
    assert proposal['operation'] == 'replace_cells'
    assert len(proposal['changes']) == 1
    change = proposal['changes'][0]
    assert (change['row'], change['column']) == (7, 1)
    assert change['before'] == original
    assert '|' not in change['after']
    assert 'k__' not in change['after']
    assert same_taxonomy_identity(change['before'], change['after'])
