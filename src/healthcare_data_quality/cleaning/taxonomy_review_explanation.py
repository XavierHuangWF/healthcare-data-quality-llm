"""Source-only taxonomy-level inspection for a reviewer's question.

This is an assessment, not a cleaning plan or a biological taxon lookup.
It never authorizes a source edit or infers taxonomy correctness from syntax.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

_PREFIXES = ('k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__')
_RANKS = ('Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species')


def taxonomy_level_answer(question: str, finding: dict, source: Path | None) -> str | None:
    """Answer a taxonomy *inspection* request using exact original CSV evidence.

    Never convert an inspection question into approval of the already proposed
    separator edit. A request to verify biological identity isn't answerable
    from a count-table CSV alone and must not be claimed verified.
    """
    if source is None or finding.get('check_key') not in {
        'inspect_taxonomy_structure', 'inspect_taxonomy_naming',
    }:
        return None
    text = ' '.join(str(question).casefold().split())
    if not (re.search(r'\b(?:verify|check|inspect|assess|review|correct|right|valid)\b', text)
            and re.search(r'\b(?:classification|taxonomy|taxonomic|rank|level|species)\b', text)):
        return None
    with Path(source).open('r', encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.reader(stream, strict=True))
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('Original CSV is not rectangular')
    evidence = finding.get('evidence') or []
    targets = []
    for item in evidence:
        if not isinstance(item, dict) or type(item.get('CSV row')) is not int:
            continue
        csv_row = item['CSV row']
        if csv_row < 2 or csv_row > len(rows):
            raise ValueError('Taxonomy evidence references an invalid CSV row')
        value = rows[csv_row - 1][0]
        original = item.get('Original classification') or item.get('Original classification (verbatim)')
        if original is not None and str(original) != value:
            raise ValueError('Taxonomy evidence disagrees with original CSV')
        tokens = [token.strip() for token in re.split(r'[;|]', value)]
        prefix_correct = (len(tokens) == len(_RANKS)
                          and all(token.casefold().startswith(prefix)
                                  for token, prefix in zip(tokens, _PREFIXES)))
        has_empty = any(not token or token.upper() == 'NA' for token in tokens)
        targets.append((csv_row, tokens, prefix_correct, has_empty, '|' in value))
    if not targets:
        return None
    parts = []
    for csv_row, tokens, prefix_correct, has_empty, alternate_sep in targets[:8]:
        parts.append(
            f'Row {csv_row}: {len(tokens)}/7 taxonomic positions; '
            + ('rank prefixes follow k/p/c/o/f/g/s order' if prefix_correct
               else 'rank prefixes are absent, mixed or out of order')
            + ('; at least one rank is empty/NA' if has_empty else '')
            + ('; uses | rather than ; between levels' if alternate_sep else '')
            + '.')
    if len(targets) > 8:
        parts.append(f'{len(targets) - 8} additional evidenced rows were not listed.')
    from healthcare_data_quality.cleaning.direct_actions import (
        taxonomy_convention_evidence, draft_format_correction)
    consensus = taxonomy_convention_evidence(source)
    parts.append(
        f"Across {consensus['data_rows']} original rows, "
        f"{consensus['semicolon_rows']} use semicolon separators and "
        f"{consensus['unprefixed_seven_rank_rows']} seven-rank rows "
        "use unprefixed naming. Both presentation conventions should be considered "
        "together, with all rank names and strain suffixes preserved.")
    possible = draft_format_correction(finding, source)
    if possible and possible.get('changes'):
        parts.append(
            'A source-exact combined formatting proposal is available for the '
            'evidenced row; review its Before → After before authorization.')
    parts.append(
        'This verifies the CSV hierarchy structure only, not the biological '        'accuracy of the organism classification. No changes were applied. '
        'The separator-only proposal cannot repair a scientifically incorrect '
        'taxonomic label; keep those decisions separate.')
    return ' '.join(parts)
