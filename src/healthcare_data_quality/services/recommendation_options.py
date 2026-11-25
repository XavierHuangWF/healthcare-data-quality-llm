"""Ground Agent B's proposed controls in immutable CSV evidence.

A recommendation is never authorization. Only exact source-backed proposals
are displayed as before/after choices; Agent C and the independent approval
validator still own the staging of any chosen operation.
"""
from __future__ import annotations

import csv
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

OPERATIONS = {'replace_cells', 'drop_rows', 'drop_column', 'rename_headers', 'merge_rows'}
_ROW_COL = re.compile(
    r'row\s+(\d+)\s*,?\s*column\s+(\d+)\s*:\s*[\'\"]([^\'\"]*)[\'\"]\s*(?:→|->|to)\s*[\'\"]([^\'\"]*)[\'\"]',
    re.IGNORECASE,
)


def _rows(source: Path) -> list[list[str]]:
    with Path(source).open(encoding='utf-8-sig', newline='') as stream:
        result = list(csv.reader(stream))
    if len(result) < 2 or not result[0] or any(len(row) != len(result[0]) for row in result):
        raise ValueError('The original source is not a rectangular CSV')
    return result


def _named_blank_positions(finding: dict, rows: list[list[str]]) -> list[int]:
    details = finding.get('gemma_check_details') or {}
    proposal = details.get('proposal') or {}
    text = ' '.join(str(x or '') for x in (proposal.get('hypothesis'), proposal.get('method'), finding.get('title'), finding.get('method')))
    if not re.search(r'\b(?:blank|negative.control)\b', text, re.I):
        return []
    return [i for i,h in enumerate(rows[0],1) if i>1 and 'blank' in h.casefold()
            and re.search(r'(?<![\w])'+re.escape(h)+r'(?![\w])',text,re.I)]


def _verified(operations: list[dict], rows: list[list[str]], finding: dict) -> list[dict] | None:
    """No hallucinated coordinates, stale before values, or source-independent edits."""
    if not isinstance(operations, list) or not 1 <= len(operations) <= 16:
        return None
    result, seen = [], set()
    evidence = [e for e in (finding.get('evidence') or []) if isinstance(e, dict)]
    allowed_rows = {e.get('CSV row') for e in evidence}
    for e in evidence:
        group = e.get('CSV rows') or e.get('csv_rows')
        if isinstance(group, str):
            allowed_rows.update(int(num) for num in re.findall(r'(?<!\d)\d+(?!\d)', group))
        elif isinstance(group, (list, tuple)):
            allowed_rows.update(num for num in group if type(num) is int)
    allowed_cells = {(e.get('CSV row'), e.get('Column')) for e in evidence}
    # Taxonomy-level evidence has no numeric "Column" field: its editable
    # source value is always C1. Bind it to the verbatim original label before
    # allowing Gemma's proposed Before -> After to become a clickable choice.
    if finding.get('check_key') in {'inspect_taxonomy_structure', 'inspect_taxonomy_naming'}:
        for item in evidence:
            row = item.get('CSV row')
            original = item.get('Original classification (verbatim)',
                                item.get('Original classification'))
            if (type(row) is int and 2 <= row <= len(rows)
                    and isinstance(original, str) and original == rows[row - 1][0]):
                allowed_cells.add((row, 1))
    # Duplicate taxa must bind BOTH row identities to the same immutable
    # source label. Do not reuse an unrelated finding's row coordinates.
    duplicate_groups: dict[int, str] = {}
    if finding.get('check_key') == 'find_duplicate_taxa':
        for e in evidence:
            label = e.get('Taxon / feature')
            positions = e.get('CSV rows')
            if not isinstance(label, str) or not isinstance(positions, str):
                continue
            for value in re.findall(r'(?<!\d)\d+(?!\d)', positions):
                pos = int(value)
                if 2 <= pos <= len(rows) and rows[pos - 1][0] == label:
                    duplicate_groups[pos] = label
    missing_rows = _source_verified_missing_rows(finding, rows)
    if missing_rows:
        allowed_rows.update(missing_rows)
        allowed_cells.update((r,c) for r in missing_rows for c in range(2,len(rows[0])+1)
                             if not rows[r-1][c-1].strip())
    allowed_cols = {v for e in evidence for v in (e.get('Column'), e.get('Column A'), e.get('Column B'))}
    if finding.get('check_key') == 'gemma_check':
        allowed_cols.update(_named_blank_positions(finding, rows))
    for operation in operations:
        if not isinstance(operation, dict) or operation.get('operation') not in OPERATIONS:
            return None
        kind = operation['operation']
        changes = operation.get('changes')
        if not isinstance(changes, list) or not 1 <= len(changes) <= 3000:
            return None
        bound = []
        for change in changes:
            if not isinstance(change, dict):
                return None
            row = change.get('row')
            col = change.get('column') if kind == 'replace_cells' else change.get('position')
            if type(col) is not int or not 1 <= col <= len(rows[0]):
                if kind not in {'drop_rows', 'merge_rows'}:
                    return None
            if kind == 'drop_rows':
                if type(row) is not int or not 2 <= row <= len(rows):
                    return None
                if row not in allowed_rows:
                    return None
                coord, before, after = ('row', row), rows[row-1][0], 'Delete row'
                bound.append({'row': row, 'before': before, 'after': ''})
            elif kind == 'merge_rows':
                other = change.get('source_row')
                if (type(row) is not int or type(other) is not int
                        or row == other or row not in duplicate_groups
                        or other not in duplicate_groups
                        or duplicate_groups[row] != duplicate_groups[other]):
                    return None
                # The backend separately validates every numeric input before
                # allowing a named approval. Check here too so the Yes button
                # never offers a merger of nonnumeric or nonfinite values.
                for value in rows[row-1][1:] + rows[other-1][1:]:
                    if value.strip().upper() in {'', 'NA'}:
                        continue
                    try:
                        if not Decimal(value).is_finite():
                            return None
                    except InvalidOperation:
                        return None
                coord, before, after = ('merge', row, other), rows[row-1][0], 'sum_numeric'
                if ('row', row) in seen or ('row', other) in seen:
                    return None
                bound.append({'row': row, 'source_row': other,
                              'before': before, 'after': 'sum_numeric'})
                seen.add(('row', row))
                seen.add(('row', other))
            elif kind in {'rename_headers', 'drop_column'}:
                if col not in allowed_cols:
                    return None
                coord = ('column', col)
                before = rows[0][col-1]
                after = ('' if kind == 'drop_column' else change.get('after'))
                if kind == 'rename_headers' and (not isinstance(after, str) or not after.strip()):
                    return None
                if kind == 'drop_column' and col == 1:
                    return None
                bound.append({'position': col, 'before': before, 'after': after})
            else:
                if type(row) is not int or not 2 <= row <= len(rows):
                    return None
                if (row, col) not in allowed_cells:
                    return None
                coord = ('cell', row, col)
                before = rows[row-1][col-1]
                after = change.get('after')
                if not isinstance(after, str) or len(after) > 1500 or after == before:
                    return None
                bound.append({'row': row, 'column': col, 'before': before, 'after': after})
            # Even where we can recalculate Before, reject mismatches rather
            # than silently rewriting Gemma's alleged source evidence.
            if 'before' in change and change['before'] != before:
                return None
            if coord in seen:
                return None
            seen.add(coord)
        result.append({'operation': kind, 'dataset_id': finding.get('dataset_id', ''),
                       'changes': bound})
    return result


def _display(operations: list[dict]) -> list[dict]:
    output = []
    for op in operations:
        for change in op['changes']:
            kind = op['operation']
            if kind == 'replace_cells':
                location = f"R{change['row']} · C{change['column']}"
                after = change['after']
            elif kind == 'drop_rows':
                location, after = f"Row {change['row']}", 'Delete row'
            elif kind == 'merge_rows':
                location = f"Rows {change['row']} + {change['source_row']}"
                after = (f"Keep row {change['row']}, sum numeric counts; "
                         f"remove row {change['source_row']}")
            else:
                location = f"Column {change['position']}"
                after = 'Delete column' if kind == 'drop_column' else change['after']
            output.append({'Location': location, 'Before': change['before'], 'After': after})
    return output


def _replacements_from_evidence(finding: dict, rows: list[list[str]], after: str) -> list[dict]:
    evidence = finding.get('evidence') or []
    changes = []
    if len(evidence) != finding.get('total_evidence_records', len(evidence)):
        return []  # Evidence is only a sample; do not misrepresent coverage.
    for cell in evidence:
        if not isinstance(cell, dict):
            return []
        row, col = cell.get('CSV row'), cell.get('Column')
        if type(row) is not int or type(col) is not int or not (2 <= row <= len(rows) and 1 <= col <= len(rows[0])):
            return []
        if 'Original value' in cell and str(cell['Original value']) != rows[row-1][col-1]:
            return []
        if rows[row-1][col-1] != after:
            changes.append({'row': row, 'column': col, 'before': rows[row-1][col-1], 'after': after})
    return changes


def _taxonomy_na_padding(finding: dict, rows: list[list[str]], advice: str) -> list[dict]:
    """Create one conservative, exact option for the seven-level QC finding.

    This is positional padding, NOT an inferred biological rank assignment.
    The original labels and all affected source rows must match the QC
    evidence.  Missing rank positions cannot be inferred from plain labels.
    """
    if finding.get('check_key') != 'inspect_taxonomy_structure':
        return []
    if not (re.search(r'\b(?:pad|append|fill|complete|missing)\b', advice, re.I)
            and re.search(r'\bNA\b', advice, re.I)):
        return []
    evidence = finding.get('evidence') or []
    if (not isinstance(evidence, list) or not evidence
            or len(evidence) != finding.get('total_evidence_records', len(evidence))):
        return []
    observed_problem_rows = {
        row for row, cells in enumerate(rows[1:], start=2)
        if len(re.split(r'[;|]', cells[0])) != 7
    }
    changed = []
    evidenced_rows = set()
    for item in evidence:
        if not isinstance(item, dict):
            return []
        row = item.get('CSV row')
        if type(row) is not int or not 2 <= row <= len(rows) or row in evidenced_rows:
            return []
        before = rows[row - 1][0]
        if item.get('Original classification (verbatim)') != before:
            return []
        levels = re.split(r'[;|]', before)
        # The source report counts NA as a level, including repeated NA.
        if ('|' in before or not all(level.strip() for level in levels)
                or not 1 <= len(levels) < 7
                or item.get('Levels found') != len(levels)
                or item.get('Expected levels') != 7):
            return []
        evidenced_rows.add(row)
        changed.append({'row': row, 'column': 1, 'before': before,
                        'after': before + '; NA' * (7 - len(levels))})
    if evidenced_rows != observed_problem_rows:
        return []  # Source changed or QC provided only an evidence sample.
    return changed


def _source_verified_missing_rows(finding: dict, rows: list[list[str]]) -> list[int]:
    """Derive exact physical CSV rows that contain an empty *count* cell.

    Never guess from the eight-cell Gemma prompt sample, and never treat NA
    inside a taxonomic label, a valid numeric zero, or a blank header as an
    empty measurement. Every supplied evidence coordinate must match source;
    the baseline issue count must agree with the full source scan.
    """
    if finding.get('check_key') != 'find_invalid_counts':
        return []
    if not re.search(r'\bmissing values\b', str(finding.get('title') or ''), re.I):
        return []
    evidence = finding.get('evidence') or []
    if not isinstance(evidence, list) or not evidence:
        return []
    declared = re.match(r'^\s*(\d+)\s+cells?\s+contain\s+missing values', str(finding.get('title') or ''), re.I)
    count = finding.get('total_evidence_records', int(declared.group(1)) if declared else len(evidence))
    if type(count) is not int or count < len(evidence) or count <= 0:
        return []
    source_missing = {(r, c) for r, values in enumerate(rows[1:], 2)
                      for c in range(2, len(rows[0]) + 1)
                      if not values[c - 1].strip()}
    if len(source_missing) != count:
        return []
    observed = set()
    for item in evidence:
        if not isinstance(item, dict):
            return []
        row, col = item.get('CSV row'), item.get('Column')
        if type(row) is not int or type(col) is not int or (row, col) not in source_missing:
            return []
        if 'Original value' in item and str(item['Original value']) != rows[row - 1][col - 1]:
            return []
        if (row, col) in observed:
            return []
        observed.add((row, col))
    if len(evidence) == count and observed != source_missing:
        return []
    return sorted({r for r, _ in source_missing})


def _missing_row_choice(finding: dict, rows: list[list[str]]) -> dict | None:
    row_ids = _source_verified_missing_rows(finding, rows)
    if not row_ids or len(row_ids) > 3000 or len(row_ids) >= len(rows) - 1:
        # Never offer a button that removes every data record; the executor
        # independently prohibits the same destructive outcome.
        return None
    # Bind these exact source-derived row IDs to the current finding. Do not
    # accept arbitrary model-supplied row IDs by widening the verifier.
    operations = [{'operation': 'drop_rows', 'dataset_id': finding.get('dataset_id', ''),
                   'changes': [{'row': r, 'before': rows[r - 1][0], 'after': ''}
                               for r in row_ids]}]
    return {
        'label': f'Delete {len(row_ids)} rows containing empty cells',
        'operations': operations,
        'rows': _display(operations),
        'impact': (f'Deletes all values in {len(row_ids)} entire CSV rows, not just '
                   'the empty cells. Other measurements in those rows are lost.'),
        'derived_from_source': True,
    }




def _fractional_na_reviewer_choice(finding: dict, rows: list[list[str]]) -> dict | None:
    """A conservative human-*selectable* fallback if Gemma emits only prose.

    All anomalous values must be exactly evidenced fractional source counts;
    never guess a decimal shift, rounding rule, cell, or absent observation.
    Marking NA is a potentially lossy option, NOT a scientific conclusion.
    """
    if (finding.get('check_key') != 'find_invalid_counts'
            or not re.search(r'\bfractional\b', str(finding.get('title') or ''), re.I)):
        return None
    changes = _replacements_from_evidence(finding, rows, 'NA')
    if not changes:
        return None
    for change in changes:
        try:
            value = Decimal(change['before'])
        except (InvalidOperation, TypeError):
            return None
        if not value.is_finite() or value == value.to_integral_value():
            return None
    verified = _verified([{'operation': 'replace_cells', 'changes': changes}], rows, finding)
    if not verified:
        return None
    return {
        'label': f'Mark {len(changes)} fractional count(s) as NA',
        'operations': verified, 'rows': _display(verified),
        'derived_from_source': True,
        'impact': ('This deliberately removes the affected numeric measurements. '
                   'It does not reconstruct integer counts or establish that a '
                   'decimal-place correction is appropriate. Compare the same '
                   'columns or discuss an alternative before choosing.'),
        'provenance': ('Reviewer option derived from every evidenced source cell; '
                       'not a model-inferred true count.'),
    }


def _memory_prefers_row_deletion(suggestion: dict) -> bool:
    """Memory can rank an option but cannot supply its targets or approve it."""
    assessment = suggestion.get('memory_assessment') or {}
    # Gemma's rejection remains advisory, but must not be misrepresented as a
    # reusable preferred rule. The source-verified option remains available.
    if isinstance(assessment, dict) and assessment.get('decision') == 'rejected':
        return False
    if suggestion.get('approved_row_deletion_method_seen') is True:
        return True
    return any(isinstance(method, dict) and
               'drop_rows' in (method.get('operation_types') or []) and
               method.get('issue_relation') in {'same', 'related'}
               for method in (suggestion.get('memory_context') or []))


def _header_review_choices(finding: dict, rows: list[list[str]]) -> list[dict]:
    """Source-grounded preserve/rename choices for the first unnamed column.

    Do not invent a scientific label merely because a header is blank. The
    rename option is offered only if the column *actually* contains mainly
    seven-level taxonomy labels. Preserve is a valid human review decision.
    """
    if finding.get("check_key") != "inspect_headers":
        return []
    evidence = [e for e in (finding.get("evidence") or []) if isinstance(e, dict)]
    blanks = [e.get("Column") for e in evidence
              if type(e.get("Column")) is int and 1 <= e["Column"] <= len(rows[0])
              and not rows[0][e["Column"] - 1].strip()]
    if len(blanks) != 1:
        return []
    column = blanks[0]
    if column != 1:
        return [{"label": f"Keep column {column} header blank", "decision": "no_change",
                 "operations": [], "rows": [{"Location": f"Column {column}",
                                               "Before": "(blank)", "After": "(blank — unchanged)"}]}]
    choices = [{"label": "Keep taxonomy header blank", "decision": "no_change",
                "operations": [], "rows": [{"Location": "Column 1",
                                              "Before": "(blank)", "After": "(blank — unchanged)"}]}]
    taxonomy_like = sum(len(re.split(r"[;|]", r[0])) == 7 for r in rows[1:])
    if taxonomy_like > (len(rows) - 1) / 2:
        # A defensible descriptive name, not an inferred missing taxon value.
        rename = _verified([{"operation": "rename_headers", "changes": [
            {"position": 1, "before": "", "after": "Taxon / feature"}]}], rows, finding)
        if rename:
            choices.append({"label": "Name column 'Taxon / feature'", "operations": rename,
                            "rows": _display(rename)})
    return choices


def _source_backed_merge(finding: dict, rows: list[list[str]],
                         advice: str) -> dict | None:
    """Make an explicitly recommended duplicate-row merge actionable.

    This never extrapolates to other duplicate groups. A model's unrelated
    taxonomy-padding instructions cannot become buttons in this finding.
    """
    if finding.get('check_key') != 'find_duplicate_taxa':
        return None
    if not (re.search(r'\bmerge\b', advice, re.I)
            and re.search(r'\bsum\s+(?:numeric\s+)?counts?\b', advice, re.I)):
        return None
    match = re.search(r'\b(?:CSV\s*)?rows?\s*(\d+)\s*\+\s*(\d+)', advice, re.I)
    if not match:
        return None
    row, other = map(int, match.groups())
    if not (2 <= row <= len(rows) and 2 <= other <= len(rows)):
        return None
    checked = _verified([{'operation': 'merge_rows', 'changes': [{
        'row': row, 'source_row': other, 'before': rows[row-1][0],
        'after': 'sum_numeric',
    }]}], rows, finding)
    if not checked:
        return None
    return {'label': f'Merge rows {row} + {other} (sum counts)',
            'operations': checked, 'rows': _display(checked)}


def _matching_approved_memory_choice(finding: dict, source: Path,
                                     rows: list[list[str]], suggestion: dict) -> dict | None:
    """Promote a truly source-identical remembered edit, not raw memory text.

    An unrelated older decision is context only. The present finding and
    immutable source must BOTH match; the full existing host validator remains
    the final authority after the reviewer clicks Yes.
    """
    assessment = suggestion.get('memory_assessment') or {}
    if isinstance(assessment, dict) and assessment.get('decision') in {'rejected', 'not_applicable'}:
        return None
    try:
        from healthcare_data_quality.jobs.cleaning_jobs import ranked_reviewed_solutions
        matches = ranked_reviewed_solutions(finding, source, limit=4)
    except (OSError, ValueError, TypeError):
        return None
    for match in matches:
        if not match.get('source_match') or match.get('issue_relation') != 'same':
            continue
        record = match.get('record') or {}
        operations = record.get('operations') or []
        verified = _verified(operations, rows, finding)
        if verified:
            return {'label': 'Reuse source-verified approved method',
                    'operations': verified, 'rows': _display(verified),
                    'memory_note': 'Prior approved method on this source and issue; every current target independently reverified.'}
    return None


def source_backed_choices(finding: dict, source: Path | None,
                          suggestion: dict) -> list[dict]:
    """Select up to five model-proposed controls only when exact CSV proof is available.

    The fallback can ground simple textual replacements. A model's unverified
    title or label is never itself proof that a proposed edit is safe.
    """
    if source is None:
        return []
    try:
        rows = _rows(source)
    except (OSError, ValueError, UnicodeError, csv.Error):
        return []
    # A negative-control Gemma check can be valid with NO deletion. Do not
    # present a default-selected multiselect that silently assumes all control
    # columns should be removed. Gemma assesses BOTH the preserve choice and
    # an optional whole-column deletion before any Yes button is rendered.
    if finding.get('check_key') == 'gemma_check':
        controls = _named_blank_positions(finding, rows)
        if controls:
            keep = {'label': 'Keep negative-control columns unchanged',
                    'decision': 'no_change', 'operations': [],
                    'rows': [{'Location': f'Columns {", ".join(map(str, controls))}',
                              'Before': ', '.join(rows[0][i-1] for i in controls),
                              'After': 'Keep original columns unchanged'}]}
            all_ops = [{'operation': 'drop_column',
                        'changes': [{'position': i, 'before': rows[0][i-1], 'after': ''}]}
                       for i in controls[:12]]
            deletions = _verified(all_ops, rows, finding)
            if deletions and len(deletions) < len(rows[0]):
                return [keep, {'label': 'Delete the displayed negative-control columns',
                               'operations': deletions, 'rows': _display(deletions),
                               'impact': ('Optional structural deletion; retained negative controls '
                                          'can be important for contamination assessment.') }]
            return [keep]
        # A reproduced exploratory check with no candidates can have a genuine
        # final review decision without inventing a cleaning operation. This is
        # an exact no-edit option, not an assertion that the dataset is clean.
        details = finding.get('gemma_check_details') or {}
        result = details.get('result') or {}
        verification = details.get('verification') or {}
        if (finding.get('actionable') is False and not finding.get('evidence')
                and result.get('findings') == []
                and verification.get('status') == 'source_cells_verified_and_output_reproduced'):
            return [{'label': 'Keep data unchanged for this check',
                     'decision': 'no_change', 'operations': [],
                     'rows': [{'Location': 'Reproduced exploratory check',
                               'Before': 'No additional candidate findings reported',
                               'After': 'Keep data unchanged'}],
                     'impact': 'No new edit; not a certification of full dataset coverage.'}]
    # A taxonomy-separator finding is also a naming-convention finding if the
    # same row uses minority rank prefixes. Show the *combined* verified repair
    # first, not an incomplete separator-only answer from the model.
    if finding.get('check_key') == 'inspect_taxonomy_structure':
        from healthcare_data_quality.cleaning.direct_actions import (
            draft_format_correction, taxonomy_convention_evidence)
        combined = draft_format_correction(finding, source)
        if combined and combined.get('changes'):
            checked = _verified([{'operation': 'replace_cells',
                                  'changes': combined['changes']}], rows, finding)
            if checked:
                note = taxonomy_convention_evidence(source)
                return [{'label': 'Standardize row 7 separators and naming',
                         'operations': checked, 'rows': _display(checked),
                         'provenance': 'Entire original CSV checked; ' + str(combined['reason']),
                         'impact': ('Presentation change only: ranks and any SGB suffix preserved. '
                                    'Biological identity is not independently certified.') }]
    header_choices = _header_review_choices(finding, rows)
    if header_choices:
        # These are directly grounded in the WHOLE original CSV. A model
        # should not override a reviewer-valid keep-blank choice with a
        # speculative header replacement.
        return header_choices
    output = []
    candidates = suggestion.get('choices') or []
    if not isinstance(candidates, list):
        candidates = []
    for candidate in candidates[:5]:
        if not isinstance(candidate, dict):
            continue
        label = ' '.join(str(candidate.get('label') or '').split())[:90]
        if not label:
            continue
        operations = candidate.get('operations')
        if not operations and candidate.get('operation') in OPERATIONS:
            operations = [candidate]
        # A bounded rule lets the model choose an approach without enumerating
        # hundreds of missing cells. Source evidence, not Gemma, fixes targets.
        if not operations and candidate.get('rule') == 'replace_evidenced_cells':
            replacement = candidate.get('replacement')
            if isinstance(replacement, str):
                changes = _replacements_from_evidence(finding, rows, replacement)
                if changes:
                    operations = [{'operation': 'replace_cells', 'changes': changes}]
        grounded = _verified(operations, rows, finding)
        if grounded:
            output.append({'label': label, 'operations': grounded,
                           'rows': _display(grounded)})
    # Derive conservative complete options from the original finding; the same
    # Gemma assessment and independent approval gates apply as for model JSON.
    extra = []
    if finding.get('check_key') == 'find_invalid_counts' and 'missing values' in str(finding.get('title') or '').casefold():
        missing_rows_verified = _source_verified_missing_rows(finding, rows)
        if missing_rows_verified:
            changes = [{'row': r, 'column': c, 'before': rows[r-1][c-1], 'after': 'NA'}
                       for r in missing_rows_verified for c in range(2, len(rows[0])+1)
                       if not rows[r-1][c-1].strip()]
            extra.append(('Mark '+str(len(changes))+' missing cells as NA', changes))
            # A model sometimes enumerates only sampled missing cells, even
            # though the finding declares a complete source-verified scope.
            # The complete NA alternative replaces that redundant partial
            # suggestion. Historical exact choices are added separately below.
            full = {(c['row'], c['column']) for c in changes}
            def sampled_na(option):
                operations = option.get('operations') or []
                if len(operations) != 1 or operations[0].get('operation') != 'replace_cells':
                    return False
                subset = operations[0].get('changes') or []
                return bool(subset and all(str(c.get('before', '')).strip() == '' and
                    str(c.get('after', '')).upper() == 'NA' for c in subset) and
                    {(c['row'], c['column']) for c in subset} < full)
            output = [option for option in output if not sampled_na(option)]
    if finding.get('check_key') in {'find_invalid_counts', 'find_negative_counts'} and 'negative' in str(finding.get('title') or '').casefold():
        from decimal import Decimal, InvalidOperation
        changes = []
        for e in finding.get('evidence') or []:
            r,c = e.get('CSV row'), e.get('Column')
            if type(r) is not int or type(c) is not int or not 2<=r<=len(rows) or not 2<=c<=len(rows[0]):
                changes=[]; break
            before=rows[r-1][c-1]
            try:
                val=Decimal(before)
                if not val.is_finite() or val>=0: raise InvalidOperation
            except InvalidOperation:
                changes=[]; break
            changes.append({'row':r,'column':c,'before':before,'after':format(abs(val),'f')})
        if changes and len(changes)==finding.get('total_evidence_records',len(finding.get('evidence') or [])):
            extra.append(('Use absolute values for '+str(len(changes))+' negative counts',changes))
            extra.append(('Mark '+str(len(changes))+' negative counts as NA',
                          [{**c, 'after':'NA'} for c in changes]))
    pads = _taxonomy_na_padding(finding, rows, 'pad missing taxonomy ranks with NA')
    if pads:
        extra.append(('Pad to 7 levels with NA',pads))
    for label, changes in extra:
        verified = _verified([{'operation':'replace_cells','changes':changes}], rows, finding)
        if verified and not any(x.get('operations') == verified for x in output):
            output.insert(0, {'label':label, 'operations':verified, 'rows':_display(verified)})
    missing_rows = _missing_row_choice(finding, rows)
    if missing_rows:
        # The recommendation may only list fill approaches despite a suitable
        # prior deletion method. Add the exact source-derived alternative once.
        already_present = any(
            len(option.get('operations') or []) == 1 and
            option['operations'][0].get('operation') == 'drop_rows' and
            {c['row'] for c in option['operations'][0]['changes']} ==
            {c['row'] for c in missing_rows['operations'][0]['changes']}
            for option in output
        )
        if already_present:
            for option in output:
                operations = option.get('operations') or []
                if (len(operations) == 1 and operations[0].get('operation') == 'drop_rows'
                        and {c['row'] for c in operations[0]['changes']} ==
                            {c['row'] for c in missing_rows['operations'][0]['changes']}):
                    option['impact'] = missing_rows['impact']
        else:
            if _memory_prefers_row_deletion(suggestion):
                missing_rows['memory_note'] = (
                    'Relevant prior approved row-deletion method considered; '
                    'current row targets independently verified.')
                output = [missing_rows] + output
            else:
                output = output + [missing_rows]
    # Memory must be visible in the OPTIONS, not merely in a prose assessment.
    # Reserve one slot for the same verified original/source prior decision.
    remembered = _matching_approved_memory_choice(finding, Path(source), rows, suggestion)
    if remembered:
        signature = json.dumps(remembered['operations'], sort_keys=True)
        if not any(json.dumps(option.get('operations') or [], sort_keys=True) == signature
                   for option in output):
            output = [remembered, *output]
        else:
            for option in output:
                if json.dumps(option.get('operations') or [], sort_keys=True) == signature:
                    option['memory_note'] = remembered['memory_note']
    # A remembered approval and the current Gemma proposal may disagree.
    # Memory ranks the prior action; it must not silently suppress a fully
    # verified alternative the named researcher might deliberately select.
    # In particular, preserve both 0.2 → 2 / 0.5 → 5 from an earlier human
    # decision and 0.2 → NA / 0.5 → NA proposed for the same immutable cells.
    if finding.get('check_key') == 'find_invalid_counts':
        advice = str(suggestion.get('suggestion') or '')
        pairs = _ROW_COL.findall(advice)
        alternative = None
        if pairs and all(after.strip().upper() == 'NA' for _, _, _, after in pairs):
            changes = [{'row': int(r), 'column': int(c), 'before': old, 'after': 'NA'}
                       for r, c, old, after in pairs]
            verified_na = _verified([{'operation': 'replace_cells',
                                     'changes': changes}], rows, finding)
            if verified_na and not ('missing values' in str(finding.get('title') or '').casefold()
                                    and _source_verified_missing_rows(finding, rows)):
                alternative = {
                    'label': f'Mark {len(changes)} evidenced cell(s) as NA',
                    'operations': verified_na, 'rows': _display(verified_na),
                    'explicit_model_alternative': True,
                    'impact': ('Sets the exact displayed measurements to missing. '
                               'This does NOT reconstruct or verify raw integer counts.'),
                    'provenance': 'Current Gemma proposal; original source coordinates reverified.',
                }
        if alternative and not any(
                json.dumps(x.get('operations') or [], sort_keys=True) ==
                json.dumps(alternative['operations'], sort_keys=True) for x in output):
            output.append(alternative)
    if output:
        from healthcare_data_quality.ui.human_choice_replay import rank_choices
        return rank_choices(finding, source, output)

    # A same-taxon row merger is supported by the application's validated
    # row-operation path. Ground the two exact rows from THIS finding instead
    # of showing unactionable prose plus unrelated Yes buttons.
    advice = str(suggestion.get('suggestion') or '')
    merge = _source_backed_merge(finding, rows, advice)
    if merge:
        return [merge]

    # Prefer a complete source-exact normal-format correction to another LLM
    # drafting loop when the finding's evidence agrees with the dominant
    # observed source format. These are STILL suggestions, not approvals.
    if finding.get('check_key') in {'inspect_taxonomy_naming', 'inspect_taxonomy_structure'}:
        from healthcare_data_quality.cleaning.direct_actions import draft_format_correction
        direct = draft_format_correction(finding, source)
        if direct and direct.get('changes'):
            normalized = _verified([{'operation': 'replace_cells',
                                     'changes': direct['changes']}], rows, finding)
            if normalized:
                return [{'label': 'Use dataset-consistent taxonomy format',
                         'operations': normalized, 'rows': _display(normalized)}]

    # Older cached recommendation prose: recover source-exact replacements.
    advice = str(suggestion.get('suggestion') or '')
    matches = _ROW_COL.findall(advice)
    if matches:
        changes = [{'row': int(r), 'column': int(c), 'before': old, 'after': new}
                   for r, c, old, new in matches]
        checked = _verified([{'operation': 'replace_cells', 'changes': changes}], rows, finding)
        if checked:
            return [{'label': 'Use suggested replacements', 'operations': checked,
                     'rows': _display(checked)}]
    # A seven-level suggestion must not degrade to prose just because Gemma
    # omitted per-row coordinates. The host can derive exact before/after from
    # the complete, source-matched taxonomy QC evidence.  Offer only trailing
    # positional NA padding; do not guess where an unknown biological rank goes.
    taxonomy_changes = _taxonomy_na_padding(finding, rows, advice)
    if taxonomy_changes:
        checked = _verified([{'operation': 'replace_cells',
                              'changes': taxonomy_changes}], rows, finding)
        if checked:
            return [{'label': 'Append NA to reach 7 levels', 'operations': checked,
                     'rows': _display(checked)}]
    # Narrow fallback for a fully evidenced, explicit constant replacement.
    if re.search(r'\b(?:replace|set|fill)\b', advice, flags=re.I):
        m = re.search(r'(?:with|to)\s+[\'\"]?(NA|0|1)[\'\"]?\b', advice, flags=re.I)
        if m:
            changes = _replacements_from_evidence(finding, rows, m.group(1).upper())
            if changes:
                checked = _verified([{'operation': 'replace_cells', 'changes': changes}], rows, finding)
                if checked:
                    return [{'label': 'Use suggested replacement', 'operations': checked,
                             'rows': _display(checked)}]
    # A fractional measurement must not be erased merely to create a button.
    # The shared candidate pipeline offers a non-destructive annotation instead.
    return []
