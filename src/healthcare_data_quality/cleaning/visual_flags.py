"""Reviewer-authorized display flags; never rewrite CSV cell contents.

CSV is plain text and cannot carry font colors or weights. This module resolves
only original source coordinates. The UI renders approved flags in red/bold;
the separate styled Excel export preserves those visual annotations without
changing the original or cleaned CSV bytes.
"""

from __future__ import annotations

import csv
import hashlib
import html
import re
from pathlib import Path

_FLAG_INTENT = re.compile(r"\b(?:flag(?:ged)?|highlight(?:ed)?|highline|hilight|mark(?:ed)?|bold|red\s+font|font\s+red)\b", re.I)
_STYLE_INTENT = re.compile(r"\b(?:red|bold|highlight(?:ed)?|highline|hilight|flag(?:ged)?|mark(?:ed)?)\b", re.I)
_EXPLICIT_ROWS = re.compile(
    r"\b(?:csv\s+)?rows?\s*(?:numbers?\s*)?(\d+(?:\s*(?:,|and|&|\s)\s*\d+)*)\b", re.I
)
_DEICTIC_ROWS = re.compile(
    r"\b(?:these|those|this|the|all(?:\s+of)?(?:\s+(?:these|those))?)\s+"
    r"(?:(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+)?rows?\b", re.I
)
_SINGLE_ROW = re.compile(r"\brow\s*#?\s*(\d+)\b", re.I)
_COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_COLUMN = re.compile(r"\b(?:columns?|cols?|c)\s*#?\s*(\d+(?:\s*(?:,|and|&)\s*\d+)*)\b", re.I)


def is_visual_flag_request(message: str) -> bool:
    """Identify visual intent without requiring a model or guessing a value edit."""
    return isinstance(message, str) and bool(_FLAG_INTENT.search(message) and _STYLE_INTENT.search(message))


def requested_visual_flag(message: str, finding: dict, source: Path) -> dict | None:
    """Resolve exact visual scope; explicit row means every cell, not taxonomy only.

    Unqualified flag requests retain the taxonomy default for taxonomy findings;
    elsewhere they target the evidenced cells. No values are changed.
    """
    if not is_visual_flag_request(message):
        return None
    if re.search(r"\b(?:unflag|unhighlight|remove\s+(?:the\s+)?(?:flag|highlight)|clear\s+(?:the\s+)?(?:flag|highlight))\b", message, re.I):
        raise ValueError("To remove an approved flag, reopen its completed audit and revise it")
    if re.search(r"\b(?:replace|delete|drop|merge|set\s+to|rename)\b", message, re.I):
        raise ValueError("Request styling separately from any value or structural edit")
    text = message.strip()
    explicit = _EXPLICIT_ROWS.search(text)
    deictic = _DEICTIC_ROWS.search(text)
    repeated_rows = [int(value) for value in _SINGLE_ROW.findall(text)]
    raw = Path(source).read_bytes()
    from healthcare_data_quality.investigation.evidence_registry import verify_finding_registry
    verify_finding_registry(finding, source, enforce_source_hash=True)
    import io
    records = list(csv.reader(io.StringIO(raw.decode('utf-8-sig'), newline=''), strict=True))
    if len(records) < 2 or not records[0]:
        raise ValueError("The CSV does not contain any data rows")
    width = len(records[0])
    evidence = [e for e in finding.get('evidence', []) if isinstance(e, dict)]
    evidence_rows = {e.get('CSV row', e.get('csv_row')) for e in evidence
                     if type(e.get('CSV row', e.get('csv_row'))) is int}
    evidence_cells = {(e.get('CSV row', e.get('csv_row')), e.get('Column', e.get('column')))
                      for e in evidence if type(e.get('CSV row', e.get('csv_row'))) is int
                      and type(e.get('Column', e.get('column'))) is int}
    if explicit or repeated_rows:
        rows = sorted(set(repeated_rows) | ({int(x) for x in re.findall(r"\d+", explicit.group(1))} if explicit else set()))
    else:
        rows = sorted(evidence_rows)
    if not rows or len(rows) > 50:
        raise ValueError("Specify between 1 and 50 exact CSV row IDs")
    if any(row < 2 or row > len(records) or len(records[row-1]) != width for row in rows):
        raise ValueError("A requested source row does not exist or has inconsistent width")
    if not (explicit or repeated_rows):
        count = deictic.group(1) if deictic else None
        expected = _COUNT_WORDS.get(count.casefold(), int(count) if count and count.isdigit() else None) if count else None
        if expected is None and re.search(r"\b(?:this|that|the)\s+(?:entire\s+|whole\s+)?row\b", text, re.I):
            expected = 1
        if expected is not None and len(rows) != expected:
            raise ValueError(f"You requested {expected} rows, but this finding identifies {len(rows)}; specify the row IDs")
    column_matches = [int(n) for group in _COLUMN.findall(text) for n in re.findall(r'\d+', group)]
    if re.search(r'\b(?:columns?|rows?)\s+\d+\s*(?:-|–|through|to)\s*\d+', text, re.I):
        raise ValueError('List exact row or column IDs rather than a range')
    taxonomy = bool(re.search(r"\b(?:taxonomy|taxon|taxa|classification)\b", text, re.I))
    row_scope = bool(re.search(r"\brows?\b", text, re.I))
    cell_scope = bool(re.search(r"\bcells?\b", text, re.I))
    if column_matches:
        columns = sorted(set(column_matches))
        if any(c < 1 or c > width for c in columns):
            raise ValueError("A requested column does not exist in the source CSV")
        if taxonomy and columns != [1]:
            raise ValueError("Taxonomy and explicit column targets disagree")
        positions = [(r, c) for r in rows for c in columns]
        scope = 'cells'
    elif taxonomy:
        positions = [(r, 1) for r in rows]
        scope = 'taxonomy'
    elif row_scope and not cell_scope:
        positions = [(r, c) for r in rows for c in range(1, width + 1)]
        scope = 'rows'
    elif cell_scope or evidence_cells:
        positions = sorted((r, c) for r, c in evidence_cells if r in rows)
        if not positions or set(rows) != {r for r, _ in positions}:
            raise ValueError("Specify exact columns for the cells you want to highlight")
        scope = 'cells'
    elif finding.get('check_key') in {'inspect_taxonomy_structure', 'inspect_taxonomy_naming'} or not finding.get('check_key'):
        positions = [(r, 1) for r in rows]
        scope = 'taxonomy'
    else:
        raise ValueError("Specify whether to highlight the entire row, the taxonomy, or particular cells")
    if len(positions) > 20000 or any(not 1 <= c <= width for r,c in positions):
        raise ValueError("Visual flag scope is too large or contains invalid columns")
    flags = [{"row": r, "column": c, "before": records[r-1][c-1], "style": "bold_red", "fill": not bool(re.search(r"\b(?:font|text)\b|\b(?:no|not|without|remove)\s+(?:the\s+)?(?:yellow|background|fill)", text, re.I))}
             for r,c in positions]
    return {"dataset_id": finding.get("dataset_id", ""), "finding_id": finding.get("id", ""),
            "source_sha256": hashlib.sha256(raw).hexdigest(), "flags": flags, "scope": scope}


def visual_scope_label(plan: dict) -> str:
    flags = plan['flags']
    count = len({f['row'] for f in flags})
    if plan.get('scope') == 'rows':
        return f"{count} entire {'row' if count == 1 else 'rows'} ({len(flags)} cells)"
    if plan.get('scope') == 'taxonomy':
        return f"{len(flags)} taxonomy cell(s)"
    return f"{len(flags)} cell(s)"


def flag_preview_frame(flags: list[dict]):
    """Native UI table; preserve source strings, including empty and NA values."""
    import pandas as pd
    return pd.DataFrame([
        {"Location": f"R{int(flag['row'])}:C{int(flag['column'])}",
         "Before": str(flag["before"]), "After": str(flag["before"]),
         "Style": "Bold red + yellow fill" if flag.get("fill", True) else "Bold red font only"}
        for flag in flags
    ], columns=["Location", "Before", "After", "Style"])


def flag_preview_html(flags: list[dict]) -> str:
    """Safe, legible Before/After table; both texts are identical, style differs."""
    body = []
    for flag in flags:
        value = html.escape(str(flag["before"]), quote=True)
        body.append(
            '<tr><td>Row {row} · Column {col}</td><td>{value}</td>'
            '<td style="{css}">{value}</td></tr>'.format(
                row=int(flag["row"]), col=int(flag["column"]), value=value, css=flag_css(flag)))
    return ('<div style="max-height:360px;overflow:auto"><table style="width:100%;border-collapse:collapse">'
            '<thead><tr><th style="text-align:left">Location</th>'
            '<th style="text-align:left">Before</th><th style="text-align:left">After</th></tr></thead>'
            '<tbody>' + ''.join(body) + '</tbody></table></div>')


def styled_evidence(evidence: list[dict], approved_flags: list[dict]):
    """Style visible evidence for approved physical cells or whole rows."""
    import pandas as pd
    frame = pd.DataFrame(evidence)
    if frame.empty or not approved_flags:
        return frame
    row_key = next((key for key in ('CSV row', 'csv_row') if key in frame.columns), None)
    if not row_key:
        return frame
    text_key = next((key for key in ('Taxon / feature', 'Original classification (verbatim)',
                                   'Original classification', 'Taxonomy', 'Classification')
                     if key in frame.columns), None)
    column_key = next((key for key in ('Column', 'column') if key in frame.columns), None)
    value_key = next((key for key in ('Original value', 'Value', 'value') if key in frame.columns), None)
    locations = {(f['row'], f['column']): flag_css(f) for f in approved_flags}
    mask = pd.DataFrame('', index=frame.index, columns=frame.columns)
    for i, entry in frame.iterrows():
        r = entry[row_key]
        if text_key and (r, 1) in locations:
            mask.at[i, text_key] = locations[(r, 1)]
        if column_key and value_key and (r, entry[column_key]) in locations:
            mask.at[i, value_key] = locations[(r, entry[column_key])]
    return frame.style.apply(lambda _: mask, axis=None)


def styled_dataset_preview(table, approved_flags: list[dict]):
    """Highlight exact original physical CSV coordinates without changing values."""
    if table.empty or not approved_flags or len(table.columns) == 0:
        return table
    import pandas as pd
    mask = pd.DataFrame('', index=table.index, columns=table.columns)
    for flag in approved_flags:
        row, col = flag.get('row'), flag.get('column')
        if row in mask.index and type(col) is int and 1 <= col <= len(table.columns):
            mask.iloc[mask.index.get_loc(row), col-1] = flag_css(flag)
    return table.style.apply(lambda _: mask, axis=None)


def flag_css(flag: dict) -> str:
    return 'color: #ef4444; font-weight: bold' + ('; background-color: #ffe699; color: #dc2626' if flag.get('fill', True) else '')


def contextual_visual_request(message: str, previous: str = '') -> str | None:
    """Keep scope and style independently across the reviewer's follow-up turns.

    The canonical request is a host draft, NOT a model endorsement or permission
    to execute. Always submit it along with the verbatim request for assessment.
    """
    text = message.strip()
    visual = is_visual_flag_request(text) or bool(previous and re.search(r'\b(?:red|yellow|background|fill|font|bold)\b', text, re.I) and re.search(r'\b(?:only|just|without|no|not|remove|make|change|instead)\b', text, re.I))
    if re.search(r'\b(?:replace|delete|drop|merge|set\s+to|rename|unflag|unhighlight)\b', text, re.I):
        return text if visual else None
    scope = bool(re.search(r'\b(?:rows?|cells?|columns?|cols?|taxonomy|taxon|taxa|classification)\b', text, re.I))
    followup_scope = previous and scope and bool(re.search(r'\b(?:meant|mean|whole|entire|only|instead|all|just|those|these|this)\b', text, re.I))
    if not visual and not followup_scope:
        return None
    font_only = bool(re.search(r'\b(?:font|text)\b|\b(?:no|not|without|remove)\s+(?:the\s+)?(?:yellow|background|fill)', text, re.I))
    old_font_only = bool(re.search(r'\b(?:font|text)\b', previous, re.I))
    if re.search(r'\b(?:yellow|background|fill)\b', text, re.I) and not font_only:
        old_font_only = False
    if not scope and previous:
        # Inherit the precise previous scope, not the finding's default cells.
        base = re.sub(r'\s*;\s*bold red font only$', '', previous, flags=re.I)
        return base + ('; bold red font only' if font_only or old_font_only else '')
    if scope and not visual:
        text = 'highlight ' + text
    if (font_only or old_font_only) and not re.search(r'\b(?:font|text)\b', text, re.I):
        text += '; bold red font only'
    return text


def row_flag_preview(flags: list[dict], source: Path):
    """Show full CSV rows in their natural horizontal layout, Before and After."""
    import pandas as pd
    with Path(source).open(encoding='utf-8-sig', newline='') as stream:
        headers = next(csv.reader(stream))
    rows = sorted({f['row'] for f in flags})
    by_row = {r: sorted((f for f in flags if f['row'] == r), key=lambda f: f['column']) for r in rows}
    names = ['CSV row', 'Preview'] + [f'C{i} · {name}' for i,name in enumerate(headers, 1)]
    values, masks = [], []
    for r in rows:
        entries = by_row[r]
        if [f['column'] for f in entries] != list(range(1, len(headers)+1)):
            raise ValueError('Row preview requires the complete source row')
        vals = [f['before'] for f in entries]
        values.extend([[r, 'Before', *vals], [r, 'After', *vals]])
        masks.extend([[''] * len(names), ['', '', *[flag_css(f) for f in entries]]])
    frame = pd.DataFrame(values, columns=names)
    return frame.style.apply(lambda _: pd.DataFrame(masks, index=frame.index, columns=frame.columns), axis=None)


def visual_flags_from_history(history: list[dict]) -> dict:
    """Derive active annotation ownership from named approved events only."""
    datasets = {}
    for event in history:
        if event.get('operation') != 'flag_cells':
            continue
        flags = datasets.setdefault(event['dataset_id'], {})
        for f in event.get('flags') or []:
            flags[(f['row'], f['column'])] = {**f, 'finding_id': event['finding_id'], 'reviewer': event['reviewer']}
    return {ds: list(flags.values()) for ds, flags in datasets.items()}
