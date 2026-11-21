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
    r"(?:(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+)?rows\b", re.I
)
_SINGLE_ROW = re.compile(r"\brow\s*#?\s*(\d+)\b", re.I)
_COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_COLUMN = re.compile(r"\b(?:column|col|c)\s*#?\s*(\d+)\b", re.I)


def requested_visual_flag(message: str, finding: dict, source: Path) -> dict | None:
    """Resolve a bold/red taxonomy flag using the *selected* finding evidence.

    None means the request is not a visual flag. ValueError means it was a flag
    but targets were ambiguous or invalid; do not route it to a value-edit LLM.
    """
    if not isinstance(message, str) or not _FLAG_INTENT.search(message):
        return None
    if not _STYLE_INTENT.search(message):
        return None
    # Don't hijack requests to *change* a value that happen to mention a flag.
    if re.search(r"\b(?:replace|delete|drop|merge|set\s+to|rename)\b", message, re.I):
        return None
    if re.search(r"\b(?:unflag|unhighlight|remove\s+(?:the\s+)?(?:flag|highlight)|"
                 r"clear\s+(?:the\s+)?(?:flag|highlight))\b", message, re.I):
        raise ValueError("To remove an approved flag, reopen its completed audit and revise it")

    text = message.strip()
    explicit = _EXPLICIT_ROWS.search(text)
    deictic = _DEICTIC_ROWS.search(text)
    repeated_rows = [int(value) for value in _SINGLE_ROW.findall(text)]
    # A selected finding with exact source evidence already defines scope.
    # Thus "highlight", "flag", and "make them red" should not need a second
    # clarification round; the reviewer sees every source row before approval.

    with Path(source).open("r", encoding="utf-8-sig", newline="") as stream:
        records = list(csv.reader(stream))
    if len(records) < 2:
        raise ValueError("The CSV does not contain any data rows")
    if repeated_rows:
        rows = sorted(set(repeated_rows))
    elif explicit:
        rows = sorted({int(x) for x in re.findall(r"\d+", explicit.group(1))})
    else:
        rows = sorted({entry.get("CSV row", entry.get("csv_row"))
                       for entry in (finding.get("evidence") or [])
                       if isinstance(entry, dict) and
                       type(entry.get("CSV row", entry.get("csv_row"))) is int})
        if not rows or len(rows) > 50:
            raise ValueError("The selected finding does not identify a small, exact set of rows")
        count = deictic.group(1) if deictic else None
        expected = _COUNT_WORDS.get(count.casefold(), int(count) if count and count.isdigit() else None) if count else None
        if expected is not None and len(rows) != expected:
            raise ValueError(f"You requested {expected} rows, but this finding identifies {len(rows)}; specify the row IDs")

    col_match = _COLUMN.search(text)
    column = int(col_match.group(1)) if col_match else 1
    if column != 1:
        raise ValueError("Taxonomy font flags currently support column 1 only")
    if not rows or len(rows) > 50:
        raise ValueError("Select between 1 and 50 source rows for a single visual flag")
    if any(row < 2 or row > len(records) for row in rows):
        raise ValueError("A requested row does not exist in the original CSV")
    if any(not records[row - 1] for row in rows):
        raise ValueError("A requested row has no taxonomy value")
    flags = [{"row": row, "column": column, "before": records[row - 1][column - 1],
              "style": "bold_red"} for row in rows]
    return {"dataset_id": finding.get("dataset_id", ""),
            "finding_id": finding.get("id", ""),
            "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            "flags": flags}


def flag_preview_html(flags: list[dict]) -> str:
    """Safe, legible Before/After table; both texts are identical, style differs."""
    body = []
    for flag in flags:
        value = html.escape(str(flag["before"]), quote=True)
        body.append(
            '<tr><td>Row {row} · Column {col}</td><td>{value}</td>'
            '<td style="color:#ef4444;font-weight:800">{value}</td></tr>'.format(
                row=int(flag["row"]), col=int(flag["column"]), value=value))
    return ('<table style="width:100%;border-collapse:collapse">'
            '<thead><tr><th style="text-align:left">Location</th>'
            '<th style="text-align:left">Before</th><th style="text-align:left">After</th></tr></thead>'
            '<tbody>' + ''.join(body) + '</tbody></table>')


def styled_evidence(evidence: list[dict], approved_flags: list[dict]):
    """Render the selected taxonomy column bold/red without changing its value."""
    import pandas as pd
    frame = pd.DataFrame(evidence)
    if frame.empty or not approved_flags:
        return frame
    row_key = next((key for key in ('CSV row', 'csv_row') if key in frame.columns), None)
    text_key = next((key for key in ('Taxon / feature', 'Original classification (verbatim)',
                                   'Original classification', 'Taxonomy', 'Classification')
                     if key in frame.columns), None)
    if not row_key or not text_key:
        return frame
    flagged_rows = {item['row'] for item in approved_flags if item.get('column') == 1}
    if not flagged_rows:
        return frame
    mask = pd.DataFrame('', index=frame.index, columns=frame.columns)
    mask.loc[frame[row_key].isin(flagged_rows), text_key] = 'color: #ef4444; font-weight: bold'
    return frame.style.apply(lambda _: mask, axis=None)


def styled_dataset_preview(table, approved_flags: list[dict]):
    """Highlight approved original CSV rows in column one, preserving all values.

    The caller's index is physical CSV row numbering (header is row 1).
    """
    if table.empty or not approved_flags or len(table.columns) == 0:
        return table
    flagged = {item['row'] for item in approved_flags if item.get('column') == 1}
    if not flagged:
        return table
    import pandas as pd
    mask = pd.DataFrame('', index=table.index, columns=table.columns)
    mask.loc[mask.index.isin(flagged), table.columns[0]] = 'color: #ef4444; font-weight: bold'
    return table.style.apply(lambda _: mask, axis=None)
