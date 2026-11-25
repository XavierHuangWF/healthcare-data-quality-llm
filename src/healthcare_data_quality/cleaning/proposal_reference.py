"""Read-only reference evidence from surviving peers, never an authorization."""
import re
from .direct_actions import PREFIXES
from .numeric_context import _number


def taxonomy_format(value):
    parts = [p.strip() for p in re.split(r'[;|]', str(value))]
    meaningful = [(i,p) for i,p in enumerate(parts) if p and p.upper() != 'NA']
    prefixed = sum(p.casefold().startswith(PREFIXES[i]) for i,p in meaningful if i < len(PREFIXES))
    style = ('Unprefixed' if not prefixed else 'Fully prefixed' if prefixed == len(meaningful) else 'Mixed prefixes')
    sep = 'Mixed separators' if ';' in value and '|' in value else 'Pipe' if '|' in value else 'Semicolon' if ';' in value else 'None'
    spacing = 'Space after separator' if '; ' in value else 'No space after separator'
    return (len(parts),style,sep,spacing)


def reference_evidence(projected, plans, *, limit=5):
    """Compact, reproducible references; all counts cover surviving peers.

    Pure deletion needs no peer reference; show its exact scope in the preview.
    Use original coordinates even after deletions. Examples are not proof of
    biological identity or correctness of a proposed replacement.
    """
    records, row_ids, col_ids = projected
    if not records:
        return [], []
    ci = {c:i for i,c in enumerate(col_ids)}
    ri = {r:i for i,r in enumerate(row_ids)}
    targets = {}
    merged_rows, renamed_cols = set(), set()
    for p in plans:
        for ch in p.get('changes') or []:
            kind = p.get('operation')
            r,c = ch.get('row'),ch.get('column')
            if kind == 'replace_cells' and r in ri and c in ci and r != 1:
                # Numeric replacements have a richer same-column statistics
                # table in the proposal card. Do not duplicate those references.
                if c != 1 and _number(ch.get('after')) is not None:
                    continue
                targets.setdefault(c, set()).add(r)
            elif kind == 'merge_rows':
                # Merging changes surviving values; pure deletion does not.
                merged_rows.update(x for x in (r,ch.get('source_row')) if x in ri)
            elif kind == 'rename_headers' and ch.get('position') in ci:
                renamed_cols.add(ch['position'])
    summary, examples = [], []
    for c, excluded in sorted(targets.items()):
        pos = ci[c]
        peers = [(r,row[pos]) for r,row in zip(row_ids[1:],records[1:]) if r not in excluded]
        if not peers:
            continue
        # The application's first original column is the taxonomy field. Use
        # representative actual classifications in each format, not invented taxa.
        if c == 1:
            groups = {}
            for r,v in peers:
                groups.setdefault(taxonomy_format(v),[]).append((r,v))
            for (levels,style,sep,spacing), members in sorted(groups.items(),key=lambda item:(-len(item[1]),item[0]))[:limit]:
                summary.append({'Reference':f'C{c} · {style}', 'Rows':len(members),'Ranks':levels,'Separator':sep,'Spacing':spacing})
                for r,v in members[:2 if not examples else 1]:
                    examples.append({'Source':f'R{r}:C{c}','Reference value':v})
        elif len(targets) <= 4:
            # Text/NA replacements may still need source-grounded examples.
            distinct = {}
            for r,v in peers:
                if str(v).strip() and str(v).upper() != 'NA':
                    distinct.setdefault(v,r)
            for v,r in list(distinct.items())[:3]:
                examples.append({'Source':f'R{r}:C{c}','Reference value':v})
            summary.append({'Reference':f'C{c} · {records[0][pos]}','Rows':len(peers),'Missing':sum(not str(v).strip() or str(v).upper()=='NA' for _,v in peers),'Distinct nonmissing':len(distinct)})
    for c in sorted(renamed_cols):
        pos = ci[c]
        neighbors = sorted((x for x in col_ids if x != c),key=lambda x:(abs(x-c),x))[:3]
        for x in neighbors:
            examples.append({'Source':f'R1:C{x}','Reference value':records[0][ci[x]]})
        summary.append({'Reference':f'Column {c}','Nonempty headers':sum(bool(str(v).strip()) for v in records[0]),'Columns':len(col_ids)})
    for r in sorted(merged_rows)[:max(limit, 20)]:
        row = records[ri[r]]
        counts = [v for c,v in zip(col_ids,row) if c != 1]
        summary.append({'Reference':f'Row {r}', 'Missing':sum(not str(v).strip() or str(v).upper()=='NA' for v in counts),'Nonmissing':sum(bool(str(v).strip()) and str(v).upper()!='NA' for v in counts),'Nonzero numeric':sum(_nonzero(v) for v in counts)})
    return summary[:max(limit, min(40,len(merged_rows)))], examples[:limit]


def _nonzero(value):
    from decimal import Decimal, InvalidOperation
    try:
        n=Decimal(str(value))
        return n.is_finite() and n != 0
    except (InvalidOperation,ValueError):
        return False


def taxonomy_action_label(label, plans):
    """Describe the actual target format, independent of cached/model titles."""
    changes = [c for p in plans if p.get('operation') == 'replace_cells'
               for c in p.get('changes', []) if c.get('column') == 1 and c.get('after')]
    if not changes:
        return label  # compatibility for incomplete legacy descriptions
    formats = {taxonomy_format(str(c['after']))[1:] for c in changes}
    n = len(changes)
    scope = f'{n} row{"s" if n != 1 else ""}'
    if len(formats) != 1:
        return f'Correct taxonomy format ({scope})'
    style, sep, spacing = next(iter(formats))
    return f'Standardize taxonomy: {style} · {sep.lower()} · {"with spaces" if spacing.startswith("Space ") else "without spaces"} ({scope})'
