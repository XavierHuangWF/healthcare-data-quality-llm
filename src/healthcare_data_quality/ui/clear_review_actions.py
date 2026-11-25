"""Human-readable, source-bound final action labels for all review pathways.

The title itself does not authorize an edit. The clicked action is revalidated
against its displayed Before → After and immutable original CSV at execution.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


def direct_action_label(label: str | None) -> str:
    """Display one actionable title, without changing the reviewed plan identity.

    Normalize presentation only; widget keys, original option labels, fingerprints,
    and execution authorization remain unchanged.
    """
    text = str(label or '').strip()
    text = re.sub(r'^yes\s*[—–:-]\s*', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'^Delete rows (\d+)$', r'Delete row \1', text)
    text = re.sub(r'\b(\d+) fractional count\(s\)',
                  lambda m: f"{m[1]} fractional count" + ('s' if int(m[1]) != 1 else ''), text)
    return (text or 'Apply verified changes')[:140]


def final_action_label(plans: list[dict] | dict | None, finding: dict | None = None) -> str:
    if isinstance(plans, dict):
        plans = [plans]
    plans = [p for p in (plans or []) if isinstance(p, dict)]
    if not plans:
        return "Apply the verified changes"
    kinds = [p.get("operation") for p in plans]
    if len(set(kinds)) > 1:
        # Use the operations, not a cached model title, even for compound actions.
        parts = [final_action_label([p for p in plans if p.get('operation') == kind], finding)
                 for kind in dict.fromkeys(kinds)]
        combined = ' + '.join(parts)
        return combined if len(combined) <= 140 else f"Apply {len(kinds)} operations: " + ', '.join(
            {'replace_cells':'replace cells','drop_rows':'delete rows','drop_column':'delete columns',
             'rename_headers':'rename headers','merge_rows':'merge rows','flag_cells':'highlight'}.get(k,'verified changes')
            for k in dict.fromkeys(kinds))
    if set(kinds) == {'flag_cells'}:
        changes = [c for p in plans for c in p.get('changes') or []]
        rows = {c.get('row') for c in changes if type(c.get('row')) is int}
        scope = {p.get('visual_scope') for p in plans}
        n = len(changes)
        style = ('red/bold only' if changes and all(c.get('fill') is False for c in changes)
                 else 'red/bold + yellow fill' if changes and all(c.get('fill', True) is True for c in changes)
                 else 'approved styles')
        if scope == {'rows'} and rows:
            return f"Highlight {len(rows)} entire row{'s' if len(rows) != 1 else ''} ({n} cell{'s' if n != 1 else ''}) · {style}"
        return f"Highlight {n} cell{'s' if n != 1 else ''} · {style}"
    positions = [c.get('position') for p in plans if p.get('operation') == 'drop_column'
                 for c in p.get('changes') or []]
    rows = [c.get('row') for p in plans if p.get('operation') == 'drop_rows'
            for c in p.get('changes', [])]
    if set(kinds) == {'drop_column'} and positions and all(type(p) is int for p in positions):
        names = sorted(set(positions))
        return ("Delete column " if len(names) == 1 else "Delete columns ") + ", ".join(map(str, names[:9])) + ("…" if len(names) > 9 else "")
    if set(kinds) == {'drop_rows'} and rows and all(type(r) is int for r in rows):
        names = sorted(set(rows))
        return ("Delete row " if len(names) == 1 else "Delete rows ") + ", ".join(map(str, names[:9])) + ("…" if len(names) > 9 else "")
    if set(kinds) == {'replace_cells'}:
        count = sum(len(p.get('changes') or []) for p in plans)
        key = (finding or {}).get('check_key')
        changes = [c for p in plans for c in p.get('changes') or []]
        if changes and all(c.get('column') == 1 and
                1 <= len(str(c.get('before', '')).split(';')) < 7 and
                str(c.get('after', '')) == str(c.get('before', '')) + '; NA' * (7-len(str(c.get('before', '')).split(';')))
                for c in changes):
            return f"Pad {count} taxonomy row{'s' if count != 1 else ''} to 7 levels with NA"
        if changes and all(str(c.get('after')) == 'NA' for c in changes):
            title = str((finding or {}).get('title', '')).lower()
            noun = ('fractional count' if 'fractional' in title else 'missing cell' if 'missing' in title else 'cell')
            return f"Mark {count} {noun}{'s' if count != 1 else ''} as NA"
        try:
            if changes and all(Decimal(str(c.get('before'))) < 0 and
                    Decimal(str(c.get('after'))) == abs(Decimal(str(c.get('before'))))
                    for c in changes):
                return f"Use absolute values for {count} negative count{'s' if count != 1 else ''}"
        except (InvalidOperation, ValueError):
            pass
        if key in {'inspect_taxonomy_structure', 'inspect_taxonomy_naming'} or (
                changes and all(c.get('column') == 1 and
                    any(sep in str(c.get('after', '')) for sep in (';', '|')) for c in changes)):
            from healthcare_data_quality.cleaning.proposal_reference import taxonomy_action_label
            fallback = ("Standardize separator and naming" if key == 'inspect_taxonomy_structure'
                        else "Standardize naming") + f" for {count} taxonomy row" + ("s" if count != 1 else "")
            return taxonomy_action_label(fallback, plans)
        # Short numeric mappings distinguish alternatives without ever inserting
        # long data strings, source headers or arbitrary model text in a button.
        base = f"Apply {count} verified cell change" + ("s" if count != 1 else "")
        try:
            nums = [(Decimal(str(c.get('before'))), Decimal(str(c.get('after')))) for c in changes]
            if 0 < len(nums) <= 3 and all(a.is_finite() and b.is_finite() for a,b in nums):
                pairs = ', '.join(f"{a} → {b}" for a,b in nums)
                if len(pairs) <= 60:
                    return base + ' · ' + pairs
        except (InvalidOperation, ValueError, TypeError):
            pass
        return base
    if set(kinds) == {'rename_headers'}:
        positions = sorted({c['position'] for p in plans for c in p.get('changes') or [] if type(c.get('position')) is int})
        if positions:
            return ('Rename column ' if len(positions)==1 else 'Rename columns ') + ', '.join(map(str,positions[:9])) + ('…' if len(positions)>9 else '')
        return "Apply verified header rename"
    if set(kinds) == {'merge_rows'}:
        pairs = [(c.get('source_row'), c.get('row')) for p in plans for c in p.get('changes') or []]
        if pairs and all(type(a) is int and type(b) is int for a,b in pairs):
            return 'Merge ' + '; '.join(f'row {a} into {b}' for a,b in pairs[:4]) + ('…' if len(pairs)>4 else '')
        return "Merge the displayed duplicate rows"
    summary = ', '.join(dict.fromkeys(str(k).replace('_', ' ') for k in kinds))
    return f"Apply verified changes ({summary})"[:120]


def action_label_variants(plans, finding=None, *, decision=None):
    """Safe wording space for model reflection, not a new action language.

    Every variant retains the complete canonical scope and format qualifiers.
    Arbitrary model prose cannot become an executable control.
    """
    base = ('Keep data unchanged' if decision == 'no_change'
            else direct_action_label(final_action_label(plans, finding)))
    variants = [base]
    prefixes = (
        ('Delete ', 'Remove '), ('Highlight ', 'Flag '),
        ('Standardize taxonomy:', 'Normalize taxonomy:'),
        ('Pad ', 'Extend '), ('Mark ', 'Set '),
        ('Use absolute values for ', 'Convert to absolute values: '),
        ('Apply ', 'Make '), ('Rename ', 'Relabel '),
        ('Keep data unchanged', 'Leave data unchanged'),
    )
    for before, after in prefixes:
        if base.startswith(before):
            alternative = after + base[len(before):]
            if len(alternative) <= 140:
                variants.append(alternative)
    return list(dict.fromkeys(variants))


def reflected_action_label(plans, finding=None, proposed=None, *, decision=None):
    choices = action_label_variants(plans, finding, decision=decision)
    return proposed if isinstance(proposed, str) and proposed in choices else choices[0]


def option_action_label(option: dict, finding: dict | None = None) -> str:
    """Render from the verified operations, never from memory/model label text.

    Keep the original option unchanged: its fingerprint and approval binding
    must not depend on the presentation label.
    """
    return reflected_action_label(option.get('operations'), finding,
                                  option.get('gemma_button_label'), decision=option.get('decision'))
