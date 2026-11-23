"""Human-readable, source-bound final action labels for all review pathways.

The title itself does not authorize an edit. The clicked action is revalidated
against its displayed Before → After and immutable original CSV at execution.
"""
from __future__ import annotations


def final_action_label(plans: list[dict] | dict | None, finding: dict | None = None) -> str:
    if isinstance(plans, dict):
        plans = [plans]
    plans = [p for p in (plans or []) if isinstance(p, dict)]
    if not plans:
        return "Apply the verified changes"
    kinds = [p.get("operation") for p in plans]
    positions = [p.get('changes', [{}])[0].get('position') for p in plans
                 if p.get('operation') == 'drop_column' and p.get('changes')]
    rows = [c.get('row') for p in plans if p.get('operation') == 'drop_rows'
            for c in p.get('changes', [])]
    if set(kinds) == {'drop_column'} and positions and all(type(p) is int for p in positions):
        names = sorted(set(positions))
        return ("Delete column " if len(names) == 1 else "Delete columns ") + ", ".join(map(str, names))
    if set(kinds) == {'drop_rows'} and rows and all(type(r) is int for r in rows):
        names = sorted(set(rows))
        return ("Delete row " if len(names) == 1 else "Delete rows ") + ", ".join(map(str, names[:9])) + ("…" if len(names) > 9 else "")
    if set(kinds) == {'replace_cells'}:
        count = sum(len(p.get('changes') or []) for p in plans)
        key = (finding or {}).get('check_key')
        if key == 'inspect_taxonomy_structure':
            return f"Standardize separator and naming for {count} taxonomy row" + ("s" if count != 1 else "")
        if key == 'inspect_taxonomy_naming':
            return f"Standardize naming for {count} taxonomy row" + ("s" if count != 1 else "")
        return f"Apply {count} verified cell change" + ("s" if count != 1 else "")
    if set(kinds) == {'rename_headers'}:
        return "Apply verified header rename"
    if set(kinds) == {'merge_rows'}:
        return "Merge the displayed duplicate rows"
    summary = ', '.join(dict.fromkeys(str(k).replace('_', ' ') for k in kinds))
    return f"Apply verified changes ({summary})"[:120]
