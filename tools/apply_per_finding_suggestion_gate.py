"""Install a per-finding wait gate, preserving the existing navigation & agents.

Usage (from the project root): python tools/apply_per_finding_suggestion_gate.py
Safe, idempotent AST-based edits to app.py; refuses unfamiliar code layouts.
"""
from __future__ import annotations

import ast
from pathlib import Path

MARK = "# HDQ_PER_FINDING_SUGGESTION_GATE_V1"


def _named(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _streamlit_method(node: ast.AST, method: str) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and _named(node.func.value, "st") and node.func.attr == method)


def _line_at(source: str, number: int) -> str:
    return source.splitlines(keepends=True)[number - 1]


def transform(source: str) -> str:
    if MARK in source:
        return source
    tree = ast.parse(source)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "render_cleaning_workspace" not in fns or "render_ui" not in fns:
        raise RuntimeError("Cannot uniquely find cleaning workspace and render_ui; no changes made")
    cleaning = fns["render_cleaning_workspace"]
    render_ui = fns["render_ui"]
    lines = source.splitlines(keepends=True)
    patches: list[tuple[int, int, str]] = []

    # 1. Lock only individual expandable finding/check cards. This is a
    # noninteractive row before readiness and remains a normal expander after.
    cards = []
    for node in ast.walk(cleaning):
        if not isinstance(node, ast.With):
            continue
        for entry in node.items:
            call = entry.context_expr
            if not _streamlit_method(call, "expander") or not call.args:
                continue
            label = ast.unparse(call.args[0])
            if 'finding[\'title\']' in label and "label" in label:
                cards.append(node)
    if len(cards) != 1:
        raise RuntimeError(f"Expected one per-finding expander, found {len(cards)}; no changes made")
    card = cards[0]
    line = _line_at(source, card.lineno)
    indent = line[:len(line) - len(line.lstrip())]
    block = (
        f"{indent}{MARK}\n"
        f"{indent}from healthcare_data_quality.ui.finding_wait_gate import (\n"
        f"{indent}    is_locked as _hdq_finding_locked, render_waiting_card as _hdq_wait_row,\n"
        f"{indent})\n"
        f"{indent}if _hdq_finding_locked(\n"
        f"{indent}    finding_id=fid, snapshot=state, completed=completed,\n"
        f"{indent}    active_conversation=active_conversation,\n"
        f"{indent}):\n"
        f"{indent}    _hdq_entries = ((state or {{}}).get('entries') or {{}})\n"
        f"{indent}    _hdq_entry = _hdq_entries.get(fid) or {{}}\n"
        f"{indent}    _hdq_wait_row(st, label, finding['title'],\n"
        f"{indent}                  str(_hdq_entry.get('status') or 'queued'))\n"
        f"{indent}    continue\n"
    )
    patches.append((card.lineno, card.lineno, block))

    # 2. One readiness watcher replaces a page-level lock. The worker runs
    # independently of the UI; a readiness transition refreshes its parent.
    ready = []
    for node in ast.walk(cleaning):
        if not isinstance(node, ast.Assign):
            continue
        if any(_named(target, 'ready_ids') for target in node.targets):
            ready.append(node)
    if len(ready) != 1:
        raise RuntimeError(f"Expected one ready_ids calculation, found {len(ready)}")
    rnode = ready[0]
    rline = _line_at(source, rnode.lineno)
    rindent = rline[:len(rline) - len(rline.lstrip())]
    watcher = (
        f"{rindent}# One non-UI watcher for every pending finding; no button remounts.\n"
        f"{rindent}if job_id and hasattr(st, 'fragment'):\n"
        f"{rindent}    from healthcare_data_quality.ui.finding_wait_gate import poll_finding_readiness\n"
        f"{rindent}    st.fragment(run_every='3s')(poll_finding_readiness)(st, job_id, state or {{}})\n"
    )
    patches.append((rnode.end_lineno + 1, rnode.end_lineno + 1, watcher))

    # 3. Navigation is always visible. Only the individual finding row uses
    # readiness; no other app-level state is overridden. Supports old tabs and
    # the newer four-button UI if they still call the same readiness helper.
    model_gate_calls = [node for node in ast.walk(render_ui)
                        if isinstance(node, ast.Call) and _named(node.func, '_agent_b_first_suggestion_ready')]
    # Derive physical line ranges; old releases have two one-line expressions.
    for node in model_gate_calls:
        if node.lineno != node.end_lineno:
            raise RuntimeError("Multiline navigation readiness call: manual review required")
        physical = _line_at(source, node.lineno)
        prior = physical[:node.col_offset]
        after = physical[node.end_col_offset:]
        patches.append((node.lineno, node.lineno + 1, prior + 'True' + after))
    # In the newer UI, the prominent first button may have an independent
    # disabled gate. Explicitly lift it without altering adjacent nav buttons.
    nav_buttons = []
    for node in ast.walk(render_ui):
        if not _streamlit_method(node, "button") or not node.args:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and (
                'Clean data with LM model' in arg.value or 'Clean data with Gemma' in arg.value):
            nav_buttons.append(node)
    # Only modify a literal single-line disabled=... expression here; a
    # complex button layout requires a source-specific follow-up, not guessing.
    for node in nav_buttons:
        key = next((k for k in node.keywords if k.arg == 'disabled'), None)
        if key is not None and key.value.lineno == key.value.end_lineno:
            lno = key.value.lineno
            physical = _line_at(source, lno)
            patches.append((lno, lno + 1,
                            physical[:key.value.col_offset] + 'False' + physical[key.value.end_col_offset:]))

    # Apply insertions and substitutions backwards. Detect overlapping edits.
    covered = set()
    for start, end, _ in patches:
        if start < end:
            for ln in range(start, end):
                if ln in covered:
                    raise RuntimeError('Ambiguous overlapping source edits; no changes made')
                covered.add(ln)
    for start, end, replacement in sorted(patches, key=lambda v: (v[0], v[1]), reverse=True):
        lines[start - 1:end - 1] = [replacement]
    candidate = ''.join(lines)
    ast.parse(candidate)
    if candidate.count(MARK) != 1:
        raise RuntimeError("Finding-level gate marker incorrect; no changes made")
    return candidate


MIGRATIONS_JSON = '{"test_cleaning_review_cards.py": [["    assert len(st.expanders) == 3\\n    assert \\"Finding 1 · Fractional counts\\" in st.expanders[0]\\n    assert \\"Finding 2 · Missing counts\\" in st.expanders[1]\\n    assert \\"Gemma check 1 · Checked controls: no issue\\" in st.expanders[2]\\n", "    # All headings remain visible, but NONE is expandable until its own\\n    # current-run Agent B recommendation has finished.\\n    assert len(st.expanders) == 0\\n"], ["    st.buttons_on.add(\\"review_no_action_dataset_1:qc:2\\")\\n", "    class ReadyQueue:\\n        def snapshot(self, _):\\n            return {\\"entries\\": {\\"dataset_1:qc:2\\": {\\n                \\"status\\": \\"ready\\", \\"suggestion\\": \\"Review missing count cells\\", \\"choices\\": []}}}\\n    monkeypatch.setattr(\\"healthcare_data_quality.jobs.cleaning_jobs.recommendation_manager\\",\\n                        lambda: ReadyQueue())\\n    st.buttons_on.add(\\"review_no_action_dataset_1:qc:2\\")\\n"], ["    assert len(st.expanders) == 3  # one card per finding/check, no duplicate audit panel\\n", "    assert len(st.expanders) == 0  # headings shown, but no card expandable before Agent B\\n"]], "test_cleaning_fragment_scroll.py": [["    st.buttons_on.add(\\"review_no_action_dataset_1:qc:1\\")\\n", "    class ReadyQueue:\\n        def snapshot(self, _):\\n            return {\\"entries\\": {\\"dataset_1:qc:1\\": {\\n                \\"status\\": \\"ready\\", \\"suggestion\\": \\"Review fractional count\\", \\"choices\\": []}}}\\n    monkeypatch.setattr(\\"healthcare_data_quality.jobs.cleaning_jobs.recommendation_manager\\",\\n                        lambda: ReadyQueue())\\n    st.buttons_on.add(\\"review_no_action_dataset_1:qc:1\\")\\n"]], "test_fresh_investigation_agent_handoff.py": [["def test_direct_agent_c_discussion_is_available_before_agent_b(monkeypatch, tmp_path):\\n", "def test_direct_agent_c_discussion_locked_until_agent_b_ready(monkeypatch, tmp_path):\\n"], ["    with pytest.raises(StopIteration, match=\\"streamlit rerun\\"):\\n        app.render_cleaning_workspace([path], [SimpleNamespace(name=path.name)],\\n                                      audit, \\"same-csv-key\\", [])\\n    assert st.session_state.cleaning_discussion_mode_by_finding[finding[\\"id\\"]] == \\"alternative\\"\\n    assert st.session_state.cleaning_review_statuses[finding[\\"id\\"]] == \\"Alternative requested\\"\\n", "    app.render_cleaning_workspace([path], [SimpleNamespace(name=path.name)],\\n                                  audit, \\"same-csv-key\\", [])\\n    assert st.expanders == []\\n    assert st.session_state.cleaning_discussion_mode_by_finding == {}\\n    assert st.session_state.cleaning_review_statuses == {}\\n"]]}'

def _legacy_test_updates(root: Path) -> dict[Path, tuple[str, str]]:
    """Patch only exact obsolete expectations, never replace whole test files."""
    import json
    mappings = json.loads(MIGRATIONS_JSON)
    updates = {}
    for filename, replacements in mappings.items():
        path = root / 'tests' / filename
        if not path.is_file():
            continue
        original = path.read_text(encoding='utf-8')
        changed = original
        for before, after in replacements:
            if before and before in changed:
                changed = changed.replace(before, after, 1)
        if changed != original:
            ast.parse(changed)
            updates[path] = (original, changed)
    return updates


def install(root: Path) -> str:
    app = root / 'app.py'
    helper = root / 'src/healthcare_data_quality/ui/finding_wait_gate.py'
    if not app.is_file() or not helper.is_file():
        raise FileNotFoundError('Extract patch ZIP into the project root before installing')
    original = app.read_text(encoding='utf-8')
    changed = transform(original)
    if changed == original:
        return 'Already installed. No changes made.'
    revisions = {app: (original, changed), **_legacy_test_updates(root)}
    backups = {path: path.with_name(path.name + '.pre_finding_gate.bak') for path in revisions}
    for backup in backups.values():
        if backup.exists():
            raise FileExistsError(f'Refusing to overwrite backup: {backup}')
    for path, (previous, updated) in revisions.items():
        backups[path].write_text(previous, encoding='utf-8')
        path.write_text(updated, encoding='utf-8')
    return (f'Installed per-finding suggestion gate; migrated {len(revisions)-1} '
            f'legacy test file(s) with exact-matching edits. Backups created.')


if __name__ == '__main__':
    print(install(Path.cwd()))
