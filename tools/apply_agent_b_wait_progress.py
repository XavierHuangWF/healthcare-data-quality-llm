"""Safe, idempotent installer for the Agent B waiting status bar.

Run from the healthcare-data-quality-llm project root:
    python tools/apply_agent_b_wait_progress.py

Unlike a replacement app.py, this changes only the existing one-line waiting
message. It refuses ambiguous or unrecognized versions, so other UI work is
never silently overwritten. A .bak file is created before the first edit.
"""
from __future__ import annotations

import re
from pathlib import Path

PREFIX = "# HDQ_AGENT_B_WAIT_PROGRESS_V1"
SENTENCE = "Waiting for Agent B"
OLD_SENTENCE = "Waiting for the first Agent B"


def integrate(source: str) -> str:
    if PREFIX in source:
        return source  # idempotent
    candidates = []
    lines = source.splitlines(keepends=True)
    # Only replace one plain Streamlit waiting notice (not a condition,
    # docstring, or a progress event). Preserve its indentation and the rest
    # of the existing UI, including all newer navigation controls.
    for i, line in enumerate(lines):
        if (SENTENCE not in line and OLD_SENTENCE not in line):
            continue
        if re.match(r'^\s*st\.(?:info|caption|markdown|write)\s*\(', line) is None:
            continue
        if re.search(r'\bst\.(?:info|caption|markdown|write)\s*\(\s*["\']', line) is None:
            continue
        if ')' not in line or not line.rstrip().endswith(')'):
            continue  # Do not rewrite multiline calls or compound statements
        candidates.append(i)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one plain Agent B waiting notice; found {len(candidates)}. "
            "app.py was not modified. Send the current waiting-section code "
            "for a targeted update; do not overwrite app.py with an older ZIP."
        )
    index = candidates[0]
    indent = re.match(r'^(\s*)', lines[index]).group(1)
    # Replaces just the informative notice with a read-only three-second
    # fragment. A separate existing fragment still owns unlocking the tab.
    added = (
        f'{indent}{PREFIX}\n'
        f'{indent}def _show_hdq_wait_status():\n'
        f'{indent}    from healthcare_data_quality.ui.recommendation_wait import show_recommendation_wait_panel\n'
        f'{indent}    show_recommendation_wait_panel(st, st.session_state.get("audit_result") or audit, cache_key)\n'
        f'{indent}if hasattr(st, "fragment"):\n'
        f'{indent}    st.fragment(run_every="3s")(_show_hdq_wait_status)()\n'
        f'{indent}else:\n'
        f'{indent}    _show_hdq_wait_status()\n'
    )
    lines[index] = added
    return ''.join(lines)


def install(root: Path) -> str:
    app = root / 'app.py'
    module = root / 'src/healthcare_data_quality/ui/recommendation_wait.py'
    if not app.is_file() or not module.is_file():
        raise FileNotFoundError('Run this script in the project root after copying the ZIP files')
    before = app.read_text(encoding='utf-8')
    after = integrate(before)
    if before == after:
        return 'Already installed; no files changed.'
    # Parse syntax before replacing, preserving the original exactly on error.
    compile(after, str(app), 'exec')
    backup = root / 'app.py.agent_b_wait.bak'
    if backup.exists():
        raise FileExistsError('Backup already exists; rename it before installing a different version')
    backup.write_text(before, encoding='utf-8')
    app.write_text(after, encoding='utf-8')
    return f'Installed in {app}; backup: {backup}'


if __name__ == '__main__':
    print(install(Path.cwd()))
