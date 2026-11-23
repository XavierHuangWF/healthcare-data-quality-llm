"""Install the jogging icon without overwriting app.py or newer navigation.

Run from the project root:
    python tools/apply_agent_b_jogging_icon.py

Handles an original st.info waiting notice and the newer Agent B progress panel.
Will not silently rewrite unrecognized code. Keeps one backup per changed file.
"""
from __future__ import annotations

import ast
from pathlib import Path

MARKER = '# HDQ_AGENT_B_JOGGER_V1'
RUNNER_IMPORT = 'from healthcare_data_quality.ui.running_indicator import show_waiting_runner'


def update_notice(source: str) -> tuple[str, bool]:
    """Replace exactly one standalone text notice if present."""
    if MARKER in source:
        return source, False
    tree = ast.parse(source)
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Attribute) or call.func.attr not in ('info', 'markdown', 'write', 'caption'):
            continue
        if not (isinstance(call.func.value, ast.Name) and call.func.value.id == 'st'):
            continue
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str):
            continue
        message = call.args[0].value
        if not (message.startswith('Waiting for Agent B') or message.startswith('Waiting for the first Agent B')):
            continue
        if 'suggestion' not in message.lower():
            continue
        if node.lineno != node.end_lineno:
            continue  # never rewrite multiline/compound expressions
        line = source.splitlines()[node.lineno - 1]
        if not line.lstrip().startswith('st.'):
            continue
        matches.append((node.lineno, message))
    if len(matches) > 1:
        raise RuntimeError(f'Found {len(matches)} candidate waiting notices: no changes made')
    if not matches:
        return source, False
    index, message = matches[0]
    lines = source.splitlines(keepends=True)
    indent = lines[index - 1][:len(lines[index - 1]) - len(lines[index - 1].lstrip())]
    lines[index - 1] = (
        f'{indent}{MARKER}\n'
        f'{indent}{RUNNER_IMPORT}\n'
        f'{indent}show_waiting_runner(st, {message!r})\n'
    )
    revised = ''.join(lines)
    compile(revised, '<app.py>', 'exec')
    return revised, True


def update_existing_progress_panel(source: str) -> tuple[str, bool]:
    """Upgrade prior Agent B spinner panel to a jogging emoji (no JS changes)."""
    if MARKER in source:
        return source, False
    anchor = '<span class="spinner" aria-hidden="true"></span>'
    if anchor not in source:
        return source, False
    src = source.replace(
        anchor,
        '<span class="hdq-jogger-icon" role="img" aria-label="Agent B is working">🏃‍♂️</span>',
        1,
    )
    css_anchor = '.ended .spinner{{animation:none;border-color:#4ce6ca}}'
    if css_anchor not in src:
        raise RuntimeError('Unrecognized Agent B panel CSS: no changes made')
    src = src.replace(css_anchor, css_anchor + '''
.hdq-jogger-icon{{display:inline-block;font-size:26px;line-height:1;flex:none;
 transform-origin:50% 85%;animation:hdq-jog 650ms ease-in-out infinite alternate}}
.ended .hdq-jogger-icon{{animation:none}}
@keyframes hdq-jog{{from{{transform:translate(0,2px) rotate(-8deg)}}
 to{{transform:translate(6px,-3px) rotate(8deg)}}}}
''', 1)
    reduce_motion = '@media(prefers-reduced-motion:reduce){{.spinner{{animation:none}}.fill{{transition:none}}}}'
    if reduce_motion not in src:
        raise RuntimeError('Unrecognized Agent B reduced-motion CSS: no changes made')
    src = src.replace(reduce_motion, reduce_motion.replace('.fill{{transition:none}}', '.fill{{transition:none}}.hdq-jogger-icon{{animation:none}}'), 1)
    src = src.replace('def recommendation_wait_html(', MARKER + '\ndef recommendation_wait_html(', 1)
    compile(src, '<recommendation_wait.py>', 'exec')
    return src, True


def install(root: Path) -> str:
    app = root / 'app.py'
    widget = root / 'src/healthcare_data_quality/ui/recommendation_wait.py'
    helper = root / 'src/healthcare_data_quality/ui/running_indicator.py'
    if not app.is_file() or not helper.is_file():
        raise FileNotFoundError('Extract the ZIP into the project root first')
    original_app = app.read_text(encoding='utf-8')
    new_app, patched_notice = update_notice(original_app)
    original_widget = widget.read_text(encoding='utf-8') if widget.exists() else None
    new_widget, patched_panel = update_existing_progress_panel(original_widget) if original_widget is not None else (None, False)
    if not patched_notice and not patched_panel:
        if MARKER in original_app or (original_widget is not None and MARKER in original_widget):
            return 'Already installed; no files changed.'
        raise RuntimeError('No recognized Agent B waiting message or progress panel found. No files changed.')
    changes = [(app, original_app, new_app)] if patched_notice else []
    if patched_panel:
        changes.append((widget, original_widget, new_widget))
    for path, _old, _new in changes:
        backup = path.with_name(path.name + '.pre_jogger.bak')
        if backup.exists():
            raise FileExistsError(f'Backup already exists: {backup}. No files changed.')
    for path, old, new in changes:
        backup = path.with_name(path.name + '.pre_jogger.bak')
        backup.write_text(old, encoding='utf-8')
        path.write_text(new, encoding='utf-8')
    return f'Installed running-person indicator in: {", ".join(str(x[0]) for x in changes)}'


if __name__ == '__main__':
    print(install(Path.cwd()))
