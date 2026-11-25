"""Regression for the *real* dynamic Mark-reviewed Streamlit controls.

Earlier CSS-only/AST patch installers tested a literal button that is no longer
used by the live UI. This targets the code the reviewer actually clicks.
"""
from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / 'app.py'
SOURCE = APP.read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)
FUNCS = {item.name: item for item in TREE.body if isinstance(item, ast.FunctionDef)}


def _source(name: str) -> str:
    return ast.get_source_segment(SOURCE, FUNCS[name]) or ''


def _buttons(name: str) -> list[ast.Call]:
    return [node for node in ast.walk(FUNCS[name])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.value.__class__ is ast.Name
            and node.func.value.id == 'st' and node.func.attr == 'button']


def _kw(call: ast.Call, key: str):
    return next((item.value for item in call.keywords if item.arg == key), None)


def test_universal_dynamic_review_closure_is_secondary_not_theme_red():
    buttons = _buttons('_render_bottom_review_closure')
    actual = [x for x in buttons
              if (value := _kw(x, 'key')) is not None
              and 'review_close_any_' in ast.unparse(value)]
    assert len(actual) == 1
    button = actual[0]
    assert isinstance(button.args[0], ast.Name) and button.args[0].id == 'close_label'
    assert ast.literal_eval(_kw(button, 'type')) == 'secondary'
    assert _kw(button, 'disabled') is not None
    assert SOURCE.count("key=f'review_close_any_{fid}'") == 1


def test_css_is_colocated_with_real_closure_even_in_fragment_reruns():
    source = _source('_render_bottom_review_closure')
    assert 'st.markdown(_blue_review_closure_css(), unsafe_allow_html=True)' in source
    assert source.index('st.markdown(_blue_review_closure_css()') < source.index("key=f'review_close_any_{fid}'")
    assert '#2563eb' in _source('_blue_review_closure_css')
    assert 'st-key-review_close_any_' in _source('_blue_review_closure_css')


def test_existing_fallback_close_is_blue_instead_of_primary_red():
    buttons = _buttons('_render_prior_approval_closure')
    chosen = [x for x in buttons if x.args and isinstance(x.args[0], ast.Constant)
              and str(x.args[0].value).startswith('Mark reviewed')]
    assert len(chosen) == 1
    assert ast.literal_eval(_kw(chosen[0], 'type')) == 'secondary'
    assert 'st-key-review_resolution_model_fallback_' in _source('_blue_review_closure_css')
    # The previous CSS had an extra underscore that meant no actual key matched.
    assert 'st-key-review_resolution_model_fallback__' not in SOURCE


def test_close_css_does_not_style_approval_or_discuss_buttons():
    css_method = _source('_blue_review_closure_css')
    assert 'st-key-review_yes_' not in css_method
    assert 'st-key-review_no_' not in css_method
    assert '#1d4ed8' in css_method
    assert '#93c5fd' in css_method
    assert 'type=\'secondary\'' in _source('_render_bottom_review_closure')
    general_css = _source('_review_action_button_css')
    assert '#15803d' in general_css
    assert '#b91c1c' in general_css


def test_mismatched_stage_can_be_dismissed_without_deleting_saved_approvals():
    renderer = _source('render_cleaning_workspace')
    assert 'cleaning_skip_incompatible_checkpoint' in renderer
    assert 'cleaning_recovery_dismissed = True' in renderer
    assert 'archive_legacy_stage(' in renderer  # separate, user-directed legacy flow
