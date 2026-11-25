"""Status indicators remain near the finding title; only real work animates."""
from html import unescape

from healthcare_data_quality.ui.finding_wait_gate import waiting_card_html


def test_waiting_runner_is_between_chevron_and_finding_title():
    for status, figure, motion in (("pending", "◷", "static"), ("queued", "◷", "static"),
                                   ("running", "↻", "spin"), ("revision", "↻", "spin"),
                                   ("ordered", "🔒", "static")):
        html = unescape(waiting_card_html("Finding 4", "Duplicated taxonomy", status))
        assert html.count(f'>{figure}</span>') == 1
        assert (html.index('class="hdq-finding-wait-lock"')
                < html.index('class="hdq-finding-wait-figure"')
                < html.index('class="hdq-finding-wait-main"')
                < html.index("Finding 4 · Duplicated taxonomy"))
        assert f'data-motion="{motion}" role="img"' in html
        assert "hdq-finding-spin" in html


def test_check_runner_is_also_before_title():
    html = waiting_card_html("Gemma check 1", "Blank control columns")
    assert html.index('class="hdq-finding-wait-figure"') < html.index("Gemma check 1 ·")


def test_error_symbol_remains_near_title_not_far_right():
    html = waiting_card_html("Finding 2", "Missing counts", "failed")
    assert "🏃‍♂️" not in html
    assert html.index('class="hdq-finding-wait-failure"') < html.index("Finding 2 ·")
    assert html.count('>⚠</span>') == 1


def test_html_escapes_finding_title_and_has_no_interaction():
    html = waiting_card_html("Finding 1", '<script>&', 'queued')
    assert '&lt;script&gt;&amp;' in html
    assert '<script>' not in html
    assert '<button' not in html
    assert '<details' not in html
