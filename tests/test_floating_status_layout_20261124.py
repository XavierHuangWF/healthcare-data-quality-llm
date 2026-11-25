"""Layout-only regression: live cards must stay in the compact floating host."""
from pathlib import Path
import shutil

import pytest

from healthcare_data_quality.ui.exploration_banner import sticky_banner_css

ROOT = Path(__file__).resolve().parents[1]


def test_live_fragment_owns_its_style_and_anchor():
    source = (ROOT / 'app.py').read_text(encoding='utf-8')
    live = source.split('def render_live_investigation_status(', 1)[1].split('def render_pre_investigation_status(', 1)[0]
    assert 'with st.container(key="hdq_investigation_banner_live"):' in live
    assert 'st.html(sticky_banner_css() + \'<span class="hdq-status-live-anchor" hidden></span>\')' in live
    assert live.index('hdq-status-live-anchor') < live.index('show_status_cards()', live.index('hdq-status-live-anchor'))


@pytest.mark.parametrize('keyed', [True, False])
@pytest.mark.parametrize('width', [1600, 600])
def test_status_host_css_in_chromium(keyed, width):
    # This isolated DOM fixture tests actual browser layout, not a live Streamlit session.
    pw = pytest.importorskip('playwright.sync_api')
    chromium = shutil.which('chromium') or shutil.which('google-chrome')
    if chromium is None:
        pytest.skip('Local Chromium is not installed')
    with pw.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium, headless=True, args=['--no-sandbox'])
        try:
            page = browser.new_page(viewport={'width': width, 'height': 900})
            cls = 'st-key-hdq_investigation_banner_live' if keyed else ''
            cards = ''.join('<div data-testid="stVerticalBlock"><iframe style="border:0;width:1400px;height:140px"></iframe></div>' for _ in range(3))
            page.set_content(f'''<html><body style="margin:0"><div data-testid="stVerticalBlock" id="main"><div style="height:120px">Preview</div><div id="host" class="{cls}" data-testid="stVerticalBlock"><div data-testid="stElementContainer">{sticky_banner_css()}<span class="hdq-status-live-anchor" hidden></span></div>{cards}</div><div id="next" style="height:1200px">Cleaning workspace</div></div></body></html>''')
            host = page.locator('#host')
            box = host.bounding_box()
            assert host.evaluate('(el) => getComputedStyle(el).position') == 'fixed'
            assert box['width'] == pytest.approx(300, abs=1)
            assert box['x'] + box['width'] == pytest.approx(width - (8 if width <= 650 else 17.6), abs=1)
            assert page.locator('#main').evaluate('(el) => getComputedStyle(el).position') != 'fixed'
            assert page.locator('#next').bounding_box()['y'] == pytest.approx(120, abs=1)
            assert host.locator('iframe').count() == 3
            assert all(x.bounding_box()['width'] == pytest.approx(300, abs=1) for x in host.locator('iframe').all())
            page.evaluate('window.scrollTo(0,400)')
            assert host.bounding_box()['y'] == pytest.approx(box['y'], abs=1)
        finally:
            browser.close()
