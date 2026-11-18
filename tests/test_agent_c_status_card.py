from exploration_banner import render_agent_c_banner_html


def test_agent_c_idle_card_mentions_waiting_state():
    html = render_agent_c_banner_html(idle=True)
    assert "Agent C — waiting for Agent B" in html
    assert "Idle · No Agent C request running" in html
    assert "Agent C becomes available after the first recommendation is saved" in html


def test_agent_c_active_card_mentions_replying_state():
    html = render_agent_c_banner_html(ready_findings=2, total_findings=5, active_conversations=1)
    assert "Agent C — responding" in html
    assert "1 active request" in html
    assert 'class="card is-active"' in html


def test_agent_c_ready_card_mentions_review_state():
    html = render_agent_c_banner_html(ready_findings=3, total_findings=9, active_conversations=0)
    assert "Agent C — waiting for your input" in html
    assert "Idle · No Agent C request running" in html


def test_agent_c_does_not_reuse_agent_b_progress_as_its_own():
    early = render_agent_c_banner_html(ready_findings=1, total_findings=9)
    later = render_agent_c_banner_html(ready_findings=8, total_findings=9)
    assert early == later  # Agent B progress alone cannot change Agent C status.
    assert 'role="progressbar"' not in early
    assert "1 of 9" not in early and "8 of 9" not in later
    assert 'class="card is-active"' not in early


def test_agent_c_working_status_depends_on_submitted_request():
    idle = render_agent_c_banner_html(ready_findings=7, total_findings=9)
    working = render_agent_c_banner_html(ready_findings=7, total_findings=9, active_conversations=1)
    assert "No Agent C request running" in idle
    assert "1 active request" in working
    assert 'class="card is-active"' in working
    assert 'class="card is-active"' not in idle
    assert 'role="progressbar"' not in working
