"""Per-finding Agent B readiness UI, independent of the page navigation.

The clean-data page stays navigable; *individual* findings cannot expose review
controls until their corresponding model result has been delivered for this run.
This module creates no inference jobs, performs no cleaning and grants no approval.
"""
from __future__ import annotations

from html import escape
from typing import Any, Mapping


def state_by_finding(snapshot: Mapping[str, Any] | None) -> dict[str, str]:
    """Report the current state of each *nonhistorical* recommendation."""
    results: dict[str, str] = {}
    entries = (snapshot or {}).get("entries") or {}
    if not isinstance(entries, dict):
        return results
    for fid, entry in entries.items():
        if not isinstance(entry, dict) or entry.get("origin") == "review_memory":
            continue
        status = entry.get("status")
        if status in {"ready", "pending", "queued", "running", "failed"}:
            results[str(fid)] = str(status)
    return results


def is_locked(*, finding_id: str, snapshot: Mapping[str, Any] | None,
              completed: bool, active_conversation: bool) -> bool:
    """A finished review or already-started discussion must never be hidden."""
    if completed or active_conversation:
        return False
    return state_by_finding(snapshot).get(finding_id) not in {"ready", "failed"}


def waiting_card_html(label: str, title: str, status: str = "queued") -> str:
    """Accessible non-expandable card with a CSS-only animated jogger."""
    failed = status == "failed"
    state_text = (
        "Suggestion failed — check Agent B retry details above" if failed
        else "Agent B is preparing a suggestion" if status == "running"
        else "Waiting for Agent B's verified suggestion"
    )
    # No input, button, or details element: impossible to expand while locked.
    figure = "⚠" if failed else "🏃‍♂️"
    icon_class = "hdq-finding-wait-figure" if not failed else "hdq-finding-wait-failure"
    return f'''<style>
.hdq-finding-wait{{box-sizing:border-box;width:100%;background:#191c24;
  border:1px solid #343845;border-radius:7px;margin:8px 0 10px;
  padding:10px 14px;display:flex;align-items:center;gap:12px;
  color:#eef1f7;font:inherit;line-height:1.4}}
.hdq-finding-wait-main{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;
  min-width:0;flex:1}}
.hdq-finding-wait-label{{font-weight:600;overflow-wrap:anywhere}}
.hdq-finding-wait-status{{color:#91aec8;font-size:0.86em}}
.hdq-finding-wait-lock{{color:#a5afc0;flex:none}}
.hdq-finding-wait-figure{{font-size:23px;display:inline-block;flex:none;
  transform-origin:50% 85%;animation:hdq-finding-jog .62s ease-in-out infinite alternate}}
.hdq-finding-wait-failure{{font-size:20px;color:#ff9797;flex:none}}
@keyframes hdq-finding-jog{{from{{transform:translate(0,2px) rotate(-9deg)}}
 to{{transform:translate(5px,-3px) rotate(9deg)}}}}
@media(prefers-reduced-motion:reduce){{.hdq-finding-wait-figure{{animation:none}}}}
</style>
<div class="hdq-finding-wait" role="status" aria-live="off"
 aria-label="{escape(label)}: {escape(title)}. {escape(state_text)}">
 <span class="hdq-finding-wait-lock" aria-hidden="true">▸</span>
 <span class="{icon_class}" role="img" aria-label="{'Retry required' if failed else 'Agent B working'}">{figure}</span>
 <span class="hdq-finding-wait-main">
  <span class="hdq-finding-wait-label">{escape(label)} · {escape(title)}</span>
  <span class="hdq-finding-wait-status">{escape(state_text)}</span>
 </span>
</div>'''


def render_waiting_card(st: Any, label: str, title: str, status: str = "queued") -> None:
    st.markdown(waiting_card_html(label, title, status), unsafe_allow_html=True)


def should_refresh_for_new_results(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
) -> bool:
    """Refresh only when any finding has progressed to ready or failed."""
    previous = state_by_finding(before)
    current = state_by_finding(after)
    return any(status in {"ready", "failed"} and previous.get(fid) != status
               for fid, status in current.items())


def poll_finding_readiness(st: Any, job_id: str | None,
                           rendered_snapshot: dict) -> None:
    """A single timer for all waiting cards; never renders their buttons."""
    if not job_id:
        return
    from healthcare_data_quality.jobs.cleaning_jobs import recommendation_manager
    try:
        latest = recommendation_manager().snapshot(job_id)
    except (OSError, ValueError, KeyError, TypeError):
        return
    if not should_refresh_for_new_results(rendered_snapshot, latest):
        return
    # A live discussion, an approval, or an unsent draft must not be remounted
    # while the user is typing. Their next normal interaction refreshes cards.
    if (st.session_state.get("cleaning_pending_gemma_b")
            or st.session_state.get("cleaning_plans_by_finding")
            or any(mode in {"alternative", "accepted"}
                   for mode in (st.session_state.get(
                       "cleaning_discussion_mode_by_finding") or {}).values())):
        return
    st.rerun(scope="app")
