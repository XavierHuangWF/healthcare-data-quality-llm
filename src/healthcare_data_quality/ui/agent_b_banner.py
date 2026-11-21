"""Data-bound, read-only status panel for on-demand Gemma B conversations.

The panel never invokes a model, approves a proposal, or executes R. It shows
only conversation state already visible in this Streamlit session.
"""
from __future__ import annotations

from html import escape
from typing import Mapping


def agent_b_status_snapshot(session: Mapping | None, *, active_context: str | None,
                            context_key: str | None) -> dict[str, int | str]:
    """Report counts only for the active audit, never a previous dataset's chat."""
    state = session if isinstance(session, Mapping) else {}
    blank = {"pending": 0, "discussed": 0, "plans": 0, "applied": 0}
    if not context_key or active_context != context_key:
        return {"state": "waiting" if not context_key else "ready", **blank}
    pending = state.get("cleaning_pending_gemma_b") or {}
    threads = state.get("cleaning_messages_by_finding") or {}
    plans = state.get("cleaning_plans_by_finding") or {}
    history = state.get("cleaning_history") or []
    if not isinstance(pending, dict):
        pending = {}
    if not isinstance(threads, dict):
        threads = {}
    if not isinstance(plans, dict):
        plans = {}
    if not isinstance(history, list):
        history = []
    discussed = sum(1 for turns in threads.values()
                    if isinstance(turns, list) and any(
                        isinstance(turn, dict) and turn.get("role") == "user"
                        for turn in turns))
    applied = len({item.get("finding_id") for item in history
                   if isinstance(item, dict) and item.get("finding_id")})
    return {"state": "responding" if pending else ("follow_up" if discussed else "ready"),
            "pending": len(pending), "discussed": discussed,
            "plans": len(plans), "applied": applied}


def render_agent_b_banner_html(status: Mapping | None) -> str:
    """Agent B is an on-demand assistant, not a batch job with a progress %."""
    data = status if isinstance(status, Mapping) else {}
    state = str(data.get("state", "waiting"))
    if state not in {"waiting", "ready", "responding", "follow_up"}:
        raise ValueError("Invalid Agent B state")
    counts = {}
    for name in ("pending", "discussed", "plans", "applied"):
        number = data.get(name, 0)
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ValueError("Agent B counts must be nonnegative integers")
        counts[name] = number
    pending, discussed = counts["pending"], counts["discussed"]
    if state == "waiting":
        title = "Agent B — waiting for findings"
        activity = "Ready after baseline QC results are available"
        note = "No conversation has started"
    elif state == "responding":
        title = "Agent B — responding…"
        activity = f"{pending} finding conversation{'s' if pending != 1 else ''} in progress"
        note = "Your message remains visible in the finding"
    elif state == "follow_up":
        title = "Agent B — available for follow-up"
        activity = f"{discussed} finding{'s' if discussed != 1 else ''} discussed"
        note = "Continue the discussion or review the proposed changes"
    else:
        title = "Agent B — ready for questions"
        activity = "Open a finding and ask about its cleaning options"
        note = "No data changes without separate approval and Execute"
    if counts["plans"] or counts["applied"]:
        note = (f"{counts['plans']} draft plan(s) · "
                f"{counts['applied']} finding(s) cleaned after authorization")
    # A small original robot illustration in SVG, not a remote image or script.
    robot = '''<svg viewBox="0 0 104 104" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="hdq-bot" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#a3efff"/><stop offset="1" stop-color="#3c80e5"/></linearGradient></defs>
<path d="M24 23Q13 23 13 35V63Q13 76 26 76H32L42 87V76H77Q90 76 90 63V35Q90 23 78 23Z" fill="#11395e" stroke="#82d8ff" stroke-width="3"/>
<rect x="27" y="34" width="51" height="30" rx="13" fill="url(#hdq-bot)"/>
<circle cx="42" cy="49" r="5" fill="#0b2d52"/><circle cx="63" cy="49" r="5" fill="#0b2d52"/>
<path d="M45 59Q52 63 60 59" stroke="#0b2d52" stroke-width="3" fill="none" stroke-linecap="round"/>
<path d="M52 23V16" stroke="#7ddfff" stroke-width="3" stroke-linecap="round"/><circle cx="52" cy="12" r="5" fill="#ffda67"/>
</svg>'''
    return f'''<!doctype html><html><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
*{{box-sizing:border-box}}html,body{{margin:0;background:transparent;color:#fff;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Arial,sans-serif}}
.card{{display:grid;grid-template-columns:minmax(0,1fr) 88px;gap:5px;align-items:center;padding:9px 12px;min-height:104px;border:1px solid #7b95f8;border-radius:14px;background:linear-gradient(110deg,#15233f,#202d59 75%,#123d67);box-shadow:0 2px 11px #3962dd33;overflow:hidden}}
.title{{font-size:18px;font-weight:800;line-height:1.2;letter-spacing:-.025em}}
.sub{{font-size:11px;color:#cbd9ff;margin:4px 0 5px}}
.activity{{font-size:12px;color:white;line-height:1.25}}
.note{{font-size:10px;color:#cbd9ef;line-height:1.25;margin-top:5px}}
.bot{{width:80px;height:80px;display:grid;place-items:center}}.bot svg{{width:100%;height:100%;filter:drop-shadow(0 0 8px #65baff77)}}
.card.is-active .bot svg{{animation:hdq-reply .95s ease-in-out infinite alternate}}
@keyframes hdq-reply{{from{{transform:translateY(2px) rotate(-1deg)}}to{{transform:translateY(-4px) rotate(1deg)}}}}
@media(prefers-reduced-motion:reduce){{.card.is-active .bot svg{{animation:none}}}}
@media(max-width:550px){{.card{{grid-template-columns:minmax(0,1fr) 57px;min-height:88px;padding:8px}}.title{{font-size:15px}}.bot{{width:54px;height:54px}}}}
</style></head><body>
<section class="card{' is-active' if pending else ''}" role="status" aria-label="Gemma B conversation status">
<div><div class="title">{escape(title)}</div>
<div class="sub">Interactive finding review · user-approved R cleaning</div>
<div class="activity">{escape(activity)}</div>
<div class="note">{escape(note)}</div></div>
<div class="bot">{robot}</div>
</section></body></html>'''
