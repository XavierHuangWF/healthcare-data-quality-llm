"""Offline, read-only progress feedback while the first recommendation is pending.

Elapsed time is derived from the current recommendation queue's created_at.
The UI polls queue metadata, never makes a model call or authorizes cleaning.
"""
from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Mapping


def _started_epoch_ms(snapshot: Mapping | None) -> int | None:
    """Return the real queue start, not the time this widget was mounted."""
    if not isinstance(snapshot, Mapping):
        return None
    raw = snapshot.get("created_at")
    if not isinstance(raw, str):
        return None
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if value.tzinfo is None:
            return None
        return max(0, int(value.timestamp() * 1000))
    except (ValueError, OverflowError):
        return None


def recommendation_wait_state(snapshot: Mapping | None) -> dict:
    """State is derived exclusively from statuses saved for this investigation."""
    entries = (snapshot or {}).get("entries") if isinstance(snapshot, Mapping) else None
    entries = entries if isinstance(entries, Mapping) else {}
    counts = {name: 0 for name in ("pending", "running", "ready", "failed")}
    for entry in entries.values():
        if isinstance(entry, Mapping) and entry.get("status") in counts:
            counts[entry["status"]] += 1
    total = sum(counts.values())
    if total == 0:
        title, detail, status = (
            "Preparing the recommendation queue", "Waiting for verified findings to reach Agent B", "preparing")
    elif counts["running"]:
        title, detail, status = (
            "Agent B is generating suggestions", "The model is reviewing current findings and relevant saved methods", "generating")
    elif counts["pending"]:
        title, detail, status = (
            "Agent B is processing the queue", "Suggestions are queued for model review", "queued")
    elif counts["ready"]:
        title, detail, status = (
            "A recommendation is ready", "Opening the cleaning workspace", "ready")
    else:
        title, detail, status = (
            "Suggestions could not be generated", "No verified suggestion is ready; inspect Agent B's failure details", "failed")
    return {
        **counts, "total": total, "processed": counts["ready"] + counts["failed"],
        "title": title, "detail": detail, "status": status,
        "started_epoch_ms": _started_epoch_ms(snapshot),
    }


# HDQ_AGENT_B_JOGGER_V1
def recommendation_wait_html(snapshot: Mapping | None) -> str:
    """Generate a compact, self-ticking status bar, with no patient data or URLs.

    Only trusted internal state contributes text. The one-second animation and
    clock are browser-local; model processing and job polling happen elsewhere.
    """
    state = recommendation_wait_state(snapshot)
    started = state["started_epoch_ms"]
    safe_started = str(started) if started is not None else "null"
    title = escape(state["title"])
    detail = escape(state["detail"])
    count = state["total"]
    counts = (f"{state['ready']} ready · {state['running']} generating · "
              f"{state['pending']} queued · {state['failed']} failed" if count else
              "Waiting for the recommendation job to start")
    progress = (100 * state["processed"] / count) if count else 0
    if count:
        total_label = f"{state['processed']} of {count} processed"
    else:
        total_label = "No recommendation tasks received yet"
    end_class = " ended" if state["status"] in {"ready", "failed"} else ""
    return f'''<!doctype html><html><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/><style>
*{{box-sizing:border-box}}
html,body{{margin:0;padding:0;background:transparent;color:#e9f4ff;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}}
.panel{{background:#142c40;border:1px solid #285e8a;border-radius:12px;padding:13px 16px;min-height:122px}}
.top{{display:flex;align-items:center;justify-content:space-between;gap:14px}}
.identity{{display:flex;gap:11px;align-items:center;min-width:0}}
.spinner{{width:20px;height:20px;flex:none;border:3px solid #2b7199;border-top-color:#4ce6ca;border-radius:50%;animation:spin 1s linear infinite}}
.ended .spinner{{animation:none;border-color:#4ce6ca}}
.hdq-jogger-icon{{display:inline-block;font-size:26px;line-height:1;flex:none;
 transform-origin:50% 85%;animation:hdq-jog 650ms ease-in-out infinite alternate}}
.ended .hdq-jogger-icon{{animation:none}}
@keyframes hdq-jog{{from{{transform:translate(0,2px) rotate(-8deg)}}
 to{{transform:translate(6px,-3px) rotate(8deg)}}}}

.title{{font-size:15px;font-weight:750;line-height:1.3}}
.detail{{font-size:12px;color:#b5cbd9;margin:3px 0 0}}
.timer{{flex:none;white-space:nowrap;font-variant-numeric:tabular-nums;font-size:18px;font-weight:750;color:#e9f4ff}}
.timer span{{display:block;font-size:10px;font-weight:500;color:#93b3c4;letter-spacing:.04em;text-align:right}}
.progress{{height:6px;border-radius:999px;background:#244158;overflow:hidden;margin:14px 0 7px}}
.fill{{height:100%;width:{progress:.2f}%;background:#4ce6ca;transition:width .25s linear;border-radius:999px}}
.meta{{display:flex;justify-content:space-between;gap:12px;font-size:11px;color:#bbd1df;flex-wrap:wrap}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}
@media(prefers-reduced-motion:reduce){{.spinner{{animation:none}}.fill{{transition:none}}.hdq-jogger-icon{{animation:none}}}}
@media(max-width:550px){{.panel{{padding:10px}}.timer{{font-size:15px}}.title{{font-size:13px}}}}
</style></head><body>
<section class="panel{end_class}" role="status" aria-label="Agent B recommendation processing status">
  <div class="top"><div class="identity"><span class="hdq-jogger-icon" role="img" aria-label="Agent B is working">🏃‍♂️</span>
    <div><div class="title">{title}</div><div class="detail">{detail}</div></div></div>
    <div class="timer"><span>ELAPSED</span><strong id="hdq-elapsed">{'—' if started is None else '00:00'}</strong></div>
  </div>
  <div class="progress" role="progressbar" aria-label="Recommendation tasks processed" aria-valuemin="0" aria-valuemax="{count}" aria-valuenow="{state['processed']}"><div class="fill"></div></div>
  <div class="meta"><span>{escape(counts)}</span><span>{escape(total_label)} · no ETA</span></div>
</section><script>
const start = {safe_started};
const dest = document.getElementById('hdq-elapsed');
function updateClock() {{
 if (start === null) return;
 const s = Math.max(0, Math.floor((Date.now()-start)/1000));
 const pad = n => String(n).padStart(2,'0');
 const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), r = s%60;
 dest.textContent = h ? `${{pad(h)}}:${{pad(m)}}:${{pad(r)}}` : `${{pad(m)}}:${{pad(r)}}`;
}}
updateClock(); window.setInterval(updateClock, 1000);
</script></body></html>'''


def show_recommendation_wait_panel(st, audit: Mapping | None, context_key: str) -> None:
    """Read current Agent B queue and render its *actual* progress in-place.

    A separate Streamlit fragment calls this every three seconds. The timer
    inside the embedded HTML ticks every second without Streamlit reruns.
    """
    from healthcare_data_quality.jobs.cleaning_jobs import (
        recommendation_manager, job_fingerprint,
    )

    snapshot = None
    run_id = (audit or {}).get('cleaning_recommendation_run_id') if isinstance(audit, Mapping) else None
    if isinstance(run_id, str) and run_id:
        try:
            snapshot = recommendation_manager().snapshot(
                job_fingerprint(context_key, [], run_id=run_id))
        except (OSError, ValueError, TypeError, KeyError):
            pass
    st.iframe(recommendation_wait_html(snapshot), height=134)
    status = recommendation_wait_state(snapshot)
    if status['total'] and status['failed'] == status['total']:
        st.error('Agent B could not generate a suggestion. Check its failed-job details; no cleaning was authorized.')
    elif status['failed']:
        st.caption(f"{status['failed']} recommendation(s) failed. Remaining queued suggestions may still succeed.")
    else:
        st.caption('Cleaning unlocks automatically after the first verified suggestion. No ETA is available.')
