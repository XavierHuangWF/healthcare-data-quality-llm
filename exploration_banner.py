"""Responsive, data-bound investigation hero for Gemma phases 1–3.

Artwork is a bundled, original mascot; progress/activity are never baked into
artwork. No external assets or network calls. The same banner is used while
Phase 2's registered-tool HTTP request blocks the Streamlit Python thread and
while Phase 3's exploration runs asynchronously.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from functools import lru_cache
from html import escape
from pathlib import Path

MASCOT = Path(__file__).resolve().parent / "assets" / "gemma_electric_explorer.webp"
IDLE_MASCOT = Path(__file__).resolve().parent / "assets" / "gemma_agent_a_idle_mascot.png"
RECOMMENDATION_MASCOT = Path(__file__).resolve().parent / "assets" / "gemma_agent_a_green_dino.png"
RECOMMENDATION_PROCESSING_MASCOT = Path(__file__).resolve().parent / "assets" / "gemma_agent_b_processing_dino.webp"
AGENT_C_IDLE_MASCOT = Path(__file__).resolve().parent / "assets" / "gemma_agent_c_idle.png"
AGENT_C_WORKING_MASCOT = Path(__file__).resolve().parent / "assets" / "gemma_agent_c_working.png"


def _local_mascot_data_url(path: Path) -> str:
    """Return bundled artwork or a safe local fallback if assets were not copied.

    The project intentionally ignores assets in Git. Missing images must not
    interrupt an investigation or a review just because the UI is cloned.
    """
    if path.is_file():
        mime = 'image/webp' if path.suffix.lower() == '.webp' else 'image/png'
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode('ascii')
    placeholder = (b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
                   b'<circle cx="50" cy="50" r="42" fill="#dbeafe"/>'
                   b'<rect x="24" y="35" width="52" height="36" rx="12" fill="#2563eb"/>'
                   b'<circle cx="40" cy="51" r="5" fill="white"/>'
                   b'<circle cx="60" cy="51" r="5" fill="white"/></svg>')
    return 'data:image/svg+xml;base64,' + base64.b64encode(placeholder).decode('ascii')


@lru_cache(maxsize=1)
def mascot_data_url() -> str:
    return _local_mascot_data_url(MASCOT)


@lru_cache(maxsize=1)
def idle_mascot_data_url() -> str:
    """Calm captain mascot for waiting and finished investigation states."""
    return _local_mascot_data_url(IDLE_MASCOT)


@lru_cache(maxsize=1)
def agent_c_idle_mascot_data_url() -> str:
    return _local_mascot_data_url(AGENT_C_IDLE_MASCOT)


@lru_cache(maxsize=1)
def agent_c_working_mascot_data_url() -> str:
    return _local_mascot_data_url(AGENT_C_WORKING_MASCOT)


@lru_cache(maxsize=1)
def recommendation_mascot_data_url() -> str:
    """Agent B mascot for waiting, finished, or failed states."""
    return _local_mascot_data_url(RECOMMENDATION_MASCOT)


@lru_cache(maxsize=1)
def recommendation_processing_mascot_data_url() -> str:
    return _local_mascot_data_url(RECOMMENDATION_PROCESSING_MASCOT)


def exploration_elapsed(started_at: object, *, now: datetime | None = None) -> str:
    if not isinstance(started_at, str):
        return "—"
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        if start.tzinfo is None:
            return "—"
        reference = now or datetime.now(timezone.utc)
        seconds = max(0, int((reference - start).total_seconds()))
    except (ValueError, TypeError, OverflowError):
        return "—"
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def _iso_to_epoch_ms(started_at: object) -> int | None:
    if not isinstance(started_at, str):
        return None
    try:
        parsed = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        return max(0, int(parsed.timestamp() * 1000))
    except (ValueError, OverflowError, TypeError):
        return None


def _banner(*, phase: int, phase_count: int, activity: str,
            verified_checks: int, elapsed: str, start_epoch_ms: int | None,
            finished_state: str | None = None,
            idle_state: str | None = None) -> str:
    if phase not in (1, 2, 3) or phase_count not in (2, 3) or phase > phase_count:
        raise ValueError("Invalid investigation phase")
    if isinstance(verified_checks, bool) or not isinstance(verified_checks, int) or verified_checks < 0:
        raise ValueError("Verified checks must be nonnegative")
    labels = {
        1: ("Agent A — checking data quality…", "Running deterministic checks", "Data quality checks"),
        2: ("Agent A — investigating…", "Collecting evidence with registered tools", "Gemma registered-tool investigation"),
        3: ("Agent A — exploring…", "Collecting evidence in the read-only sandbox", "Gemma exploratory analyses"),
    }
    headline, subtitle, phase_label = labels[phase]
    if finished_state not in (None, "review", "partial", "limited_clear"):
        raise ValueError("Unsupported investigation completion state")
    if idle_state not in (None, "no_dataset", "ready", "saved_available"):
        raise ValueError("Unsupported idle state")
    if idle_state is not None and finished_state is not None:
        raise ValueError("A run cannot be both idle and finished")
    if finished_state == "partial":
        headline, subtitle, phase_label = (
            "Agent A — review partial results",
            "Some scientific questions or evidence checks remain unresolved",
            "Completed recorded workflow phases",
        )
    elif finished_state is not None:
        headline, subtitle, phase_label = (
            "Agent A — investigation finished",
            "Recorded results are available for scientific review",
            "Completed recorded workflow phases",
        )
    elif idle_state == "no_dataset":
        headline, subtitle, phase_label = (
            "Agent A — waiting for dataset",
            "QC and Gemma checks have not started",
            "Awaiting a dataset",
        )
    elif idle_state == "saved_available":
        headline, subtitle, phase_label = (
            "Agent A — saved result available",
            "Load the saved audit or start a fresh run",
            "No investigation running",
        )
    elif idle_state == "ready":
        headline, subtitle, phase_label = (
            "Agent A — ready to investigate",
            "Runs QC, registered-tool checks and optional exploration",
            "No investigation running",
        )
    activity = escape(str(activity or "Preparing the next step")[:220], quote=True)
    elapsed = escape(elapsed)
    # Active investigation uses the electric mascot. Waiting and terminal
    # states use the separate captain mascot supplied by the researcher.
    mascot = (idle_mascot_data_url()
              if idle_state is not None or finished_state is not None
              else mascot_data_url())
    # Reflect completed *phases* only. Never claim that the bar estimates model
    # work, token generation, remaining hypotheses, or time left.
    phase_fraction = 100 if finished_state is not None else round(100 * (phase - 1) / phase_count, 2)
    metric = (f"✓ Verified exploratory checks: <b>{verified_checks}</b>" if phase == 3
              else "✓ Verified exploratory checks: <b>Not started</b>")
    if idle_state is not None:
        phase_fraction = 0
        metric = "○ Exploratory checks: <b>Not started</b>"
        if idle_state == "no_dataset":
            activity = "Upload a CSV to begin a full investigation"
        elif idle_state == "saved_available":
            activity = "Load the saved result or run a new investigation"
        else:
            activity = "Select Run full investigation to start"
    footer = ("You can review verified findings and discuss cleaning while exploration continues. "
              "New verified results will appear automatically." if phase == 3 else
              "The registered-tool investigation is still running. Findings and interactive cleaning "
              "will become available when this phase finishes." if phase == 2 else
              "Checking dataset structure and values. No uploaded data is being changed.")
    if finished_state is not None:
        activity = "Review the saved results and supporting evidence"
        footer = "Run ended; findings are not a certification that the dataset is clean."
    elif idle_state is not None:
        footer = "No investigation or recommendation generation is running."
    # The script is local and only manages the elapsed time. It never reads
    # patient data, model text, storage, or the host page. Dynamic text is escaped.
    clock_script = ""
    if start_epoch_ms is not None:
        clock_script = f"""<script>
(function() {{
  const start = {int(start_epoch_ms)};
  const target = document.getElementById('hdq-live-clock');
  function render() {{
    const secs = Math.max(0, Math.floor((Date.now() - start) / 1000));
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    const s = secs % 60;
    const pad = n => String(n).padStart(2, '0');
    target.textContent = h ? `${{h}}:${{pad(m)}}:${{pad(s)}}` : `${{pad(m)}}:${{pad(s)}}`;
  }}
  render(); window.setInterval(render, 1000);
}})();
</script>"""
    return f"""<!doctype html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
* {{box-sizing:border-box}}
html,body {{margin:0;padding:0;background:transparent;color:#fff;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Arial,sans-serif;}}
.hdq-hero {{position:relative;isolation:isolate;overflow:hidden;display:grid;align-items:center;
 grid-template-columns:minmax(0,1fr) 144px;column-gap:9px;min-height:142px;
 padding:12px 14px 12px;border:1px solid #24b8ff;border-radius:14px;
 background:linear-gradient(112deg,#0a1d38 0%,#0a2850 63%,#065165 100%);
 box-shadow:0 2px 14px #077ed444,inset 0 0 16px #1059b244;}}
.hdq-hero:before {{content:"";z-index:-1;position:absolute;inset:0;
 background:radial-gradient(ellipse at 88% 55%,#18e3bb26,transparent 55%);}}
.hdq-left {{min-width:0;z-index:2;}}
.hdq-title {{font-size:18px;font-weight:800;letter-spacing:-.025em;line-height:1.25;margin:0 0 5px;}}
.hdq-title .gold {{color:#ffd34e;}}
.hdq-sub {{font-size:12px;color:#cbecff;margin:0 0 5px;}}
.hdq-phase {{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:11px;color:#cce9ff;}}
.hdq-pill {{background:#1687ef;border-radius:20px;padding:3px 9px;color:#fff;font-size:10px;font-weight:750;}}
.hdq-track {{height:5px;border-radius:10px;background:#c8defb36;margin:7px 0;overflow:hidden;}}
.hdq-fill {{background:linear-gradient(90deg,#20d5fe,#168cf8);height:100%;width:{phase_fraction}%;border-radius:10px;}}
.hdq-metrics {{display:flex;flex-wrap:wrap;gap:4px 13px;font-size:12px;color:#d9f3ff;}}
.hdq-metrics b {{color:#fff;font-variant-numeric:tabular-nums;}}
.hdq-activity {{font-size:12px;color:#bbdfff;line-height:1.3;margin-top:6px;overflow-wrap:anywhere;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;}}
.hdq-activity b {{color:white;}}
.hdq-footnote {{display:none;}}
.hdq-right {{position:relative;display:flex;justify-content:center;align-items:center;min-width:0;height:126px;}}
.hdq-mascot {{max-width:100%;max-height:122px;object-fit:contain;
 filter:drop-shadow(0 0 10px #f7cc4677);transform-origin:center 83%;animation:hdq-bound .95s ease-in-out infinite alternate;}}
@keyframes hdq-bound {{from{{transform:translateY(3px) rotate(-1deg)}}to{{transform:translateY(-4px) rotate(1deg)}}}}
.hdq-bubble {{display:none;position:absolute;z-index:3;right:-6px;top:-2px;max-width:98px;
 background:#fff2b1;border:1px solid #ffd34e;border-radius:50%;padding:5px 7px;
 color:#14254a;text-align:center;line-height:1.15;font-size:9px;font-weight:800;
 box-shadow:0 0 10px #dfae3566;}}
.hdq-footer {{display:none;}}
@media(max-width:550px) {{.hdq-hero{{grid-template-columns:minmax(0,1fr) 78px;min-height:144px;padding:10px 10px;}}
 .hdq-right{{height:94px}}.hdq-mascot{{max-height:90px}}.hdq-bubble{{display:none}}
 .hdq-title{{font-size:17px}}.hdq-metrics{{font-size:11px;gap:3px 7px}}
 .hdq-metrics span:nth-child(2){{font-size:0}}
 .hdq-metrics span:nth-child(2)::before{{content:"✓ Checks: ";font-size:10px}}
 .hdq-metrics span:nth-child(2) b{{font-size:10px}}.hdq-footer{{font-size:9px}}}}
@media(prefers-reduced-motion:reduce) {{.hdq-mascot{{animation:none}}}}
</style></head><body>
<section class="hdq-hero" role="status" aria-label="Gemma investigation activity">
 <div class="hdq-left">
   <h2 class="hdq-title">{headline.replace('…','<span class="gold">…</span>').replace('investigating','<span class="gold">investigating</span>').replace('exploring','<span class="gold">exploring</span>')}</h2>
   <div class="hdq-sub">⚡ {subtitle}</div>
   <div class="hdq-phase"><span class="hdq-pill">{'Waiting' if idle_state is not None else 'Run ended' if finished_state is not None else f'Phase {phase} of {phase_count}'}</span>{phase_label}</div>
   <div class="hdq-track" role="img" aria-label="{'Not started' if idle_state is not None else 'Recorded run ended' if finished_state is not None else f'{phase - 1} of {phase_count} workflow phases completed'}"><div class="hdq-fill"></div></div>
   <div class="hdq-metrics"><span>{'○ Not started' if idle_state is not None else '✓ Run ended' if finished_state is not None else f'◷ Elapsed: <b id="hdq-live-clock">{elapsed}</b>'}</span><span>{metric}</span></div>
   <div class="hdq-activity">Current activity: <b>{activity}</b></div>
   <div class="hdq-footnote">Progress shows workflow phases, not estimated completion time.</div>
   <div class="hdq-footer">{footer}</div>
 </div>
 <div class="hdq-right" aria-hidden="true">
   <div class="hdq-bubble">⚡ Charging evidence!</div>
   <img class="hdq-mascot" src="{mascot}" alt="" style="{'animation:none' if (finished_state is not None or idle_state is not None) else ''}"/>
 </div>
</section>
{clock_script}
</body></html>"""


def sticky_banner_css() -> str:
    """Keep the *entire* status card beside, and optionally track, the scrollbar.

    Streamlit mounts the HTML artwork in an iframe. ``position: sticky`` on
    that iframe (or on the short container that owns it) stops working when
    its parent scrolls out of view. Anchor the named host Streamlit container
    to the viewport and link its vertical offset to scrollbar progress where
    supported; this is scoped only to the active status panel.
    It does not modify the rest of the page or capture any patient data.
    """
    return """<style>
/* Attach an optional named scroll timeline to Streamlit's scrolling viewport.
   The html timeline is a fallback for applications where the document scrolls. */
html,
[data-testid="stAppViewContainer"] {
    scroll-timeline-name: --hdq-page-scroll;
    scroll-timeline-axis: block;
}
/* On Chromium, track vertical scrollbar progress: the *whole* mini-panel
   travels down the right side of the viewport as the scroll thumb moves.
   Browsers without CSS scroll timelines retain a fixed visible mini-panel. */
@keyframes hdq-follow-scroll-thumb {
    from { translate: 0 0; }
    to { translate: 0 max(0px, calc(100dvh - 35.5rem)); }
}
/* Keep desktop status cards compact so they do not obscure the dataset.
   Retain the existing mobile width to avoid clipping text at small sizes.
   The internal mascots and their aspect ratios stay unchanged.
   A genuine viewport-following mini-panel, rather than an in-flow banner.
   The same rule applies to the synchronous phases and async exploration. */
.st-key-hdq_investigation_banner,
.st-key-hdq_investigation_banner_blocking {
    position: fixed !important;
    top: 4.1rem !important;
    right: 1.1rem !important;
    left: auto !important;
    width: min(390px, calc(100vw - 2.2rem)) !important;
    max-width: min(390px, calc(100vw - 2.2rem)) !important;
    z-index: 999 !important;
    pointer-events: none !important;
    margin: 0 !important;
    padding: 0 !important;
    animation-name: hdq-follow-scroll-thumb;
    animation-duration: 1s;
    animation-timing-function: linear;
    animation-fill-mode: both;
    animation-timeline: --hdq-page-scroll;
    animation-range: 0% 100%;
}
/* Mount terminal-state close buttons as native Streamlit overlays inside
   their OWN cards. The outer floating host is pointer-events:none, but these
   explicit overlays restore pointer events so dismiss controls remain usable.
   Positioning the overlay itself avoids the DOM-dependent .stButton offset. */
.st-key-hdq_investigation_status_card,
.st-key-hdq_recommendation_status_card,
.st-key-hdq_agent_c_status_card {
    position: relative !important;
    isolation: isolate !important;
    pointer-events: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
.st-key-hdq_investigation_close_overlay,
.st-key-hdq_recommendation_close_overlay {
    position: absolute !important;
    top: 5px !important;
    right: 9px !important;
    left: auto !important;
    width: 30px !important;
    min-width: 30px !important;
    height: 30px !important;
    z-index: 100 !important;
    margin: 0 !important;
    padding: 0 !important;
    pointer-events: auto !important;
}
.st-key-hdq_investigation_close_overlay *,
.st-key-hdq_recommendation_close_overlay * {
    pointer-events: auto !important;
}
.st-key-hdq_investigation_close_overlay .stButton,
.st-key-hdq_recommendation_close_overlay .stButton {
    width: 30px !important;
    height: 30px !important;
    margin: 0 !important;
}
.st-key-hdq_investigation_close_overlay button,
.st-key-hdq_recommendation_close_overlay button {
    min-height: 26px !important;
    width: 26px !important;
    height: 26px !important;
    padding: 0 !important;
    border-radius: 50% !important;
    background: #102d42ef !important;
    color: #fff !important;
    border: 1px solid #ffffffa6 !important;
    line-height: 1 !important;
    font-size: 18px !important;
}
.st-key-hdq_investigation_close_overlay button:hover,
.st-key-hdq_recommendation_close_overlay button:hover {
    background: #205a74 !important;
    border-color: #ffffff !important;
}
/* Streamlit's internal block and the component iframe fill the small card,
   not the full page. When cleared, the empty container takes no overlay area. */
.st-key-hdq_investigation_banner iframe,
.st-key-hdq_investigation_banner_blocking iframe {
    width: 100% !important;
    max-width: 100% !important;
}
@media (max-width: 650px) {
    .st-key-hdq_investigation_banner,
    .st-key-hdq_investigation_banner_blocking {
        top: 3.6rem !important;
        right: 0.5rem !important;
        width: min(390px, calc(100vw - 1rem)) !important;
        max-width: min(390px, calc(100vw - 1rem)) !important;
    }
}
</style>"""


def render_idle_investigation_banner_html(*, state: str = "ready") -> str:
    """Render the parked investigation card before a run starts.

    This is presentation only: never starts Gemma, displays an elapsed clock,
    or claims any exploratory checks have been verified.
    """
    return _banner(phase=1, phase_count=3, activity="", verified_checks=0,
                   elapsed="", start_epoch_ms=None, idle_state=state)


def render_investigation_phase_banner_html(*, phase: int, phase_count: int,
                                           activity: str, started_epoch_ms: int,
                                           verified_rounds: int = 0) -> str:
    """Render any active phase (including blocking registered-tool Phase 2)."""
    if not isinstance(started_epoch_ms, int) or isinstance(started_epoch_ms, bool) or started_epoch_ms < 0:
        raise ValueError("Invalid start timestamp")
    return _banner(phase=phase, phase_count=phase_count, activity=activity,
                   verified_checks=verified_rounds, elapsed="00:00",
                   start_epoch_ms=started_epoch_ms)


def render_exploration_banner_html(live: dict) -> str:
    """Render the ongoing background Phase 3 without claiming final coverage."""
    if not isinstance(live, dict) or live.get("status") != "running":
        raise ValueError("Exploration banner requires a running worker snapshot")
    runs = live.get("runs")
    checks = len(runs) if isinstance(runs, list) else 0
    start = live.get("started_at")
    return _banner(phase=3, phase_count=3,
                   activity=str(live.get("activity") or "Preparing the next exploration step"),
                   verified_checks=checks, elapsed=exploration_elapsed(start),
                   start_epoch_ms=_iso_to_epoch_ms(start))


def render_finished_investigation_banner_html(*, state: str, verified_checks: int) -> str:
    """Keep the investigation card after its worker ends, including partial runs.

    This is a *run-state* display, never a scientific clean-data certificate.
    A final elapsed duration is omitted because historical audits may not save it.
    """
    if state not in {"review", "partial", "limited_clear"}:
        raise ValueError("Invalid final investigation state")
    return _banner(phase=3, phase_count=3, activity="Review saved investigation findings",
                   verified_checks=verified_checks, elapsed="", start_epoch_ms=None,
                   finished_state=state)


def recommendation_progress_counts(snapshot: dict | None) -> dict[str, int]:
    """Derive Agent B progress solely from persisted per-finding job statuses."""
    entries = snapshot.get("entries") if isinstance(snapshot, dict) else None
    if not isinstance(entries, dict):
        entries = {}
    counts = {"total": len(entries), "ready": 0, "running": 0, "failed": 0, "pending": 0}
    for entry in entries.values():
        status = entry.get("status") if isinstance(entry, dict) else None
        if status in counts and status != "total":
            counts[status] += 1
        else:
            counts["pending"] += 1
    return counts



def recommendation_is_complete(snapshot: dict | None) -> bool:
    """Only completed, saved work may be dismissed; absence is not completion.

    Failed jobs count as terminal, with the error status still inspectable in
    findings. A pending or generating recommendation is never dismissible.
    """
    counts = recommendation_progress_counts(snapshot)
    return bool(counts["total"] and not counts["running"] and not counts["pending"])


def render_recommendation_banner_html(snapshot: dict | None, *, idle: bool = False) -> str:
    """Second floating card for Agent B, including its completed state.

    Agent B is once-only: this renderer reads a saved snapshot and never queues
    work or invokes Gemma. Counts may grow during one investigation if new
    verified exploratory findings are appended to its recommendation queue.
    """
    counts = recommendation_progress_counts(None if idle else snapshot)
    total, ready, running, failed, pending = (counts[k] for k in
                                               ("total", "ready", "running", "failed", "pending"))
    pct = 100 * ready / total if total else 0.0
    if idle:
        headline = "Agent B — waiting for QC findings"
        details = "Recommendations begin automatically after QC findings; no data are changed"
    elif not total:
        headline = "Agent B — not scheduled"
        details = "No per-finding recommendation jobs are recorded"
    elif running or pending:
        headline = "Agent B — preparing suggestions…"
        details = f"{ready} ready · {running} generating · {pending} queued · {failed} failed"
    elif failed:
        headline = "Agent B — finished with errors"
        details = f"{ready} ready · {failed} failed; review individual findings"
    else:
        headline = "Agent B — recommendations ready"
        details = f"{ready} saved · Each suggestion needs researcher review"
    # Animate Agent B only while work is active, matching the investigation mascot.
    # Its completed, failed, and unscheduled states remain still.
    # Queued or generating jobs are still active; waiting/terminal states are not.
    # Explicitly select the uploaded working dinosaur instead of always using
    # the green waiting image. Idle, completed, and failed states stop shaking.
    active = bool(not idle and (running or pending))
    mascot = (recommendation_processing_mascot_data_url() if active
              else recommendation_mascot_data_url())
    return f"""<!doctype html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
*{{box-sizing:border-box}}html,body{{margin:0;background:transparent;color:#fff;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Arial,sans-serif}}
.card{{position:relative;overflow:hidden;display:grid;grid-template-columns:minmax(0,1fr) 107px;gap:6px;align-items:center;min-height:127px;padding:10px 12px;border:1px solid #38dab9;border-radius:14px;background:linear-gradient(110deg,#09253c,#0a3e4c 80%,#0a5d59);box-shadow:0 2px 12px #14a99633}}
.title{{font-size:18px;font-weight:800;letter-spacing:-.025em;line-height:1.25;overflow-wrap:anywhere}}
.sub{{color:#ccf7ec;font-size:11px;margin:4px 0 8px}}
.meter{{color:#eefdf6;font-size:12px;font-variant-numeric:tabular-nums;margin-bottom:7px}}
.track{{height:6px;border-radius:30px;background:#b4eedc33;overflow:hidden}}
.fill{{height:100%;background:linear-gradient(90deg,#5fefb1,#c7e746);width:{pct:.2f}%;border-radius:30px}}
.note{{color:#bbecdd;font-size:10px;margin-top:7px;line-height:1.3}}
figure{{margin:0;height:111px;display:flex;align-items:center;justify-content:center}}
figure img{{width:105px;height:104px;object-fit:cover;object-position:50% 47%;border-radius:11px;box-shadow:0 0 9px #9cfa9040;transform-origin:center 83%}}
.card.is-active figure img{{animation:hdq-bound .95s ease-in-out infinite alternate}}
.card:not(.is-active) figure img{{animation:none!important;transform:none!important}}
@keyframes hdq-bound{{from{{transform:translateY(3px) rotate(-1deg)}}to{{transform:translateY(-4px) rotate(1deg)}}}}
@media(prefers-reduced-motion:reduce){{.card.is-active figure img{{animation:none}}}}
@media(max-width:550px){{.card{{grid-template-columns:minmax(0,1fr) 72px;min-height:122px;padding:9px}}.title{{font-size:15px}}figure{{height:87px}}figure img{{width:70px;height:82px}}.meter{{font-size:11px}}}}
</style></head><body>
<section class="card{' is-active' if active else ''}" role="status" aria-label="Agent B recommendation progress">
<div><div class="title">{escape(headline)}</div>
<div class="sub">Read-only cleaning suggestions · once per finding per investigation</div>
<div class="meter">{'Not started' if idle else f'{ready} of {total} suggestions ready'}</div>
<div class="track" {'role="img" aria-label="Not started"' if idle else f'role="progressbar" aria-valuemin="0" aria-valuemax="{total}" aria-valuenow="{ready}" aria-label="Saved recommendations"'}><div class="fill"></div></div>
<div class="note">{escape(details)}</div></div><figure aria-hidden="true"><img src="{mascot}" alt=""/></figure>
</section></body></html>"""



def render_agent_c_banner_html(*, ready_findings: int = 0, total_findings: int = 0,
                                active_conversations: int = 0, idle: bool = False) -> str:
    """On-demand Agent C status, never a second Agent B progress meter.

    Available Agent B recommendations only unlock Agent C. They are NOT jobs
    performed by Agent C, so the card must not expose their count as an Agent C
    completion percentage. Actual in-flight Agent C requests alone set active.
    """
    for value in (ready_findings, total_findings, active_conversations):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Agent C counts must be nonnegative integers")
    if ready_findings > total_findings:
        raise ValueError("Ready findings cannot exceed total findings")
    available = not idle and ready_findings > 0
    # Agent C is strictly on demand. It cannot be active merely because
    # another background agent produced more recommendations.
    active = available and active_conversations > 0
    if not available:
        headline = "Agent C — waiting for Agent B"
        meter = "Idle · No Agent C request running"
        note = "Agent C becomes available after the first recommendation is saved"
    elif active:
        headline = "Agent C — responding…"
        meter = (f"{active_conversations} active request"
                 f"{'s' if active_conversations != 1 else ''}")
        note = "Processing your selected finding; no automatic cleaning is authorized"
    else:
        headline = "Agent C — waiting for your input"
        meter = "Idle · No Agent C request running"
        note = "Select a finding and choose Yes or No to start Agent C"
    mascot = (agent_c_working_mascot_data_url() if active
              else agent_c_idle_mascot_data_url())
    return f"""<!doctype html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
*{{box-sizing:border-box}}html,body{{margin:0;background:transparent;color:#fff;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Arial,sans-serif}}
.card{{position:relative;overflow:hidden;display:grid;grid-template-columns:minmax(0,1fr) 107px;gap:6px;align-items:center;min-height:127px;padding:10px 12px;border:1px solid #ff8ad6;border-radius:14px;background:linear-gradient(110deg,#2a173d,#3d1948 70%,#4a2470);box-shadow:0 2px 12px #ff75d833}}
.title{{font-size:18px;font-weight:800;letter-spacing:-.025em;line-height:1.25;overflow-wrap:anywhere}}
.sub{{color:#ffe2f7;font-size:11px;margin:4px 0 8px}}
.meter{{color:#fff0fb;font-size:12px;font-variant-numeric:tabular-nums;margin-bottom:7px}}
/* Decorative divider, NOT a percentage of Agent B findings. */
.rule{{height:6px;border-radius:30px;background:#ffd7f233;overflow:hidden;position:relative}}
.card.is-active .rule:after{{content:"";position:absolute;inset:0;background:#ffa6ed66;animation:hdq-c-activity 1s ease-in-out infinite alternate}}
@keyframes hdq-c-activity{{from{{opacity:.25}}to{{opacity:1}}}}
.note{{color:#f6d8ff;font-size:10px;margin-top:7px;line-height:1.3}}
figure{{margin:0;height:111px;display:flex;align-items:center;justify-content:center}}
figure img{{width:105px;height:104px;object-fit:cover;object-position:center;border-radius:11px;box-shadow:0 0 9px #ffb7e440;transform-origin:center 83%}}
.card.is-active figure img{{animation:hdq-bound .95s ease-in-out infinite alternate}}
.card:not(.is-active) figure img{{animation:none!important;transform:none!important}}
@keyframes hdq-bound{{from{{transform:translateY(3px) rotate(-1deg)}}to{{transform:translateY(-4px) rotate(1deg)}}}}
@media(prefers-reduced-motion:reduce){{.card.is-active figure img,.card.is-active .rule:after{{animation:none}}}}
@media(max-width:550px){{.card{{grid-template-columns:minmax(0,1fr) 72px;min-height:122px;padding:9px}}.title{{font-size:15px}}figure{{height:87px}}figure img{{width:70px;height:82px}}.meter{{font-size:11px}}}}
</style></head><body>
<section class="card{' is-active' if active else ''}" role="status" aria-label="Agent C discussion status">
<div><div class="title">{escape(headline)}</div>
<div class="sub">Interactive review · starts only after your selection</div>
<div class="meter">{escape(meter)}</div>
<div class="rule" aria-hidden="true"></div>
<div class="note">{escape(note)}</div></div><figure aria-hidden="true"><img src="{mascot}" alt=""/></figure>
</section></body></html>"""
