"""Presentation-only progress state for the local, read-only investigation UI.

Progress fractions represent *completed workflow phases*, never an estimate of
Gemma's unknown remaining exploratory rounds or total wall-clock duration.
"""
from __future__ import annotations

from dataclasses import dataclass


_PHASES = (
    "Deterministic data-quality checks",
    "Gemma investigation using registered tools",
    "Gemma exploratory analyses in the read-only sandbox",
)


@dataclass
class InvestigationProgress:
    include_exploration: bool
    phase: int = 1
    verified_rounds: int = 0
    last_activity: str = "Preparing the investigation..."
    finished: bool = False

    @property
    def phase_count(self) -> int:
        return 3 if self.include_exploration else 2

    @property
    def fraction(self) -> float:
        return 1.0 if self.finished else (self.phase - 1) / self.phase_count

    @property
    def phase_label(self) -> str:
        if self.finished:
            return "Investigation finished"
        return f"Phase {self.phase} of {self.phase_count}: {_PHASES[self.phase - 1]}"

    @property
    def round_label(self) -> str:
        return f"Verified exploratory rounds: {self.verified_rounds} (Gemma decides when to stop)"

    def baseline_event(self, message: str) -> None:
        """Map the existing backend's textual callbacks to known phases."""
        if not self.finished and (
            "baseline checks completed" in message.lower()
            or "profiling dataset" in message.lower()
            or "step " in message.lower()
            or "investigation objectives:" in message.lower()
        ):
            self.phase = max(self.phase, 2)
        self.last_activity = message

    def begin_exploration(self, *, previous_rounds: int = 0) -> None:
        if not self.include_exploration:
            raise ValueError("Exploration was not enabled for this investigation")
        self.phase = 3
        self.verified_rounds = previous_rounds
        self.last_activity = "Preparing Gemma's read-only exploratory analyses..."

    def exploration_event(self, message: str) -> None:
        if self.phase != 3:
            self.begin_exploration(previous_rounds=self.verified_rounds)
        self.last_activity = message

    def verified_round(self, count: int) -> None:
        if self.phase != 3:
            raise ValueError("Cannot verify an exploratory round before exploration")
        if count < self.verified_rounds:
            raise ValueError("Verified round count cannot decrease")
        self.verified_rounds = count
        self.last_activity = f"Verified exploratory investigation {count}."

    def finish(self) -> None:
        self.finished = True
        self.last_activity = "Investigation finished."


def format_elapsed(seconds: float) -> str:
    """Format an elapsed duration without interpreting it as an ETA."""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return (f"{hours:02}:{minutes:02}:{secs:02}" if hours
            else f"{minutes:02}:{secs:02}")


def live_timer_html(start_epoch_ms: int) -> str:
    """Self-updating offline timer for blocked synchronous Gemma HTTP calls.

    The only interpolated value is an integer supplied by our own UI. No CSV,
    patient information, model output, or external resources enter this HTML.
    """
    if not isinstance(start_epoch_ms, int) or isinstance(start_epoch_ms, bool) or start_epoch_ms < 0:
        raise ValueError("start_epoch_ms must be a nonnegative integer")
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
  html, body {{margin:0; padding:0; background:transparent; color:inherit;
    font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;}}
  .elapsed {{font-size:14px; line-height:32px; color:#9ca3af;}}
  #elapsed-time {{font-variant-numeric:tabular-nums; font-weight:700;
    font-size:19px; color:#f1f5f9; margin-left:8px;}}
</style></head><body>
<div class="elapsed">Elapsed time <span id="elapsed-time" aria-live="off">00:00</span></div>
<script>
const startedAt = {start_epoch_ms};
const output = document.getElementById('elapsed-time');
function updateElapsed() {{
  const seconds = Math.max(0, Math.floor((Date.now() - startedAt) / 1000));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remaining = seconds % 60;
  const pad = number => String(number).padStart(2, '0');
  output.textContent = hours
    ? `${{pad(hours)}}:${{pad(minutes)}}:${{pad(remaining)}}`
    : `${{pad(minutes)}}:${{pad(remaining)}}`;
}}
updateElapsed();
window.setInterval(updateElapsed, 1000);
</script></body></html>"""
