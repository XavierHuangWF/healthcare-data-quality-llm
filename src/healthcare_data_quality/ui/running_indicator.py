"""Compact accessible jogging indicator for Agent B's initial waiting message.

This is purely presentational: never starts jobs, polls, or calls the model.
"""
from __future__ import annotations

from html import escape

DEFAULT_MESSAGE = (
    "Waiting for Agent B's first verified suggestion. "
    "Cleaning will open automatically when it is ready."
)


def waiting_runner_html(message: str = DEFAULT_MESSAGE) -> str:
    """Provide an animated runner after the exact notice without changing its meaning."""
    return '''<style>
.hdq-jogger-notice {
  display: flex; align-items: center; justify-content: flex-start; flex-wrap: wrap;
  gap: 12px; padding: 13px 16px; border-radius: 8px;
  background: rgba(31, 66, 100, .79); color: #54b3ff;
  font: inherit; line-height: 1.55;
}
.hdq-jogger-notice .hdq-jogger-text { min-width: 0; overflow-wrap: anywhere; }
.hdq-jogger-notice .hdq-jogger-icon {
  flex: 0 0 auto; display: inline-block; font-size: 23px;
  line-height: 1; transform-origin: 50% 85%;
  animation: hdq-jog 650ms ease-in-out infinite alternate;
}
@keyframes hdq-jog {
  from { transform: translate(0, 2px) rotate(-8deg); }
  to   { transform: translate(6px, -3px) rotate(8deg); }
}
@media (prefers-reduced-motion: reduce) {
  .hdq-jogger-notice .hdq-jogger-icon { animation: none; }
}
</style>
<div class="hdq-jogger-notice" role="status" aria-live="polite">
 <span class="hdq-jogger-text">''' + escape(message) + '''</span>
 <span class="hdq-jogger-icon" role="img" aria-label="Agent B is working">🏃‍♂️</span>
</div>'''


def show_waiting_runner(st, message: str = DEFAULT_MESSAGE) -> None:
    """A single static Streamlit element; browser CSS handles jogging animation."""
    st.markdown(waiting_runner_html(message), unsafe_allow_html=True)
