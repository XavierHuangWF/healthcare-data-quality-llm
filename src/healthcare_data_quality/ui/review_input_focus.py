"""One-shot focus for a reviewer who explicitly opens Agent C discussion.

The identifier is derived from a finding ID, never embedded as user-provided
HTML/JavaScript. Runs in the Streamlit frontend via its first-party st.html API.
"""
from __future__ import annotations

import hashlib


def review_chat_container_key(finding_id: str) -> str:
    digest = hashlib.sha256(str(finding_id).encode('utf-8')).hexdigest()[:16]
    return 'hdq_review_chat_' + digest


def review_chat_focus_html(finding_id: str) -> str:
    """Focus only the selected finding's chat textarea after a No click.

    DOM mounting may be deferred by React, so use a bounded observer. Never
    modify the widget value or submit a message on behalf of the reviewer.
    """
    container_key = review_chat_container_key(finding_id)
    return f'''<script>
    (() => {{
      const containerSelector = '[class*="st-key-{container_key}"]';
      const findInput = () => {{
        const scope = document.querySelector(containerSelector);
        return scope && scope.querySelector('[data-testid="stChatInput"] textarea, textarea');
      }};
      const giveFocus = () => {{
        const input = findInput();
        if (!input || input.disabled) return false;
        input.focus({{ preventScroll: true }});
        return document.activeElement === input;
      }};
      if (giveFocus()) return;
      const observer = new MutationObserver(() => {{
        if (giveFocus()) observer.disconnect();
      }});
      observer.observe(document.body, {{ childList: true, subtree: true }});
      // Prevent leaked observers if React does not mount the input.
      setTimeout(() => observer.disconnect(), 2500);
    }})();
    </script>'''


def focus_reviewer_chat(streamlit_module, finding_id: str) -> bool:
    """Use first-party frontend JS only where the installed Streamlit supports it."""
    if not hasattr(streamlit_module, 'html'):
        return False
    try:
        streamlit_module.html(review_chat_focus_html(finding_id),
                             unsafe_allow_javascript=True)
    except TypeError:
        # Very old Streamlit versions lack this parameter. Do not fall back
        # to an untrusted/iframe script or alter the review interaction.
        return False
    return True


def review_viewport_html(finding_id: str, completed: bool = False) -> str:
    """Bounded one-shot scroll to the selected card, including prior header.

    No iframe/parent DOM access, no observer left running, and no periodic scroll.
    """
    digest = hashlib.sha256(str(finding_id).encode()).hexdigest()[:16]
    return f'''<script>(() => {{
      const selector = '.st-key-hdq_review_card_{digest}';
      let attempts = 0;
      const restore = () => {{
        const card = document.querySelector(selector);
        if (!card || !card.querySelector('details')) {{
          if (++attempts < 12) setTimeout(restore, 80);
          return;
        }}
        // Expansion is owned by Streamlit, not this script. Only scroll.
        requestAnimationFrame(() => {{
          card.scrollIntoView({{block:'start',behavior:'instant'}});
          // Offset inside the actual scrolling ancestor to keep preceding context.
          let parent = card.parentElement;
          while (parent && !(parent.scrollHeight > parent.clientHeight && /auto|scroll/.test(getComputedStyle(parent).overflowY))) parent = parent.parentElement;
          if (parent) parent.scrollTop = Math.max(0, parent.scrollTop - 96);
          else window.scrollBy(0,-96);
        }});
      }};
      requestAnimationFrame(restore);
    }})();</script>'''
