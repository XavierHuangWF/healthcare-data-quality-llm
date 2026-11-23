"""Non-destructive update: remove the extra 'Yes — Prepare Changes' stage.

Usage (PyCharm project root):
    python tools/apply_single_selection_workflow.py

Run after extracting this ZIP. The installer includes its prior one-click
prerequisite and refuses to rewrite unfamiliar app.py sections.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'app.py'
HELPER = ROOT / 'src/healthcare_data_quality/ui/single_selection_flow.py'
PREP_MARKER = 'from healthcare_data_quality.ui.single_selection_flow import ('
START = '                    elif control_phase == "discussion" and mode == "alternative":\n'
END = '                    # One conversation form per finding'

# The exact change lives here; no companion metadata files or HTML assets.
# Do not import/execute project code during preflight.
REPLACEMENT = r'''
                    elif control_phase == "discussion" and mode == "alternative":
                        # Agent C prepares its source-exact proposal READ-ONLY.
                        # A prose recommendation is not itself a user-selectable
                        # action: do not request a preliminary approval and then
                        # another "Yes — Approve & Execute" for the same intent.
                        # The reviewer gets one actionable click, *after* the
                        # validated Before→After proposal is visible.
                        last_reply = (transcript[-1] if transcript else None)
                        if (isinstance(last_reply, dict)
                                and last_reply.get("role") == "assistant"
                                and str(last_reply.get("content") or "").strip()
                                and not last_reply.get("automatic")
                                and not last_reply.get("controls_expired")
                                and any(word in str(last_reply.get("content") or "").casefold()
                                        for word in ("recommend", "propos", "replace", "delet",
                                                     "remov", "merge", "standardiz", "correct",
                                                     "rename", "clean", "apply", "convert", "pad "))):
                            from healthcare_data_quality.ui.single_selection_flow import (
                                proposal_request_once,
                            )
                            human_turns = [
                                f"{index}:{item.get('content', '')}"
                                for index, item in enumerate(transcript)
                                if item.get("role") == "user" and not item.get("automatic")
                            ]
                            receipt = proposal_request_once(
                                st.session_state, fid, last_reply["content"],
                                source.read_bytes() if source else b"",
                                user_turn=human_turns[-1] if human_turns else "",
                            )
                            if receipt is not None:
                                # Idempotent: only one hidden READ-ONLY request
                                # for this assistant reply. A later rerun cannot
                                # turn this into a second authorization.
                                st.session_state.cleaning_review_statuses[fid] = (
                                    "Preparing verified changes")
                                try:
                                    _queue_agent_c_for_finding(
                                        finding, source, uploaded, dataset_paths,
                                        "Prepare the exact structured operation(s) for the "
                                        "most recent recommended action. Identify every "
                                        "original CSV row, column, before value, and after "
                                        "value. Return the operation data now. This is "
                                        "READ-ONLY planning, not authorization to execute.",
                                        automatic=True,
                                    )
                                except (OSError, ValueError, KeyError, RuntimeError) as exc:
                                    st.session_state.cleaning_review_statuses[fid] = "Discussion open"
                                    st.error("Could not prepare source-verified changes: " + str(exc))
                                else:
                                    _rerun_cleaning_fragment(fid)
                            else:
                                st.caption("Preparing the verified Before → After proposal." if
                                           fid in st.session_state.cleaning_pending_gemma_b else
                                           "A verified action is needed before approval.")
                                if st.button(
                                    "No — Discuss Another Idea",
                                    key=f"review_discuss_no_{fid}_{_active_agent_c_control_generation(fid)}_agent_c",
                                ):
                                    last_reply["controls_expired"] = True
                                    _open_agent_c_alternative_chat(fid)
                                    _rerun_cleaning_fragment(fid)
'''


def patch_app(source: str) -> str:
    if source.count(START) != 1 or source.count(END) != 1:
        raise ValueError('Agent C UI layout differs; no files changed')
    a = source.index(START)
    b = source.index(END, a)
    existing = source[a:b]
    if PREP_MARKER in existing:
        return source  # already applied
    if ('"Yes — Prepare Changes"' not in existing or
            '_queue_agent_c_for_finding(' not in existing or
            '"No — Discuss Another Idea"' not in existing):
        raise ValueError('Unknown Agent C approval flow; refusing to alter code')
    replacement = REPLACEMENT.lstrip("\n")
    if not replacement.startswith(START) or not replacement.endswith('\n'):
        raise ValueError('Installer replacement was damaged')
    result = source[:a] + replacement + source[b:]
    ast.parse(result, filename=str(APP))
    if '"Yes — Prepare Changes"' in result:
        raise ValueError('Another preparatory Yes button still exists; review it manually')
    return result


def main():
    if not APP.is_file() or not HELPER.is_file():
        raise SystemExit('Extract ZIP in the project root with its directories intact')
    # Preflight BOTH transformations before mutating any file. An older
    # project is upgraded to first-click Agent B execution by the bundled
    # prerequisite installer; an already-upgraded project is left intact.
    original = APP.read_text(encoding='utf8')
    if (ROOT / 'tools/apply_verified_human_choices.py').is_file():
        from apply_verified_human_choices import (
            patch_app as patch_prior_app, main as install_prior,
        )
        patch_app(patch_prior_app(original))  # preflight the combined result
        install_prior()
    original = APP.read_text(encoding='utf8')
    if ('from healthcare_data_quality.ui.one_click_choice import execute_verified_choice'
            not in original):
        raise SystemExit('Missing Agent B direct-execute support; prior patch must be installed first')
    new = patch_app(original)
    if new == original:
        print('Already installed; no changes made.')
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backups = (ROOT / '.git' / 'hdq-patch-backups' / stamp
               if (ROOT / '.git').is_dir()
               else Path(tempfile.mkdtemp(prefix='hdq-single-selection-backup-')))
    backups.mkdir(parents=True, exist_ok=True)
    shutil.copy2(APP, backups / 'app.py')
    # The importer must exist and be parseable before touching app.py.
    ast.parse(HELPER.read_text(encoding='utf8'), filename=str(HELPER))
    tmp = APP.with_name('app.single_selection.tmp')
    try:
        tmp.write_text(new, encoding='utf8')
        tmp.replace(APP)
    finally:
        tmp.unlink(missing_ok=True)
    print('Updated: app.py')
    print('Backup:', backups / 'app.py')
    print('Run: python -m pytest -q --tb=short')


if __name__ == '__main__':
    main()
