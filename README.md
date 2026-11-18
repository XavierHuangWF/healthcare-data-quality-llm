Agent C status card patch
=========================

What changed
------------
- Added a third floating status card for **Agent C** so it appears alongside Agent A and Agent B.
- Added two new local mascot images for Agent C:
  - `assets/gemma_agent_c_idle.png` for idle / ready states
  - `assets/gemma_agent_c_working.png` for active reply / processing state
- Parked Agent C before investigation starts, matching the pre-run idle display of Agent A and Agent B.
- Agent C now shows three honest states:
  - **waiting to discuss**: no Agent B suggestion is available yet
  - **ready to discuss**: at least one suggestion is available and Agent C is idle
  - **replying**: Agent C is currently processing a user request
- Added a small regression test for the new Agent C banner renderer.

Files in patch
--------------
- `app.py`
- `exploration_banner.py`
- `assets/gemma_agent_c_idle.png`
- `assets/gemma_agent_c_working.png`
- `tests/test_agent_c_status_card.py`

Validation
----------
- `PYTHONPATH=. pytest -q tests/test_agent_c_status_card.py`
- `python -m py_compile app.py exploration_banner.py`
