"""Access one local vLLM model with role-aware request admission.

Three logical agents share one loaded Gemma instance. With
HDQ_LLM_MAX_ACTIVE=3, Agent A investigation, Agent B recommendation and
Agent C interaction each have one independent HTTP request lane. When Agent C
is idle, one recommendation worker may borrow its lane for another finding.
This is an application-side limit, not a guarantee of vLLM/GPU concurrency.
In-flight calls cannot be preempted. All model calls stay local.
"""

import json
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from urllib.request import Request, urlopen
from urllib.error import HTTPError

MODEL_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "gemma4-12b"

_REQUEST_PRIORITY = ContextVar("hdq_llm_request_priority", default="interactive")


def _max_active_requests() -> int:
    value = os.environ.get("HDQ_LLM_MAX_ACTIVE", "3")
    try:
        number = int(value)
    except ValueError as exc:
        raise ValueError("HDQ_LLM_MAX_ACTIVE must be 1, 2, or 3") from exc
    if number not in (1, 2, 3):
        raise ValueError("HDQ_LLM_MAX_ACTIVE must be 1, 2, or 3")
    return number


class _RequestGate:
    """Admit role-specific requests without letting background jobs crowd out chat.

    Capacity=1: all roles serialize, with waiting interactive requests first.
    Capacity=2: one pooled background lane (A/B), one interactive lane (C).
    Capacity=3: one lane each for A, B, and C; all can overlap.

    "background" remains an alias for "investigation" to support older callers.
    At capacity=3 an optional idle-assist job can borrow Agent C's lane
    while no interactive request is active or waiting. A new chat takes
    priority at the next request boundary, but cannot preempt an in-flight
    inference. This is a *local process* gate; other vLLM clients are not managed.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._active = 0
        self._active_by_role = {"investigation": 0, "recommendation": 0, "interactive": 0}
        self._waiting_interactive = 0
        self._active_idle_assist = 0

    @staticmethod
    def _role(priority: str) -> str:
        if priority == "background":
            return "investigation"
        if priority not in {"investigation", "recommendation", "interactive", "idle_assist"}:
            raise ValueError(f"Unknown model request priority: {priority}")
        return priority

    def interaction_busy(self) -> bool:
        """True while a user model request is active or waiting.

        A cheap scheduling hint, not an authorization check. Admission still
        checks the interactive reservation atomically under the condition.
        """
        with self._condition:
            return bool(self._waiting_interactive or self._active_by_role["interactive"])

    def _can_admit(self, role: str, capacity: int) -> bool:
        if self._active >= capacity:
            return False
        if role == "idle_assist":
            return (capacity == 3 and not self._waiting_interactive
                    and self._active_by_role["interactive"] == 0
                    and self._active_idle_assist == 0)
        if capacity == 1:
            return role == "interactive" or self._waiting_interactive == 0
        if capacity == 2:
            if role == "interactive":
                return self._active_by_role["interactive"] == 0
            return (self._active_by_role["investigation"] +
                    self._active_by_role["recommendation"] == 0 and
                    self._waiting_interactive == 0)
        # 3 active requests maximum: keep A/B separate. C gets the third
        # slot unless one already-running idle-assist inference borrowed it.
        return (self._active_by_role[role] == 0
                and (role != "interactive" or self._active_idle_assist == 0))

    @contextmanager
    def admit(self, priority: str):
        role = self._role(priority)
        capacity = _max_active_requests()
        with self._condition:
            if role == "interactive":
                self._waiting_interactive += 1
            try:
                while not self._can_admit(role, capacity):
                    self._condition.wait()
                self._active += 1
                if role == "idle_assist":
                    self._active_idle_assist += 1
                else:
                    self._active_by_role[role] += 1
            finally:
                if role == "interactive":
                    self._waiting_interactive -= 1
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                if role == "idle_assist":
                    self._active_idle_assist -= 1
                else:
                    self._active_by_role[role] -= 1
                self._condition.notify_all()


_GATE = _RequestGate()


def interactive_model_request_pending() -> bool:
    """Allow background scheduling to avoid borrowing the user's model lane."""
    return _GATE.interaction_busy()


@contextmanager
def model_request_priority(priority: str):
    if priority not in {"background", "investigation", "recommendation", "interactive", "idle_assist"}:
        raise ValueError("Unknown model request priority")
    token = _REQUEST_PRIORITY.set(priority)
    try:
        yield
    finally:
        _REQUEST_PRIORITY.reset(token)


def query_local_model(
        prompt: str,
        json_schema: dict | None = None,
        max_tokens: int = 4096,
) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if json_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "structured_response", "schema": json_schema},
        }
    request = Request(
        MODEL_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # A short queued recommendation cannot monopolize the GPU while the
    # researcher is waiting. This cannot interrupt an already running call.
    with _GATE.admit(_REQUEST_PRIORITY.get()):
        try:
            with urlopen(request, timeout=600) as response:
                result = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM HTTP {exc.code}: {detail}") from exc

    choice = result["choices"][0]
    if choice["finish_reason"] != "stop":
        usage = result.get("usage") or {}
        raise RuntimeError(
            f"Incomplete generation: {choice['finish_reason']}; "
            f"prompt_tokens={usage.get('prompt_tokens')}; "
            f"completion_tokens={usage.get('completion_tokens')}; "
            f"completion_details={usage.get('completion_tokens_details')}"
        )
    return choice["message"]["content"]
