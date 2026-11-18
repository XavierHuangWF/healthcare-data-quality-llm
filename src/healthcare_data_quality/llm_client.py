"""Access one local vLLM model with role-aware request admission.

Three logical agents share one loaded Gemma instance. With
HDQ_LLM_MAX_ACTIVE=3, Agent A investigation, Agent B recommendation and
Agent C interaction each have one independent HTTP request lane.
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
    value = os.environ.get("HDQ_LLM_MAX_ACTIVE", "1")
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
    A role's lane is never stolen by another role at capacity=2/3. This is a
    *local process* gate; other clients of vLLM are not managed by it.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._active = 0
        self._active_by_role = {"investigation": 0, "recommendation": 0, "interactive": 0}
        self._waiting_interactive = 0

    @staticmethod
    def _role(priority: str) -> str:
        if priority == "background":
            return "investigation"
        if priority not in {"investigation", "recommendation", "interactive"}:
            raise ValueError(f"Unknown model request priority: {priority}")
        return priority

    def _can_admit(self, role: str, capacity: int) -> bool:
        if self._active >= capacity:
            return False
        if capacity == 1:
            return role == "interactive" or self._waiting_interactive == 0
        if capacity == 2:
            if role == "interactive":
                return self._active_by_role["interactive"] == 0
            return (self._active_by_role["investigation"] +
                    self._active_by_role["recommendation"] == 0 and
                    self._waiting_interactive == 0)
        # 3 active requests maximum: reserve an independent lane per role.
        # The waiting-interactive check is not needed here, because Agent C's
        # lane cannot be occupied by A or B.
        return self._active_by_role[role] == 0

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
                self._active_by_role[role] += 1
            finally:
                if role == "interactive":
                    self._waiting_interactive -= 1
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._active_by_role[role] -= 1
                self._condition.notify_all()


_GATE = _RequestGate()


@contextmanager
def model_request_priority(priority: str):
    if priority not in {"background", "investigation", "recommendation", "interactive"}:
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
