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
from urllib.error import HTTPError, URLError
from http.client import IncompleteRead, RemoteDisconnected, BadStatusLine
from time import sleep, monotonic

from healthcare_data_quality.services.model_connectivity import (
    note_connection_failure, note_connection_success, forget_finished_request,
    model_connection_snapshot,
)

# MODEL_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_URL = "http://192.168.5.81:8000/v1/chat/completions"

MODEL_NAME = "gemma4-12b"
# HEALTH_URL = "http://127.0.0.1:8000/health"
HEALTH_URL = "http://192.168.5.81:8000/health"

_REQUEST_PRIORITY = ContextVar("hdq_llm_request_priority", default="interactive")
_AUTO_RECONNECT = ContextVar("hdq_llm_auto_reconnect", default=False)
_REVIEW_DEADLINE = ContextVar("hdq_review_deadline", default=None)
_REQUEST_OBSERVER = ContextVar("hdq_request_observer", default=None)
_SPECULATIVE_STOP = ContextVar("hdq_speculative_stop", default=None)


@contextmanager
def cancel_speculative_requests(stop):
    """Retired private work may finish a dispatched POST, but never retry it."""
    token = _SPECULATIVE_STOP.set(stop)
    try:
        yield
    finally:
        _SPECULATIVE_STOP.reset(token)



@contextmanager
def observe_model_requests(observer):
    """Per-worker, privacy-safe telemetry; never pass prompt or response text."""
    token = _REQUEST_OBSERVER.set(observer)
    try:
        yield
    finally:
        _REQUEST_OBSERVER.reset(token)


def report_model_stage(stage, **metrics):
    observer = _REQUEST_OBSERVER.get()
    if observer is not None:
        try:
            observer(stage, metrics)
        except Exception:
            pass  # telemetry cannot affect an inference or approval



@contextmanager
def bounded_recommendation_review():
    """Limit one automatic assessment/draft, not the entire investigation.

    A nested retry shares the same deadline. A timeout never authorizes a
    candidate; the normal background recovery can try again later.
    """
    if _REVIEW_DEADLINE.get() is not None:
        yield
        return
    try:
        seconds = max(15.0, min(600.0, float(os.getenv('HDQ_RECOMMENDATION_BUDGET_SECONDS', '120'))))
    except ValueError:
        seconds = 120.0
    token = _REVIEW_DEADLINE.set(monotonic() + seconds)
    try:
        yield
    finally:
        _REVIEW_DEADLINE.reset(token)


@contextmanager
def bounded_final_verification_batch():
    """One bounded inference per export batch, independent of draft job budget.

    A large verified export must not fail solely because earlier successful
    batches used the recommendation deadline. The number of batches is finite;
    timeout/transport errors still abort without resubmitting a live request.
    """
    try:
        seconds = max(15.0,min(600.0,float(os.getenv('HDQ_RECOMMENDATION_BUDGET_SECONDS','120'))))
    except ValueError:
        seconds = 120.0
    token = _REVIEW_DEADLINE.set(monotonic()+seconds)
    try:
        yield
    finally:
        _REVIEW_DEADLINE.reset(token)


def _remaining_time(default=None):
    stop = _SPECULATIVE_STOP.get()
    if stop is not None and stop.is_set():
        raise TimeoutError('Obsolete private recommendation cancelled')
    deadline = _REVIEW_DEADLINE.get()
    if deadline is None:
        return default
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError('Automatic recommendation assessment timed out; background recovery required')
    return min(default, remaining) if default is not None else remaining



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
                    wait = _remaining_time()
                    if _SPECULATIVE_STOP.get() is not None:
                        wait = min(0.25, wait) if wait is not None else 0.25
                    self._condition.wait(timeout=wait)
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


def _reconnect_interval() -> float:
    """Seconds between attempts; kept configurable for on-device deployments."""
    try:
        return max(0.1, min(60.0, float(os.getenv("HDQ_GEMMA_RECONNECT_SECONDS", "5"))))
    except ValueError:
        return 5.0


def _automatic_reconnect() -> bool:
    override = os.getenv("HDQ_GEMMA_AUTO_RECONNECT")
    if override is not None:
        return override.strip().lower() not in {"0", "false", "no", "off"}
    # Only application workers explicitly opt in; bare CLI/test calls fail
    # promptly instead of hanging indefinitely when no server is installed.
    return _AUTO_RECONNECT.get()


@contextmanager
def model_reconnect_mode():
    """Scope indefinite local Gemma reconnection to an app-owned worker."""
    token = _AUTO_RECONNECT.set(True)
    try:
        yield
    finally:
        _AUTO_RECONNECT.reset(token)


def _retryable_http(code: int, detail: str = "") -> bool:
    # 400/401/403/404 and invalid-model/schema failures are permanent errors.
    # A generic 500 could be CUDA OOM or a genuine model bug: do not hang on it.
    if code in {429, 502, 503, 504}:
        return True
    return code == 500 and any(hint in detail.lower() for hint in (
        "connection refused", "upstream unavailable", "server restarting",
        "temporarily unavailable", "connection reset", "server shutting down",
        "enginedeaderror", "enginecore encountered an issue", "engine is dead",
        "engine core is dead", "engine has died", "engine core died",
    ))


def _engine_is_unavailable() -> bool:
    """Probe loopback health after ambiguous 500; never retry a healthy 500 blindly."""
    try:
        with urlopen(Request(HEALTH_URL, method="GET"), timeout=3) as response:
            return response.status != 200
    except HTTPError as exc:
        try:
            return exc.code in {500, 502, 503, 504}
        finally:
            exc.close()
    except (URLError, TimeoutError, OSError, IncompleteRead, RemoteDisconnected, BadStatusLine):
        return True


def _ambiguous_server_error(detail: str) -> bool:
    """Avoid retrying explicit CUDA, invalid request or context errors indefinitely."""
    text = detail.lower()
    return not any(hint in text for hint in (
        "out of memory", "cuda error", "illegal memory access", "invalid schema",
        "context length", "maximum context", "invalid request", "validation error",
    ))


class ModelCompletionTimeout(TimeoutError):
    """A dispatched request did not return; server execution may still be active."""


def _completion_timeout_seconds() -> float:
    # Keep long investigations and interactive replies compatible with the
    # existing default. B's shorter overall assessment deadline still wins.
    try:
        return max(5.0, min(600.0, float(os.getenv('HDQ_GEMMA_REQUEST_TIMEOUT_SECONDS', '600'))))
    except ValueError:
        return 600.0


def _is_transport_timeout(exc: BaseException) -> bool:
    return isinstance(exc, TimeoutError) or (
        isinstance(exc, URLError) and isinstance(exc.reason, TimeoutError))


def _submit_with_reconnect(request: Request) -> dict:
    """Wait through temporary local-server outages without losing agent progress.

    A retry is always a model *read* request; application-side cleaning actions
    are never replayed here. We release the request-admission lane before
    sleeping so an offline server cannot occupy all three agent slots.
    No Streamlit APIs, source records or prompts are put into status metadata.
    """
    wait_for_health = False
    try:
        while True:
            _remaining_time()
            if wait_for_health:
                if _engine_is_unavailable():
                    report_model_stage('waiting_for_reconnect')
                    note_connection_failure("Gemma engine unavailable")
                    sleep(_remaining_time(_reconnect_interval()))
                    continue
                wait_for_health = False
            dispatched = False
            try:
                report_model_stage('waiting_for_inference_slot')
                with _GATE.admit(_REQUEST_PRIORITY.get()):
                    timeout = _remaining_time(_completion_timeout_seconds())
                    report_model_stage('completion_sent', timeout_seconds=round(timeout, 1))
                    started = monotonic()
                    dispatched = True
                    with urlopen(request, timeout=timeout) as response:
                        result = json.load(response)
                    dispatched = False  # response has actually been received
                    _remaining_time()
                    usage = result.get('usage') or {}
                    report_model_stage('completion_received', elapsed_seconds=round(monotonic()-started, 2),
                        prompt_tokens=usage.get('prompt_tokens'), completion_tokens=usage.get('completion_tokens'))
            except HTTPError as exc:
                try:
                    detail = exc.read(1500).decode("utf-8", errors="replace")
                finally:
                    exc.close()
                report_model_stage('http_error', status_code=exc.code)
                if not _automatic_reconnect():
                    raise RuntimeError(f"vLLM HTTP {exc.code}: {detail}") from exc
                retryable = _retryable_http(exc.code, detail)
                if exc.code == 500 and _ambiguous_server_error(detail):
                    # EngineDeadError may be serialized as a generic 500 while
                    # the API process is shutting down. Check actual health,
                    # outside the inference admission gate.
                    wait_for_health = retryable or _engine_is_unavailable()
                    retryable = wait_for_health
                if not retryable:
                    raise RuntimeError(f"vLLM HTTP {exc.code}: {detail}") from exc
                note_connection_failure("Gemma engine unavailable" if wait_for_health else f"HTTP {exc.code}")
            except (URLError, TimeoutError, ConnectionError, OSError,
                    IncompleteRead, RemoteDisconnected, BadStatusLine) as exc:
                # A timed-out POST is ambiguous: vLLM may still own its
                # inference request. Retrying here can add duplicate GPU work
                # and obscure an engine stall. Releasing our local gate does
                # not prove server-side cancellation. Never classify a mere
                # completion timeout as a confirmed server outage.
                if _is_transport_timeout(exc):
                    if dispatched:
                        report_model_stage('inference_timeout')
                        # Stop higher-level automatic fallbacks in this B job
                        # from sending a second inference after an ambiguous timeout.
                        if _REVIEW_DEADLINE.get() is not None:
                            _REVIEW_DEADLINE.set(monotonic())
                        raise ModelCompletionTimeout(
                            'Gemma completion timed out; original server request may still be active'
                        ) from exc
                    report_model_stage('assessment_timed_out')
                    raise
                report_model_stage('transport_error', error_type=type(exc).__name__)
                if not _automatic_reconnect():
                    raise
                # Do not copy potentially private request text into the UI.
                note_connection_failure(type(exc).__name__)
            else:
                note_connection_success()
                return result
            report_model_stage('waiting_for_reconnect')
            sleep(_remaining_time(_reconnect_interval()))
    finally:
        forget_finished_request()


def _token_budget(prompt: str, output: int) -> int:
    """Preflight large prompts using the local vLLM tokenizer, including its chat template.

    Never silently truncate source values or instructions. Callers can retry a
    purpose-built compact prompt. The server remains authoritative on capacity.
    """
    if len(prompt.encode('utf-8')) < 6000:
        return output  # inexpensive short requests retain their existing path
    report_model_stage('tokenizing', prompt_bytes=len(prompt.encode('utf-8')))
    try:
        request = Request(MODEL_URL.rsplit('/v1/', 1)[0] + '/tokenize',
            data=json.dumps({'model':MODEL_NAME,'messages':[{'role':'user','content':prompt}],
                             'add_generation_prompt':True}).encode('utf-8'),
            headers={'Content-Type':'application/json'}, method='POST')
        with urlopen(request, timeout=_remaining_time(5)) as response:
            stats = json.load(response)
        count = stats.get('count'); limit = stats.get('max_model_len')
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, TypeError, AttributeError) as exc:
        report_model_stage('tokenizer_unavailable', error_type=type(exc).__name__)
        _remaining_time()
        return output  # legacy tokenizer endpoint: server error is handled by caller
    if type(count) is not int or type(limit) is not int or limit <= 0:
        return output
    # Reserve chat/schema overhead and enough completion room for structured plans.
    available = limit - count - 128
    report_model_stage('tokenized', input_tokens=count, context_limit=limit, requested_output=output)
    if available < min(output, 1024):
        report_model_stage('context_overflow', input_tokens=count, context_limit=limit)
        raise RuntimeError(f'Context budget exceeded: {count} input tokens, limit {limit}, output reserve {output}')
    return min(output, available)


def _json_output_mode() -> str:
    """Avoid the GPU guided-decoding path by default; explicit opt-in for A/B tests."""
    mode = os.getenv('HDQ_GEMMA_JSON_MODE', 'local').strip().lower()
    if mode not in {'local', 'guided'}:
        raise ValueError('HDQ_GEMMA_JSON_MODE must be local or guided')
    return mode


def _json_schema_prompt(prompt: str, schema: dict) -> str:
    return ('Return exactly one JSON value matching this output schema. '
            'No Markdown, commentary, or extra keys. The schema is an output '
            'contract, not dataset evidence.\n'
            + json.dumps(schema, ensure_ascii=False, separators=(',', ':'))
            + '\n\nTASK:\n' + prompt)


def _validate_json_response(content: str, schema: dict) -> str:
    """Fail closed on malformed/ambiguous output, before any caller consumes it."""
    import jsonschema
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key in Gemma response')
            result[key] = value
        return result
    def reject_constant(_):
        raise ValueError('Non-finite JSON number in Gemma response')
    if not isinstance(content, str):
        raise ValueError('Gemma did not return JSON text')
    text = content.strip()
    # Accept only a fence wrapping the entire response, never extract a JSON
    # substring from arbitrary prose or accept a partial generation.
    if text.startswith('```json\n') and text.endswith('\n```'):
        text = text[len('```json\n'):-len('\n```')].strip()
    try:
        value = json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)
        jsonschema.validate(value, schema)
    except (ValueError, jsonschema.ValidationError, jsonschema.SchemaError) as exc:
        report_model_stage('invalid_json_response', error_type=type(exc).__name__)
        # Never echo model-produced values or source data in transport logs.
        raise ValueError('Gemma JSON response failed local schema validation') from None
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def query_local_model(
        prompt: str,
        json_schema: dict | None = None,
        max_tokens: int = 4096,
) -> str:
    _remaining_time()
    mode = _json_output_mode() if json_schema is not None else None
    if mode == "local":
        prompt = _json_schema_prompt(prompt, json_schema)
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": _token_budget(prompt, max_tokens),
    }
    if mode == "guided":
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

    # Retry the same read-only model request while vLLM is temporarily down.
    # The original worker continues after recovery; no investigation restart.
    result = _submit_with_reconnect(request)

    choice = result["choices"][0]
    if choice["finish_reason"] != "stop":
        report_model_stage('incomplete_generation')
        usage = result.get("usage") or {}
        raise RuntimeError(
            f"Incomplete generation: {choice['finish_reason']}; "
            f"prompt_tokens={usage.get('prompt_tokens')}; "
            f"completion_tokens={usage.get('completion_tokens')}; "
            f"completion_details={usage.get('completion_tokens_details')}"
        )
    content = choice["message"]["content"]
    if json_schema is not None:
        return _validate_json_response(content, json_schema)
    return content
