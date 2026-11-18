"""Smoke-test 3 simultaneous HTTP requests against ONE locally hosted vLLM.

Use synthetic, non-sensitive text only. This is not a memory-safety or
full-context benchmark. Inspect vLLM logs for KV cache preemption and OOM.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from urllib.request import Request, urlopen

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "gemma4-12b"


def _running_requests_from_metrics(payload: str) -> int | None:
    """Read vLLM RUNNING gauge if available; None for unsupported metrics."""
    values = []
    for line in payload.splitlines():
        if line.startswith('#'):
            continue
        if re.match(r'^vllm(?::|_)num_requests_running(?:\{|\s)', line):
            try:
                values.append(float(line.split()[-1]))
            except (ValueError, IndexError):
                pass
    return int(sum(values)) if values else None



def benchmark(prompt_words: int = 400, max_tokens: int = 192, timeout_s: int = 240) -> dict:
    if not (1 <= prompt_words <= 2500):
        raise ValueError("--prompt-words must be in the range 1–2500")
    if not (1 <= max_tokens <= 1024):
        raise ValueError("--max-tokens must be in the range 1–1024")
    barrier = threading.Barrier(4)
    results: list[dict | None] = [None, None, None]
    active = 0
    peak_client_active = 0
    mutex = threading.Lock()
    finished = threading.Event()
    observed_vllm_peak = None
    metrics_error = None

    def sample_vllm_metrics():
        nonlocal observed_vllm_peak, metrics_error
        while not finished.is_set():
            try:
                with urlopen("http://127.0.0.1:8000/metrics", timeout=1) as response:
                    live = _running_requests_from_metrics(response.read().decode("utf-8"))
                if live is not None:
                    observed_vllm_peak = max(observed_vllm_peak or 0, live)
            except Exception as exc:
                if metrics_error is None:
                    metrics_error = type(exc).__name__
            finished.wait(0.2)

    # Deliberately synthetic content, never uploaded research data.
    vocabulary = ("csv taxonomy values validation missing duplicate sample "
                  "measurement scientific evidence verification reproducibility").split()
    sample = " ".join(vocabulary[i % len(vocabulary)] for i in range(prompt_words))

    def run(index: int):
        nonlocal active, peak_client_active
        role = ("investigation", "recommendation", "interactive")[index]
        prompt = (f"Request {index + 1}, role {role}. Review this synthetic vocabulary "
                  f"and provide a short independent checklist. {sample}")
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        request = Request(
            URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        barrier.wait()
        started = time.perf_counter()
        with mutex:
            active += 1
            peak_client_active = max(active, peak_client_active)
        try:
            with urlopen(request, timeout=timeout_s) as response:
                data = json.load(response)
            usage = data.get("usage") or {}
            result = {
                "role": role,
                "http": 200,
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "finish_reason": data["choices"][0]["finish_reason"],
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
        except Exception as exc:
            result = {
                "role": role,
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            }
        finally:
            with mutex:
                active -= 1
        results[index] = result

    workers = [threading.Thread(target=run, args=(i,), daemon=True)
               for i in range(3)]
    sampler = threading.Thread(target=sample_vllm_metrics, daemon=True)
    sampler.start()
    for worker in workers:
        worker.start()
    barrier.wait()
    started = time.perf_counter()
    for worker in workers:
        worker.join()
    finished.set()
    sampler.join(2)
    report = {
        "model": MODEL,
        "requested_parallelism": 3,
        "peak_client_in_flight": peak_client_active,
        "wall_seconds": round(time.perf_counter() - started, 2),
        "vllm_running_peak_if_available": observed_vllm_peak,
        "metrics_error_if_unavailable": metrics_error if observed_vllm_peak is None else None,
        "requests": results,
        "note": "Three vLLM RUNNING sequences are supported only if the sampled gauge reaches 3; a smaller sampled peak is inconclusive on short requests. Inspect GPU usage and server logs.",
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-words", type=int, default=400)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    report = benchmark(args.prompt_words, args.max_tokens, args.timeout)
    print(json.dumps(report, indent=2))
    if any(not isinstance(row, dict) or "error" in row or row.get("http") != 200
           for row in report["requests"]):
        raise SystemExit("A request failed. Reduce concurrency to 2 and inspect vLLM logs.")
