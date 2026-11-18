"""Optional on-device benchmark: two concurrent requests to ONE vLLM server.

Uses harmless prompts and no research data. A successful pair means the server
can accept two requests; it does not prove lower latency or sufficient headroom
for full investigation contexts.
"""

import json
import threading
import time
from urllib.request import Request, urlopen

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "gemma4-12b"
barrier = threading.Barrier(3)
results = [None, None]


def run(index: int) -> None:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content":
                      f"Respond with one short sentence about checking CSV columns. Test {index + 1}."}],
        "temperature": 0,
        "max_tokens": 80,
    }
    barrier.wait()
    started = time.perf_counter()
    try:
        req = Request(URL, data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=180) as response:
            reply = json.load(response)
        results[index] = {"elapsed_s": round(time.perf_counter() - started, 2),
                          "finish_reason": reply["choices"][0]["finish_reason"],
                          "completion_tokens": (reply.get("usage") or {}).get("completion_tokens")}
    except Exception as exc:
        results[index] = {"elapsed_s": round(time.perf_counter() - started, 2),
                          "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    started = time.perf_counter()
    for thread in threads:
        thread.join()
    print(json.dumps({"wall_time_s": round(time.perf_counter() - started, 2),
                      "requests": results}, indent=2))
    if any("error" in r for r in results):
        raise SystemExit("One or more requests failed. Keep vLLM at one sequence and review its logs.")
