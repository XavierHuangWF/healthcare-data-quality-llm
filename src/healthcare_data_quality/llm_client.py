import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError

MODEL_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "gemma4-12b"


def query_local_model(
        prompt: str,
        json_schema: dict | None = None,
        max_tokens: int = 4096,
) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    if json_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_response",
                "schema": json_schema,
            },
        }

    request = Request(
        MODEL_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=180) as response:
            result = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"vLLM HTTP {exc.code}: {detail}"
        ) from exc

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
