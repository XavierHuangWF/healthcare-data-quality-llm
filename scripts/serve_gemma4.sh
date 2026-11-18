
#!/usr/bin/env bash
set -euo pipefail

# Resolve paths independently of the current directory.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/models/gemma-4-12b"
VLLM="$HOME/venvs/healthcare-vllm/bin/vllm"

# Require local model files. No online model downloads.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# The investigated workflow needs >= 12,288 tokens of model context.  Do not
# shrink the context to accommodate concurrent inference requests:
# reduce HDQ_MAX_NUM_SEQS instead if GPU/KV-cache memory is insufficient.
MODEL_LEN="${HDQ_MAX_MODEL_LEN:-12288}"
if [[ ! "$MODEL_LEN" =~ ^[0-9]+$ ]] || (( 10#$MODEL_LEN < 12288 )); then
    echo "ERROR: HDQ_MAX_MODEL_LEN must be an integer >= 12288 (received: '$MODEL_LEN')." >&2
    echo "Keep the tested minimum context; use HDQ_MAX_NUM_SEQS=1 or 2 explicitly if GPU memory cannot support 3." >&2
    exit 2
fi

MAX_SEQS="${HDQ_MAX_NUM_SEQS:-3}"
if [[ ! "$MAX_SEQS" =~ ^[123]$ ]]; then
    echo "ERROR: HDQ_MAX_NUM_SEQS must be 1, 2, or 3 (received: '$MAX_SEQS')." >&2
    exit 2
fi

echo "Starting Gemma 4 12B..."
echo "Model: $MODEL_DIR"
echo "Server: http://127.0.0.1:8000"
echo "Model context: $MODEL_LEN tokens (minimum: 12288)"
echo "vLLM max simultaneous sequences: $MAX_SEQS"

exec "$VLLM" serve "$MODEL_DIR" \
    --served-model-name gemma4-12b \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len "$MODEL_LEN" \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs "$MAX_SEQS" \
    --language-model-only \
    --enforce-eager
