
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

echo "Starting Gemma 4 12B..."
echo "Model: $MODEL_DIR"
echo "Server: http://127.0.0.1:8000"

exec "$VLLM" serve "$MODEL_DIR" \
    --served-model-name gemma4-12b \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 8192  \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 1 \
    --language-model-only \
    --enforce-eager
