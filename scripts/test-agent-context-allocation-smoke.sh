#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=${LLAMA_AGENT_BUILD_DIR:-"$repo_root/build-agent-vulkan-cozo"}
model=${LLAMA_AGENT_MODEL:-}
log_dir=${LLAMA_AGENT_SMOKE_LOG_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/llama-agent-context-smoke.XXXXXX")}
mkdir -p "$log_dir"

ctest --test-dir "$build_dir" -R '^test-agent-context-(allocation|config)$|^test-agent-chat-routing$' --output-on-failure
"$build_dir/bin/llama-agent-deliberate-runtime-smoke"
"$build_dir/bin/llama-agent-research-runtime-smoke"
if [[ -z "$model" ]]; then
    echo "model_smoke=skipped (set LLAMA_AGENT_MODEL to enable)"
    exit 0
fi
[[ -f "$model" ]] || { echo "model not found: $model" >&2; exit 1; }

for context in 0 4096; do
    requests="$log_dir/requests-$context.jsonl"
    cat > "$requests" <<'EOF'
{"mode":"chat","prompt":"Reply with OK only.","session_id":"context-smoke","namespace_id":"smoke"}
{"command":"shutdown"}
EOF
    LLAMA_AGENT_RESIDENT_TRACE=1 /usr/bin/time -f 'elapsed_seconds=%e max_rss_kib=%M' \
        "$build_dir/bin/llama-agent-daemon" --model "$model" --backend in-memory \
        --plan-backend in-memory --default-mode chat --context-size "$context" \
        --n-predict 32 --n-gpu-layers "${LLAMA_AGENT_GPU_LAYERS:--1}" --generation-trace \
        < "$requests" > "$log_dir/stdout-$context.log" 2> "$log_dir/stderr-$context.log"
    python3 - "$log_dir/stdout-$context.log" "$log_dir/stderr-$context.log" <<'PY'
import json
import pathlib
import sys

messages = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines()
            if line.startswith('{')]
if not any(item.get('ok') and item.get('response') for item in messages):
    raise SystemExit('model-backed chat did not complete successfully')
trace = pathlib.Path(sys.argv[2]).read_text()
if 'actual_slot_context=' not in trace or 'model_training_context=' not in trace:
    raise SystemExit('model context capacity trace is missing')
PY
done
long_prompt=$(python3 -c 'print("context " * 300)')
if "$build_dir/bin/llama-agent" run --backend in-memory --plan-backend in-memory \
        --model "$model" --agent-profile static --agent-bootstrap none --agent-plan off \
        --agent-blueprint off --agent-inference-backend cli --context-size 128 \
        --n-predict 16 --threads 4 -ngl "${LLAMA_AGENT_GPU_LAYERS:--1}" \
        --prompt "$long_prompt" > "$log_dir/cli-limited.log" 2>&1; then
    echo "CLI accepted a prompt exceeding its explicit context" >&2
    exit 1
fi
if ! grep -q 'generation context budget exceeded' "$log_dir/cli-limited.log"; then
    echo "CLI failed before the expected context check; see $log_dir/cli-limited.log" >&2
    exit 1
fi
echo "context_model_smoke=passed logs=$log_dir"
