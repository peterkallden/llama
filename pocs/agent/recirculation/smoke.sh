#!/usr/bin/env bash
set -euo pipefail
# Usage: bash smoke.sh BUILD_DIR TRACE_DIR [MODEL.gguf [EVALUATION_TEXT]]
build_dir=${1:?build directory required}
trace_dir=${2:?trace directory required}
model=${3:-}
mkdir -p "$trace_dir"
ctest --test-dir "$build_dir" -R '^(llama-agent-recirculation-(contract|runner|memory|window|analysis)-ctest|test-agent-flydelta-recirculation-memory|test-agent-prepared-generation)$' --no-tests=error --output-on-failure 2>&1 | tee "$trace_dir/contracts.log"
git -C "$(dirname "$0")/../../.." rev-parse HEAD > "$trace_dir/source-revision.txt"
repo_dir=$(git -C "$(dirname "$0")/../../.." rev-parse --show-toplevel)
git -C "$repo_dir" status --short > "$trace_dir/source-status.txt"
git -C "$repo_dir" diff --binary > "$trace_dir/source-tracked.patch"
tar -czf "$trace_dir/poc-sources.tar.gz" -C "$repo_dir" \
    pocs/agent/CMakeLists.txt \
    pocs/agent/recirculation/CMakeLists.txt pocs/agent/recirculation/README.md \
    pocs/agent/recirculation/runner.h pocs/agent/recirculation/llama-backend.h pocs/agent/recirculation/window.h \
    pocs/agent/recirculation/experiment.h pocs/agent/recirculation/data/development.jsonl \
    pocs/agent/recirculation/data/evaluation.jsonl \
    pocs/agent/recirculation/main.cpp pocs/agent/recirculation/smoke.sh \
    pocs/agent/recirculation/resident-smoke.cpp common/agent/agent-recirculation.h \
    common/agent/adaptation/flydelta/flydelta-recirculation-memory.h \
    common/agent/adaptation/flydelta/flydelta-recirculation-memory.cpp \
    tests/test-agent-flydelta-recirculation-memory.cpp \
    pocs/agent/recirculation/test-runner.cpp pocs/agent/recirculation/test-memory.cpp pocs/agent/recirculation/test-window.cpp \
    pocs/agent/recirculation-contract.h pocs/agent/test-recirculation-contract.cpp
if [[ -z "$model" ]]; then
    echo '{"event":"model_smoke","status":"skipped","reason":"no_model"}' | tee "$trace_dir/model-status.json"
    exit 0
fi
args=(--model "$model" --trace "$trace_dir/model.jsonl" --steps 128)
if [[ -n "${4:-}" ]]; then args+=(--text-file "$4"); fi
"$build_dir/bin/llama-recirculation-poc" "${args[@]}" > "$trace_dir/model.stdout.log" 2> "$trace_dir/model.stderr.log"
"$build_dir/bin/llama-recirculation-resident-smoke" "$model" > "$trace_dir/resident.stdout.log" 2> "$trace_dir/resident.stderr.log"
