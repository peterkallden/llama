#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/agent-model-smoke-common.sh"
repo_root=$(agent_smoke_repo_root)
build_dir=$(agent_smoke_build_dir)
model="${LLAMA_AGENT_MODEL:-}"
if [[ $# -gt 0 ]]; then
    if [[ "$1" != "--model" || $# -ne 2 ]]; then
        echo "usage: $0 [--model PATH]" >&2
        exit 2
    fi
    model="$2"
fi
queue_bin=$(agent_smoke_binary "$build_dir" llama-agent-flydelta-plateau-queue-smoke)
replay_bin=$(agent_smoke_binary "$build_dir" llama-agent-flydelta-repair-echo-replay-smoke)
model_bin=$(agent_smoke_binary "$build_dir" llama-agent-flydelta-dataset-question-repair-model-smoke)
replay_suite="$repo_root/docs/examples/agent-flydelta-dataset-repair-echo-replay.jsonl"
suite="${LLAMA_AGENT_DATASET_QUESTION_SUITE:-$repo_root/pocs/agent/smoke/adaptation/fixtures/tfo-dose-model.jsonl}"
agent_smoke_require_executable "$queue_bin" "FlyDelta queue smoke"
agent_smoke_require_executable "$replay_bin" "FlyDelta replay smoke"
"$queue_bin"
"$replay_bin" "$replay_suite"
if [[ -z "$model" ]]; then
    echo 'tfo_direction_dose_model=skipped_no_model'
    exit 0
fi
agent_smoke_require_file "$model" "FlyDelta TFO model"
agent_smoke_require_executable "$model_bin" "FlyDelta TFO model smoke"
model_digest=$(sha256sum -- "$model")
export LLAMA_AGENT_DOSE_MODEL_FINGERPRINT="sha256:${model_digest%% *}"
LLAMA_AGENT_DOSE_EXPERIMENT=direction-dose "$model_bin" --model "$model" --suite "$suite" \
    --dose-comparison --force-tfo-lite --production-bindings \
    --behavior-key tool_use/dataset/data_aggregate --max-scenarios 2 \
    --baseline-repeats "${LLAMA_AGENT_BASELINE_REPEATS:-3}" --n-predict "${LLAMA_AGENT_N_PREDICT:-512}" \
    --threads "${LLAMA_AGENT_THREADS:-3}" --n-gpu-layers "${LLAMA_AGENT_GPU_LAYERS:--1}" \
    --n-parallel "${LLAMA_AGENT_N_PARALLEL:-2}" --n-ctx-per-sequence "${LLAMA_AGENT_N_CTX_PER_SEQUENCE:-6144}"
