#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/agent-model-smoke-common.sh"

repo_root=$(agent_smoke_repo_root)
build_dir=$(agent_smoke_build_dir)
model=$(agent_smoke_model)
suite="${LLAMA_AGENT_DATASET_QUESTION_SUITE:-${repo_root}/docs/examples/agent-flydelta-dataset-question-suite.json}"
work_dir=$(agent_smoke_prepare_workdir flydelta-dataset-question-model)

if [[ "$build_dir" = /* ]]; then
    smoke_bin="${build_dir}/bin/llama-agent-flydelta-dataset-question-model-smoke"
else
    smoke_bin="${repo_root}/${build_dir}/bin/llama-agent-flydelta-dataset-question-model-smoke"
fi

agent_smoke_require_file "$model" "chat model"
agent_smoke_require_file "$suite" "FlyDelta dataset question suite"
agent_smoke_require_executable "$smoke_bin" "FlyDelta dataset question model smoke"

agent_smoke_build_if_requested

args=(
    --model "$model"
    --suite "$suite"
    --threads "${LLAMA_AGENT_THREADS:-3}"
    --n-predict "${LLAMA_AGENT_N_PREDICT:-96}"
    --n-gpu-layers "${LLAMA_AGENT_GPU_LAYERS:-0}"
)
if [[ "${LLAMA_AGENT_STRICT:-0}" == 1 ]]; then
    args+=(--strict)
fi

log_path="$work_dir/dataset-question-model.log"
agent_smoke_run_logged "$log_path" "$smoke_bin" "${args[@]}"
grep -Fq 'flydelta_dataset_question_model_smoke scenarios=' "$log_path"

echo "flydelta_dataset_question_model_smoke=passed"
echo "log=${log_path}"
