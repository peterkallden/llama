#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/agent-model-smoke-common.sh"

repo_root=$(agent_smoke_repo_root)
build_dir=$(agent_smoke_build_dir)
model="${LLAMA_AGENT_MODEL:-$(agent_smoke_model)}"
if [[ "$model" != /* ]]; then model="${repo_root}/${model}"; fi
work_dir=$(agent_smoke_prepare_workdir semi-oracle-model)
smoke_bin=$(agent_smoke_binary "$build_dir" llama-agent-semi-oracle-model-smoke)

agent_smoke_require_file "$model" "Qwen Instruct model"
agent_smoke_require_executable "$smoke_bin" "semi-oracle model smoke"

log_path="$work_dir/semi-oracle-model.log"
LLAMA_AGENT_TIMEOUT_SECONDS="${LLAMA_AGENT_TIMEOUT_SECONDS:-1800}" \
    agent_smoke_run_logged "$log_path" "$smoke_bin" \
        --model "$model" \
        --threads "${LLAMA_AGENT_THREADS:-3}" \
        --n-predict "${LLAMA_AGENT_N_PREDICT:-128}" \
        --n-gpu-layers "${LLAMA_AGENT_GPU_LAYERS:-99}" \
        --strict
grep -Fq 'semi_oracle_model_smoke model=' "$log_path"
echo "semi_oracle_model_smoke=passed"
echo "log=${log_path}"
