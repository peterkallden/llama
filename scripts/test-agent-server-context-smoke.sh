#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/agent-model-smoke-common.sh"
repo_root=$(agent_smoke_repo_root); build_dir=$(agent_smoke_build_dir); model=$(agent_smoke_model)
work_dir=$(agent_smoke_prepare_workdir server-context); agent_bin="${repo_root}/${build_dir}/bin/llama-agent"
resident_bin="${repo_root}/${build_dir}/bin/llama-agent-resident-smoke"
agent_smoke_require_file "$model" "chat model"; agent_smoke_require_executable "$resident_bin" "resident smoke executable"
agent_smoke_build_if_requested
log_path="${work_dir}/server-context.log"
agent_smoke_run_logged "$log_path" "$resident_bin" --model "$model" --first-prompt "${LLAMA_AGENT_FIRST_PROMPT:-Reply with OK only.}" --second-prompt "${LLAMA_AGENT_SECOND_PROMPT:-Reply with DONE only.}" --n-predict "${LLAMA_AGENT_N_PREDICT:-8}" --second-n-predict "${LLAMA_AGENT_SECOND_N_PREDICT:-8}" -ngl 0
grep -Eiq 'ok' "$log_path" || { echo "server-context smoke did not contain OK; log=${log_path}" >&2; exit 1; }
echo "server_context_smoke=passed"; echo "log=${log_path}"
