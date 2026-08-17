#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/agent-model-smoke-common.sh"
repo_root=$(agent_smoke_repo_root); build_dir=$(agent_smoke_build_dir); model=$(agent_smoke_model)
embedding_model="${LLAMA_AGENT_EMBEDDING_MODEL:-${HOME}/models/nomic-embed-text-v1.5.Q4_K_M.gguf}"
work_dir=$(agent_smoke_prepare_workdir qwen-nomic-agent); agent_bin="${repo_root}/${build_dir}/bin/llama-agent"
agent_smoke_require_file "$model" "chat model"; agent_smoke_require_file "$embedding_model" "embedding model"; agent_smoke_require_executable "$agent_bin" "llama-agent"
agent_smoke_build_if_requested
log_path="${work_dir}/qwen-nomic-agent.log"
agent_smoke_run_logged "$log_path" "$agent_bin" run --backend in-memory --plan-backend in-memory --model "$model" --embedding-model "$embedding_model" --agent-profile static --agent-bootstrap none --thinking-mode "${LLAMA_AGENT_THINKING_MODE:-reflective}" --agent-plan off --agent-inference-backend cli --n-predict "${LLAMA_AGENT_N_PREDICT:-16}" --threads "${LLAMA_AGENT_THREADS:-4}" -ngl 0 --prompt "${LLAMA_AGENT_PROMPT:-Say OK after making a tiny plan. Reply with OK only.}"
grep -Eiq 'ok' "$log_path" || { echo "Qwen smoke did not contain OK; log=${log_path}" >&2; exit 1; }
echo "qwen_nomic_agent=passed"; echo "log=${log_path}"
