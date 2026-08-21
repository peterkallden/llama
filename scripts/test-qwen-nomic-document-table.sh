#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/agent-model-smoke-common.sh"
repo_root=$(agent_smoke_repo_root); build_dir=$(agent_smoke_build_dir); model=$(agent_smoke_model)
embedding_model="${LLAMA_AGENT_EMBEDDING_MODEL:-${HOME}/models/nomic-embed-text-v1.5.Q4_K_M.gguf}"
document="${LLAMA_AGENT_DOCUMENT_FIXTURE:-${repo_root}/pocs/agent/smoke/data/fixtures/document-table/document-table-model.json}"
work_dir=$(agent_smoke_prepare_workdir qwen-document-table); agent_bin="${repo_root}/${build_dir}/bin/llama-agent"
agent_smoke_require_file "$model" "chat model"; agent_smoke_require_file "$embedding_model" "embedding model"; agent_smoke_require_file "$document" "document fixture"; agent_smoke_require_executable "$agent_bin" "llama-agent"
agent_smoke_build_if_requested
log_path="$work_dir/document-table.log"
prompt='Use the attached JSON document representation. First call document.tables for the attached resource. Then call document.table using the unique table name "Budget summary". Finally call data.aggregate on the returned dataset and calculate the sum of the amount column. Stop after data.aggregate; do not call data.filter, data.transform, or any other tool. Report the table name, dataset reference and total. Use the tools and do not guess. The expected total is 200.'
agent_smoke_run_logged "$log_path" "$agent_bin" run --backend in-memory --plan-backend in-memory --data-backend cozo --data-db "$work_dir/data.cozo" --model "$model" --embedding-model "$embedding_model" --agent-profile research --tool-profile research --thinking-mode "${LLAMA_AGENT_THINKING_MODE:-reflective}" --max-reflection-rounds 1 --max-research-iterations 1 --agent-plan auto --max-plan-revisions 1 --max-tool-rounds "${LLAMA_AGENT_MAX_TOOL_ROUNDS:-4}" --resource-blob-backend fs --resource-blob-root "$work_dir/resources" --resource-metadata-backend in-memory --resource "$document" --resource-mime-type application/json --memory-project qwen-nomic-document-table --plan-scope project --agent-trace --generation-trace --plan-show-summary --prompt "$prompt" --n-predict "${LLAMA_AGENT_N_PREDICT:-128}" --context-size "${LLAMA_AGENT_CONTEXT_SIZE:-3072}" --threads "${LLAMA_AGENT_THREADS:-4}" -ngl 0
for required in document.tables document.table data.aggregate 200; do grep -Fq "$required" "$log_path" || { echo "document-table smoke missing: $required" >&2; exit 1; }; done
if grep -Eq 'kind=failed|tool_call_limit_reached|document representation is unavailable' "$log_path"; then echo "document-table smoke reported a tool failure" >&2; exit 1; fi
echo "qwen_nomic_document_table=passed"; echo "log=${log_path}"
