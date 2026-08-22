#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/agent-model-smoke-common.sh"
repo_root=$(agent_smoke_repo_root); build_dir=$(agent_smoke_build_dir); model=$(agent_smoke_model)
embedding_model="${LLAMA_AGENT_EMBEDDING_MODEL:-${HOME}/models/nomic-embed-text-v1.5.Q4_K_M.gguf}"
work_dir=$(agent_smoke_prepare_workdir qwen-nomic-data); agent_bin="${repo_root}/${build_dir}/bin/llama-agent"; seed_bin="${repo_root}/${build_dir}/bin/llama-agent-data-store-cozo-seed"
agent_smoke_require_file "$model" "chat model"; agent_smoke_require_file "$embedding_model" "embedding model"; agent_smoke_require_executable "$agent_bin" "llama-agent"; agent_smoke_require_executable "$seed_bin" "Cozo seed executable"
agent_smoke_build_if_requested
orders="$work_dir/orders.csv"; customers="$work_dir/customers.csv"; data_db="$work_dir/data.cozo"
printf '%s\n' 'order_id,customer_id,amount' '1,10,12' '2,10,8' '3,11,20' > "$orders"
printf '%s\n' 'customer_id,segment' '10,enterprise' '11,consumer' > "$customers"
seed_log="$work_dir/seed.log"; agent_log="$work_dir/data-research.log"
agent_smoke_run_logged "$seed_log" "$seed_bin" --db "$data_db" --orders "$orders" --customers "$customers"
grep -Eq 'seeded_orders=[1-9]' "$seed_log"; grep -Eq 'seeded_customers=[1-9]' "$seed_log"
prompt='Perform a bounded data-analysis task using the two CSV datasets in the workspace. Discover and inspect them, join orders with customers on customer_id using data.join on:[{left:customer_id,right:customer_id}], aggregate the joined rows by segment using data.aggregate measures:[{function:sum,column:amount}], and use statistics.describe on amount. Report the exact total 40 and mention the tools/results used. Do not guess; use the data tools.'
# The planner reserves at least 512 tokens and reflection reserves 384.  Keep
# enough draft/context/time budget for the CPU smoke to complete after the
# tool chain, especially when tracing is enabled.
LLAMA_AGENT_TIMEOUT_SECONDS="${LLAMA_AGENT_TIMEOUT_SECONDS:-900}" agent_smoke_run_logged "$agent_log" "$agent_bin" run --backend cozo --memory-db "$work_dir/memory.cozo" --plan-backend cozo --plan-db "$work_dir/plan.cozo" --data-backend cozo --data-db "$data_db" --model "$model" --embedding-model "$embedding_model" --agent-profile research --tool-profile analysis --thinking-mode "${LLAMA_AGENT_THINKING_MODE:-reflective}" --max-reflection-rounds 2 --max-research-iterations 1 --agent-plan auto --max-plan-revisions 3 --repository-root "$work_dir" --max-tool-rounds 16 --memory-project qwen-nomic-data --plan-scope project --agent-trace --generation-trace --require-tool-execution --plan-show-summary --prompt "$prompt" --n-predict "${LLAMA_AGENT_N_PREDICT:-256}" --context-size "${LLAMA_AGENT_CONTEXT_SIZE:-4096}" --threads "${LLAMA_AGENT_THREADS:-4}" -ngl 0
grep -Eq 'stage=tool kind=succeeded .*tool=data.join' "$agent_log"; grep -Eq 'stage=tool kind=succeeded .*tool=data.aggregate' "$agent_log"; grep -Fq '40' "$agent_log"
echo "qwen_nomic_data=passed"; echo "log=${agent_log}"
