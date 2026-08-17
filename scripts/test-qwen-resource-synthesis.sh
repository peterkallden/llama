#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/agent-model-smoke-common.sh"
repo_root=$(agent_smoke_repo_root); build_dir=$(agent_smoke_build_dir); model=$(agent_smoke_model)
fixture="${LLAMA_AGENT_RESOURCE_SYNTHESIS_FIXTURE:-${repo_root}/tests/data/agent-resource-synthesis.txt}"
work_dir=$(agent_smoke_prepare_workdir qwen-resource-synthesis); agent_bin="${repo_root}/${build_dir}/bin/llama-agent"
agent_smoke_require_file "$model" "chat model"; agent_smoke_require_file "$fixture" "resource synthesis fixture"; agent_smoke_require_executable "$agent_bin" "llama-agent"
agent_smoke_build_if_requested
mapfile -t paragraphs < <(perl -0pe 's/\r\n/\n/g' "$fixture" | awk -v RS='\n\n' '{ gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print }')
if [[ ${#paragraphs[@]} -ne 4 ]]; then echo "expected four synthesis chunks, got ${#paragraphs[@]}" >&2; exit 1; fi
prompt='Synthesize these four resource observations into one concise answer. Preserve every factual requirement. Include all three binary variants, the test gate, the Agent CI package requirement, and the three separate test result categories. Do not invent facts and do not mention chunking.'
for i in "${!paragraphs[@]}"; do prompt+=$'\n\n[chunk '"$((i + 1))"'/4]\n'"${paragraphs[$i]}"; done
log_path="${work_dir}/resource-synthesis.log"
agent_smoke_run_logged "$log_path" "$agent_bin" run --model "$model" --agent-profile static --agent-bootstrap none --agent-plan off --agent-inference-backend cli --n-predict "${LLAMA_AGENT_N_PREDICT:-96}" --threads "${LLAMA_AGENT_THREADS:-4}" -ngl 0 --prompt "$prompt"
for required in CPU CUDA Vulkan 'Agent CI' passed failed; do grep -Fqi "$required" "$log_path" || { echo "resource synthesis missing: $required" >&2; exit 1; }; done
grep -Eiq 'not[- ]run' "$log_path" || { echo "resource synthesis missing not-run category" >&2; exit 1; }
echo "qwen_resource_synthesis=passed"; echo "resource_chunks=4"; echo "log=${log_path}"
