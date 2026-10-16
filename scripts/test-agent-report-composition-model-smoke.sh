#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/agent-model-smoke-common.sh"

repo_root=$(agent_smoke_repo_root)
build_dir=$(agent_smoke_build_dir)
model="${LLAMA_AGENT_MODEL:-${HOME}/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf}"
embedding_model="${LLAMA_AGENT_EMBEDDING_MODEL:-${HOME}/models/nomic-embed-text-v1.5.Q4_K_M.gguf}"
if [[ -n "${LLAMA_AGENT_SMOKE_WORKDIR:-}" ]]; then
    work_dir=$(agent_smoke_prepare_workdir report-composition)
else
    work_dir="${HOME}/llama-agent-smokes/report-composition-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$work_dir"
fi
agent_bin=$(agent_smoke_binary "$build_dir" llama-agent)
gpu_layers="${LLAMA_AGENT_GPU_LAYERS:-99}"

agent_smoke_require_file "$model" "Qwen Instruct model"
agent_smoke_require_file "$embedding_model" "embedding model"
agent_smoke_require_executable "$agent_bin" "llama-agent"
agent_smoke_require_file "$repo_root/docs/examples/agent-bootstrap-workflows-v1.json" "workflow package"
agent_smoke_build_if_requested llama-agent

log_path="$work_dir/report-composition.log"
prompt='Compose a research report from the evidence available to this plan about whether the source material supports a conclusion. If the frozen source snapshot has no evidence, say that coverage is incomplete and do not invent findings or citations.'
LLAMA_AGENT_RESIDENT_TRACE="${LLAMA_AGENT_RESIDENT_TRACE:-1}" \
    LLAMA_AGENT_TIMEOUT_SECONDS="${LLAMA_AGENT_TIMEOUT_SECONDS:-1800}" \
    agent_smoke_run_logged "$log_path" "$agent_bin" run \
    --backend in-memory --plan-backend in-memory \
    --model "$model" --embedding-model "$embedding_model" \
    --agent-profile default --tool-profile analysis \
    --agent-bootstrap none --agent-import "$repo_root/docs/examples/agent-bootstrap-workflows-v1.json" \
    --agent-blueprint research-report-composition-v1 \
    --agent-inference-backend server-context --tool-output-format dsl \
    --thinking-mode deliberate --memory-learn off --max-tool-rounds 4 \
    --agent-trace --generation-trace --plan-show-summary \
    --prompt "$prompt" --n-predict "${LLAMA_AGENT_N_PREDICT:-256}" \
    --context-size "${LLAMA_AGENT_CONTEXT_SIZE:-4096}" \
    --threads "${LLAMA_AGENT_THREADS:-4}" -ngl "$gpu_layers"

if [[ "$gpu_layers" =~ ^[0-9]+$ ]] && (( gpu_layers >= 99 )); then
    grep -Fq "requested_gpu_layers=$gpu_layers effective_gpu_layers=-1 fit_params=true workspace_reservation=true legacy_full_offload_fit=true" "$log_path" || {
        echo "report composition smoke did not prove resident GPU auto-fit; log=$log_path" >&2
        exit 1
    }
fi
grep -Fq 'artifact source snapshot captured before report composition' "$log_path" || {
    echo "report composition smoke did not freeze its source snapshot; log=$log_path" >&2
    exit 1
}
grep -Eq 'artifact section validated and recorded' "$log_path" || {
    echo "report composition smoke did not validate a generated section; log=$log_path" >&2
    exit 1
}
grep -Eq 'stage=tool kind=(succeeded|completed).*tool=artifact[.]export' "$log_path" || {
    echo "report composition smoke did not publish through artifact.export; log=$log_path" >&2
    exit 1
}
grep -Eq '(report_resource: "|Structured report: )agent-resource://resource/[^"[:space:]]+' "$log_path" || {
    echo "report composition smoke did not expose the persisted structured report resource; log=$log_path" >&2
    exit 1
}
grep -Eq '(markdown_resource: "|Markdown: )agent-resource://resource/[^"[:space:]]+' "$log_path" || {
    echo "report composition smoke did not expose the persisted Markdown resource; log=$log_path" >&2
    exit 1
}
grep -Fq 'host-generated completion summary from verified report export' "$log_path" || {
    echo "report composition smoke did not use the host-owned completion summary; log=$log_path" >&2
    exit 1
}
if grep -Fq 'purpose=draft' "$log_path"; then
    echo "report composition smoke unexpectedly generated a free-form second report; log=$log_path" >&2
    exit 1
fi
echo "agent_report_composition_model_smoke=passed"
echo "log=$log_path"
