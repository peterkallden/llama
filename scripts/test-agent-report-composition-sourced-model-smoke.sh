#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/agent-model-smoke-common.sh"

repo_root=$(agent_smoke_repo_root)
build_dir=$(agent_smoke_build_dir)
model="${LLAMA_AGENT_MODEL:-${HOME}/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf}"
embedding_model="${LLAMA_AGENT_EMBEDDING_MODEL:-${HOME}/models/nomic-embed-text-v1.5.Q4_K_M.gguf}"
if [[ -n "${LLAMA_AGENT_SMOKE_WORKDIR:-}" ]]; then
    work_dir=$(agent_smoke_prepare_workdir report-composition-sourced)
else
    work_dir="${HOME}/llama-agent-smokes/report-composition-sourced-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$work_dir"
fi
agent_bin=$(agent_smoke_binary "$build_dir" llama-agent)
gpu_layers="${LLAMA_AGENT_GPU_LAYERS:-99}"
log_path="$work_dir/report-composition-sourced.log"
fixture_a="$repo_root/tests/fixtures/report-research-smoke.txt"
fixture_b="$repo_root/tests/fixtures/report-research-smoke-second.txt"
session_id="report-composition-sourced-smoke"
storage_args=(
    --backend cozo --memory-db "$work_dir/memory.db"
    --plan-backend cozo --plan-db "$work_dir/plans.db" --plan-scope session
    --memory-scope session --memory-session "$session_id"
    --resource-blob-backend fs --resource-blob-root "$work_dir/resource-blobs"
    --resource-metadata-backend cozo --resource-metadata-db "$work_dir/resource-metadata.db"
)

agent_smoke_require_file "$model" "Qwen Instruct model"
agent_smoke_require_file "$embedding_model" "embedding model"
agent_smoke_require_file "$fixture_a" "synthetic research fixture"
agent_smoke_require_file "$fixture_b" "second synthetic research fixture"
agent_smoke_require_executable "$agent_bin" "llama-agent"
agent_smoke_build_if_requested llama-agent

prompt='Using only the two supplied synthetic records, report the date they both state for the sample mission. Cite both evidence handles and both source handles for this comparison. Explicitly treat the records as synthetic fixtures, not historical authorities.'
LLAMA_AGENT_RESIDENT_TRACE="${LLAMA_AGENT_RESIDENT_TRACE:-1}" \
    LLAMA_AGENT_TIMEOUT_SECONDS="${LLAMA_AGENT_TIMEOUT_SECONDS:-1800}" \
    agent_smoke_run_logged "$log_path" "$agent_bin" run \
    "${storage_args[@]}" \
    --model "$model" --embedding-model "$embedding_model" \
    --agent-profile research --tool-profile research \
    --agent-bootstrap none --agent-import "$repo_root/docs/examples/agent-bootstrap-workflows-v1.json" \
    --agent-blueprint research-report-composition-v1 \
    --agent-inference-backend server-context --tool-output-format dsl \
    --thinking-mode research --memory-learn off --max-research-iterations 4 --max-tool-rounds 8 \
    --resource "$fixture_a" --resource "$fixture_b" --resource-mime-type text/plain \
    --agent-trace --generation-trace --plan-show-summary \
    --prompt "$prompt" --n-predict "${LLAMA_AGENT_N_PREDICT:-512}" \
    --planner-n-predict "${LLAMA_AGENT_PLANNER_N_PREDICT:-256}" \
    --context-size "${LLAMA_AGENT_CONTEXT_SIZE:-4096}" \
    --threads "${LLAMA_AGENT_THREADS:-4}" -ngl "$gpu_layers"

if [[ "$gpu_layers" =~ ^[0-9]+$ ]] && (( gpu_layers >= 99 )); then
    grep -Fq "requested_gpu_layers=$gpu_layers effective_gpu_layers=-1 fit_params=true workspace_reservation=true legacy_full_offload_fit=true" "$log_path" || {
        echo "sourced report smoke did not prove resident GPU auto-fit; log=$log_path" >&2
        exit 1
    }
fi
grep -Fq 'artifact source snapshot captured before report composition' "$log_path" || {
    echo "sourced report smoke did not freeze its source snapshot; log=$log_path" >&2
    exit 1
}
grep -Fq '"evidence_handle":"E1"' "$log_path" || {
    echo "sourced report smoke did not expose evidence records to section generation; log=$log_path" >&2
    exit 1
}
grep -Fq '"source_handle":"S1"' "$log_path" || {
    echo "sourced report smoke did not preserve evidence-to-source handles; log=$log_path" >&2
    exit 1
}
grep -Eq 'artifact section validated and recorded' "$log_path" || {
    echo "sourced report smoke did not validate a generated section; log=$log_path" >&2
    exit 1
}
section_outputs=$(awk '
    /agent generation trace output: event=nonstream-success purpose=artifact_section/ { capture=1; next }
    capture && /agent generation trace output: end/ { capture=0; next }
    capture { print }
' "$log_path")
grep -Fq 'status: findings' <<<"$section_outputs" || {
    echo "sourced report smoke did not produce a model-generated findings section; log=$log_path" >&2
    exit 1
}
grep -Fq '20 July 1969' <<<"$section_outputs" || {
    echo "sourced report smoke did not generate the fixture-supported date in the section; log=$log_path" >&2
    exit 1
}
grep -Eq '^text: .{1,160}[.!?]$' <<<"$section_outputs" || {
    echo "sourced report smoke did not produce a complete bounded sentence; log=$log_path" >&2
    exit 1
}
grep -Eq 'evidence: E1([,[:space:]]|$)' <<<"$section_outputs" || {
    echo "sourced report smoke did not cite the first host evidence handle; log=$log_path" >&2
    exit 1
}
grep -Eq 'evidence: .*E2([,[:space:]]|$)' <<<"$section_outputs" || {
    echo "sourced report smoke did not cite the second host evidence handle; log=$log_path" >&2
    exit 1
}
grep -Eq 'source: S1([,[:space:]]|$)' <<<"$section_outputs" || {
    echo "sourced report smoke did not cite the first host source handle; log=$log_path" >&2
    exit 1
}
grep -Eq 'source: .*S3([,[:space:]]|$)' <<<"$section_outputs" || {
    echo "sourced report smoke did not cite the second host source handle; log=$log_path" >&2
    exit 1
}
grep -Eq 'stage=tool kind=(succeeded|completed).*tool=artifact[.]export' "$log_path" || {
    echo "sourced report smoke did not publish through artifact.export; log=$log_path" >&2
    exit 1
}
grep -Eq '(report_resource: "|Structured report: )agent-resource://resource/[^"[:space:]]+' "$log_path" || {
    echo "sourced report smoke did not expose the persisted structured report resource; log=$log_path" >&2
    exit 1
}
grep -Eq '(markdown_resource: "|Markdown: )agent-resource://resource/[^"[:space:]]+' "$log_path" || {
    echo "sourced report smoke did not expose the persisted Markdown resource; log=$log_path" >&2
    exit 1
}
report_uri=$(grep -oE '(report_resource: "|Structured report: )agent-resource://resource/[^"[:space:]]+' "$log_path" | head -n1 | sed -E 's/^[^:]+: "?//')
markdown_uri=$(grep -oE '(markdown_resource: "|Markdown: )agent-resource://resource/[^"[:space:]]+' "$log_path" | head -n1 | sed -E 's/^[^:]+: "?//')
readback_logs=()
mcp_server=$(agent_smoke_binary "$build_dir" llama-agent-mcp-stdio-server)
agent_smoke_require_executable "$mcp_server" "llama-agent-mcp-stdio-server"
readback_log="$work_dir/report-composition-mcp-readback.log"
readback_logs+=("$readback_log")
{
    printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"report-smoke-readback","version":"1"}}}'
    printf '%s\n' '{"jsonrpc":"2.0","method":"notifications/initialized"}'
    printf '{"jsonrpc":"2.0","id":2,"method":"resources/read","params":{"uri":"%s"}}\n' "$report_uri"
    printf '{"jsonrpc":"2.0","id":3,"method":"resources/read","params":{"uri":"%s"}}\n' "$markdown_uri"
    printf '%s\n' '{"jsonrpc":"2.0","id":4,"method":"shutdown"}' '{"jsonrpc":"2.0","method":"exit"}'
} | timeout --foreground "${LLAMA_AGENT_TIMEOUT_SECONDS:-1800}" "$mcp_server" \
    --backend cozo --memory-db "$work_dir/memory.db" \
    --memory-namespace local --memory-session "$session_id" \
    --resource-blob-backend fs --resource-blob-root "$work_dir/resource-blobs" \
    --resource-metadata-backend cozo --resource-metadata-db "$work_dir/resource-metadata.db" \
    >"$readback_log" 2>&1 || {
        echo "persistent report smoke MCP resource readback failed; log=$readback_log" >&2
        exit 1
    }
report_text=$(jq -R -s -r --arg uri "$report_uri" 'split("\n") | map(fromjson? | select(.id == 2 and .result.contents[0].uri == $uri))[0].result.contents[0].text' "$readback_log")
markdown_text=$(jq -R -s -r --arg uri "$markdown_uri" 'split("\n") | map(fromjson? | select(.id == 3 and .result.contents[0].uri == $uri))[0].result.contents[0].text' "$readback_log")
grep -Fq '20 July 1969' <<<"$report_text" || {
    echo "persistent report smoke readback did not expose the fixture-supported date; log=$readback_log" >&2
    exit 1
}
grep -Fq 'source_bibliography' <<<"$report_text" || {
    echo "persistent report smoke readback did not expose the deterministic bibliography; log=$readback_log" >&2
    exit 1
}
grep -Fq '20 July 1969' <<<"$markdown_text" || {
    echo "persistent Markdown readback did not expose the fixture-supported date; log=$readback_log" >&2
    exit 1
}
grep -Fq '## Sources' <<<"$markdown_text" || {
    echo "persistent Markdown readback did not expose its source bibliography; log=$readback_log" >&2
    exit 1
}
echo "agent_report_composition_sourced_model_smoke=passed"
echo "log=$log_path"
printf 'readback_log=%s\n' "${readback_logs[@]}"
