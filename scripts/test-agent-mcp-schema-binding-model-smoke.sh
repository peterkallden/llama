#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/agent-model-smoke-common.sh"

repo_root=$(agent_smoke_repo_root)
build_dir=$(agent_smoke_build_path)
model="${LLAMA_AGENT_MODEL:-${HOME}/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf}"
daemon="$build_dir/bin/llama-agent-daemon"
fake_mcp="$build_dir/bin/llama-agent-mcp-stdio-fake-server"
fixture="pocs/agent/smoke/data/fixtures/research/TAB6623_sv_sample.csv"
work_dir="${LLAMA_AGENT_SMOKE_WORKDIR:-${HOME}/llama-agent-smoke/mcp-schema-binding}"
log_path="$work_dir/mcp-schema-binding.log"
timeout_seconds="${LLAMA_AGENT_TIMEOUT_SECONDS:-1800}"

agent_smoke_require_file "$model" "Qwen Instruct model"
agent_smoke_require_file "$fixture" "CSV fixture"
agent_smoke_require_executable "$daemon" "agent daemon"
agent_smoke_require_executable "$fake_mcp" "fake MCP server"
mkdir -p "$work_dir"
: > "$log_path"

coproc AGENT_DAEMON {
    GGML_VK_VISIBLE_DEVICES="${GGML_VK_VISIBLE_DEVICES:-0}" \
        LLAMA_AGENT_RESIDENT_TRACE=1 \
        timeout --foreground "$timeout_seconds" "$daemon" \
            --model "$model" \
            --default-mode agent \
            --thinking-mode deliberate \
            --tool-profile analysis \
            --repository-root "$repo_root" \
            --mcp-tool-command "$fake_mcp" \
            --mcp-tool-server-name github \
            --mcp-tool-prefix github \
            --agent-trace \
            --generation-trace \
            --n-predict "${LLAMA_AGENT_N_PREDICT:-256}" \
            --context-size "${LLAMA_AGENT_CONTEXT_SIZE:-4096}" \
            --threads "${LLAMA_AGENT_THREADS:-4}" \
            --max-tool-rounds 8 \
            --n-gpu-layers "${LLAMA_AGENT_GPU_LAYERS:-99}" \
            2>>"$log_path"
}
daemon_pid=$AGENT_DAEMON_PID
daemon_in=${AGENT_DAEMON[1]}
daemon_out=${AGENT_DAEMON[0]}

cleanup() {
    if kill -0 "$daemon_pid" 2>/dev/null; then
        printf '%s\n' '{"command":"shutdown","request_id":"mcp-smoke-shutdown"}' >&"$daemon_in" 2>/dev/null || true
        wait "$daemon_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

read_protocol_reply() {
    local line
    while IFS= read -r -t "$timeout_seconds" line <&"$daemon_out"; do
        printf '%s\n' "$line" | tee -a "$log_path" >/dev/null
        if jq -e '(.message_type == "event") or (.type == "event")' >/dev/null 2>&1 <<<"$line"; then
            continue
        fi
        REPLY=$line
        return 0
    done
    echo "timed out waiting for daemon protocol response; log=$log_path" >&2
    return 1
}

send_turn() {
    printf '%s\n' "$1" >&"$daemon_in"
    read_protocol_reply
    TURN_REPLY=$REPLY
}

read_protocol_reply
printf 'daemon_ready=%s\n' "$REPLY" | tee -a "$log_path"
jq -e '.ok == true and .event == "ready"' >/dev/null <<<"$REPLY"

validation_request=$(jq -cn --arg fixture "$fixture" '{request_id:"mcp-dataset-validate",turn_id:"turn-dataset-validate",mode:"agent",session_id:"mcp-schema-smoke",namespace_id:"mcp-smoke",project_id:"mcp-schema-smoke",memory_scope:"session",plan_scope:"turn",prompt:("Execute exactly this one dataset.validate plan; preserve the path exactly and do not add dataset or resource arguments:\nplan goal=\"validate csv\"\nstep | open! dataset.validate\npath: " + $fixture + "\nrules[0].type: not_null\nrules[0].column: inkomstslag\nReport its returned valid and rows_checked values.")}')
send_turn "$validation_request"
validation_reply=$TURN_REPLY
printf 'dataset_validate_reply=%s\n' "$validation_reply" | tee -a "$log_path"
validation_ok=$(jq -r '.ok // false' <<<"$validation_reply")
validation_tools_completed=$(jq -r '.metrics.tools_completed // 0' <<<"$validation_reply")
printf 'dataset_validate_tools_completed=%s\n' "$validation_tools_completed" | tee -a "$log_path"

printf '%s\n' '{"command":"reset_session","session_id":"mcp-schema-smoke","namespace_id":"mcp-smoke"}' >&"$daemon_in"
read_protocol_reply
reset_reply=$REPLY
printf 'session_reset=%s\n' "$reset_reply" | tee -a "$log_path"
jq -e '.ok == true and .event == "session_reset"' >/dev/null <<<"$reset_reply"

binding_request=$(jq -cn '{request_id:"mcp-result-binding",turn_id:"turn-mcp-result-binding",mode:"agent",session_id:"mcp-schema-smoke",namespace_id:"mcp-smoke",project_id:"mcp-schema-smoke",memory_scope:"session",plan_scope:"turn",prompt:"Call github_search_issues with query output-schema-smoke, naming that step issues. Then call repository.search with query bound to the title field returned by the issues step using the DSL reference $issues.title. Do not invent or paraphrase the title. Report the returned files."}')
send_turn "$binding_request"
binding_reply=$TURN_REPLY
printf 'mcp_result_binding_reply=%s\n' "$binding_reply" | tee -a "$log_path"
binding_ok=$(jq -r '.ok // false' <<<"$binding_reply")
binding_tools_completed=$(jq -r '.metrics.tools_completed // 0' <<<"$binding_reply")
binding_tool_delta=$((binding_tools_completed - validation_tools_completed))
printf 'mcp_result_binding_tools_completed=%s delta=%s\n' "$binding_tools_completed" "$binding_tool_delta" | tee -a "$log_path"

printf '%s\n' '{"command":"shutdown","request_id":"mcp-smoke-shutdown"}' >&"$daemon_in"
read_protocol_reply
printf 'daemon_shutdown=%s\n' "$REPLY" | tee -a "$log_path"
wait "$daemon_pid"
trap - EXIT

echo "dataset_validate_status=$validation_ok"
echo "mcp_result_binding_status=$binding_ok"
if [[ "$validation_ok" != true || "$binding_ok" != true || "$validation_tools_completed" -lt 1 || "$binding_tool_delta" -lt 2 ]]; then
    echo "mcp_schema_binding_model_smoke=incomplete_or_failed"
    echo "trace=$log_path"
    exit 1
fi

grep -Fq '"tool_name":"dataset.validate"' "$log_path" || {
    echo "model smoke did not show dataset.validate in the trace; log=$log_path" >&2
    exit 1
}
grep -Fq '"tool_name":"github_search_issues"' "$log_path" || {
    echo "model smoke did not call the MCP issue-search tool; log=$log_path" >&2
    exit 1
}
grep -Fq '"tool_name":"repository.search"' "$log_path" || {
    echo "model smoke did not call the downstream repository search; log=$log_path" >&2
    exit 1
}
grep -Fq 'stub issue' "$log_path" || {
    echo "model smoke trace lacks the MCP result used for binding; log=$log_path" >&2
    exit 1
}

echo "mcp_schema_binding_model_smoke=passed"
echo "trace=$log_path"
