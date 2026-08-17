#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/agent-model-smoke-common.sh"
repo_root=$(agent_smoke_repo_root); build_dir=$(agent_smoke_build_dir); model=$(agent_smoke_model)
daemon_bin="${repo_root}/${build_dir}/bin/llama-agent-daemon"
agent_smoke_require_file "$model" "chat model"; agent_smoke_require_executable "$daemon_bin" "agent daemon"
agent_smoke_build_if_requested
work_dir=$(agent_smoke_prepare_workdir daemon-integration); log_path="$work_dir/daemon.log"
exec 3>"$log_path"
coproc AGENT_DAEMON { "$daemon_bin" --model "$model" --default-mode chat -n "${LLAMA_AGENT_N_PREDICT:-16}" -ngl 0; }
daemon_pid=$AGENT_DAEMON_PID
cleanup() { printf '%s\n' '{"command":"shutdown","request_id":"linux-shutdown"}' >&"${AGENT_DAEMON[1]}" 2>/dev/null || true; kill "$daemon_pid" 2>/dev/null || true; wait "$daemon_pid" 2>/dev/null || true; exec 3>&-; }
trap cleanup EXIT
read_response() {
    local line
    while IFS= read -r -t "${LLAMA_AGENT_DAEMON_TIMEOUT:-120}" line <&"${AGENT_DAEMON[0]}"; do
        printf '%s\n' "$line" >&3
        [[ -n "$line" ]] || continue
        if jq -e . >/dev/null 2>&1 <<<"$line"; then
            if jq -e '.message_type == "event"' >/dev/null 2>&1 <<<"$line"; then continue; fi
            printf '%s\n' "$line"; return 0
        fi
        echo "daemon emitted non-JSON output" >&2; return 1
    done
    echo "timed out waiting for daemon response" >&2; return 1
}
send_command() { printf '%s\n' "$1" >&"${AGENT_DAEMON[1]}"; read_response; }
ready=$(read_response); jq -e '.ok == true and .event == "ready"' <<<"$ready" >/dev/null
turn1=$(send_command '{"request_id":"linux-turn-1","mode":"chat","prompt":"Reply with OK only.","session_id":"linux-session","namespace_id":"linux","project_id":"linux-project","memory_scope":"project","plan_scope":"project"}')
jq -e '.ok == true and .response == "OK" and .runtime_reused == false' <<<"$turn1" >/dev/null
turn2=$(send_command '{"request_id":"linux-turn-2","mode":"chat","prompt":"Reply with DONE only.","session_id":"linux-session","namespace_id":"linux","project_id":"linux-project","memory_scope":"project","plan_scope":"project"}')
jq -e '.ok == true and .response == "DONE" and .runtime_reused == true' <<<"$turn2" >/dev/null
status=$(send_command '{"command":"status","request_id":"linux-status"}')
jq -e '.ok == true and .sessions == 1' <<<"$status" >/dev/null
echo "daemon_chat_lifecycle=passed"; echo "daemon_runtime_reuse=passed"; echo "log=${log_path}"
