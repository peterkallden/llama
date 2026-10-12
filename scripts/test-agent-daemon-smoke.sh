#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${LLAMA_AGENT_BUILD_DIR:-build-agent}"
CONFIGURATION="${LLAMA_AGENT_CONFIGURATION:-RelWithDebInfo}"
CHAT_MODEL="${LLAMA_AGENT_CHAT_MODEL:-}"

usage() {
    echo "usage: $0 [--build-dir DIR] [--configuration CONFIG] --chat-model MODEL"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --configuration) CONFIGURATION="$2"; shift 2 ;;
        --chat-model) CHAT_MODEL="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$CHAT_MODEL" ]] || { echo "A chat model is required; set LLAMA_AGENT_CHAT_MODEL or use --chat-model" >&2; exit 2; }
[[ -f "$CHAT_MODEL" ]] || { echo "Chat model not found: $CHAT_MODEL" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ "$BUILD_DIR" = /* ]]; then BUILD_ROOT="$BUILD_DIR"; else BUILD_ROOT="$REPO_ROOT/$BUILD_DIR"; fi

resolve_executable() {
    local target="$1"
    local single_config="$BUILD_ROOT/bin/$target"
    local multi_config="$BUILD_ROOT/bin/$CONFIGURATION/$target"

    if [[ -x "$single_config" ]]; then
        printf '%s\n' "$single_config"
    elif [[ -x "$multi_config" ]]; then
        printf '%s\n' "$multi_config"
    else
        echo "Daemon executable not found in either:" >&2
        echo "  $single_config" >&2
        echo "  $multi_config" >&2
        exit 1
    fi
}

daemon="$(resolve_executable llama-agent-daemon)"
log_root="$(mktemp -d "${TMPDIR:-/tmp}/llama-agent-daemon-smoke.XXXXXX")"
requests="$log_root/requests.jsonl"
stdout_log="$log_root/stdout.log"
stderr_log="$log_root/stderr.log"
trap 'rm -rf "$log_root"' EXIT

cat > "$requests" <<'EOF'
{"mode":"chat","prompt":"Reply with OK only.","session_id":"smoke-session","namespace_id":"smoke","project_id":"repo-smoke","memory_scope":"project","plan_scope":"project"}
{"mode":"chat","prompt":"Reply with DONE only.","session_id":"smoke-session","namespace_id":"smoke","project_id":"repo-smoke","memory_scope":"project","plan_scope":"project"}
{"command":"shutdown"}
EOF

if ! "$daemon" \
    --model "$CHAT_MODEL" \
    --default-mode chat \
    -n 32 \
    -ngl 0 \
    < "$requests" \
    > "$stdout_log" \
    2> "$stderr_log"; then
    echo "Daemon exited unsuccessfully" >&2
    cat "$stderr_log" >&2
    exit 1
fi

python3 - "$stdout_log" <<'PY'
import json
import sys

path = sys.argv[1]
messages = []
with open(path, encoding="utf-8") as stream:
    for raw in stream:
        line = raw.strip()
        if not line.startswith("{"):
            continue
        message = json.loads(line)
        if message.get("message_type") != "event":
            messages.append(message)

if len(messages) != 4:
    raise SystemExit(f"expected exactly 4 protocol messages, got {len(messages)}")

ready, first, second, shutdown = messages
if not ready.get("ok") or ready.get("event") != "ready":
    raise SystemExit("daemon did not report ready")
if ready.get("protocol_version") != 1:
    raise SystemExit("daemon ready response missing protocol_version=1")
capabilities = ready.get("capabilities", [])
if "chat" not in capabilities or "agent" not in capabilities:
    raise SystemExit("daemon ready response missing expected capabilities")
if not first.get("ok") or first.get("response") != "OK":
    raise SystemExit(f"first daemon response mismatch: {first}")
if first.get("runtime_reused"):
    raise SystemExit("first daemon response unexpectedly reported runtime reuse")
if not second.get("ok") or second.get("response") != "DONE":
    raise SystemExit(f"second daemon response mismatch: {second}")
if not second.get("runtime_reused"):
    raise SystemExit("second daemon response did not report runtime reuse")
if not shutdown.get("ok") or shutdown.get("event") != "shutdown":
    raise SystemExit(f"shutdown response mismatch: {shutdown}")

print("daemon_foreground=passed")
PY
