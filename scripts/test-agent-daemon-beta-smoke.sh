#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${LLAMA_AGENT_BUILD_DIR:-build-agent}"
CONFIGURATION="${LLAMA_AGENT_CONFIGURATION:-RelWithDebInfo}"
CHAT_MODEL="${LLAMA_AGENT_CHAT_MODEL:-}"
SANDBOX_EXECUTABLE="${LLAMA_AGENT_SANDBOX_EXECUTABLE:-}"
TIMEOUT_SECONDS=120
INCLUDE_CTEST=0
KEEP_LOGS=0

usage() {
    echo "usage: $0 [--build-dir DIR] [--configuration CONFIG] [--include-ctest] [--keep-logs]"
    echo "       defaults: LLAMA_AGENT_BUILD_DIR=build-agent, LLAMA_AGENT_CONFIGURATION=RelWithDebInfo"
    echo "       optional model-backed foreground smoke: LLAMA_AGENT_CHAT_MODEL=PATH"
    echo "       optional container smoke executable: LLAMA_AGENT_SANDBOX_EXECUTABLE=docker|podman"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --configuration) CONFIGURATION="$2"; shift 2 ;;
        --include-ctest) INCLUDE_CTEST=1; shift ;;
        --keep-logs) KEEP_LOGS=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -n "$SANDBOX_EXECUTABLE" ]]; then
    command -v "$SANDBOX_EXECUTABLE" >/dev/null 2>&1 || {
        echo "Configured sandbox executable not found: $SANDBOX_EXECUTABLE" >&2
        exit 1
    }
    export LLAMA_AGENT_SANDBOX_EXECUTABLE="$SANDBOX_EXECUTABLE"
fi
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
        echo "Required executable not found in either:" >&2
        echo "  $single_config" >&2
        echo "  $multi_config" >&2
        return 1
    fi
}
LOG_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/llama-agent-beta-test.XXXXXX")"
declare -a RESULTS=()
cleanup() {
    if [[ "$KEEP_LOGS" -eq 1 ]]; then
        echo "agent_test_pack_logs=$LOG_ROOT"
    else
        rm -rf "$LOG_ROOT"
    fi
}
trap cleanup EXIT

run_suite() {
    local name="$1"; shift
    local out="$LOG_ROOT/$name.stdout.log"
    local err="$LOG_ROOT/$name.stderr.log"
    local start=$SECONDS
    if timeout --preserve-status "${TIMEOUT_SECONDS}s" "$@" >"$out" 2>"$err"; then
        RESULTS+=("$name:passed")
        echo "suite=$name status=passed duration_ms=$(( (SECONDS-start) * 1000 ))"
    else
        local code=$?
        RESULTS+=("$name:failed")
        echo "suite=$name status=failed exit_code=$code log=$err" >&2
    fi
}

smokes=(
    llama-agent-runtime-operation-manager-smoke
    llama-agent-runtime-session-manager-smoke
    llama-agent-daemon-dispatcher-smoke
    llama-agent-daemon-protocol-smoke
    llama-agent-daemon-jsonl-protocol-smoke
    llama-agent-daemon-mcp-config-smoke
    llama-agent-deliberation-policy-smoke
    llama-agent-deliberate-runtime-smoke
    llama-agent-research-runtime-smoke
    test-agent-research-contract
    llama-agent-resource-store-smoke
    llama-agent-mcp-tool-provider-smoke
    llama-agent-mcp-stdio-client-smoke
    llama-agent-mcp-http-client-smoke
    llama-agent-mcp-http-inbound-dispatcher-smoke
    llama-agent-mcp-http-vertical-smoke
    llama-agent-mcp-agent-tools-smoke
    llama-agent-daemon-wait-events-smoke
)

for target in "${smokes[@]}"; do
    executable="$(resolve_executable "$target")" || exit 1
    run_suite "$target" "$executable"
done

if [[ "$INCLUDE_CTEST" -eq 1 ]]; then
    ctest_bin="$(command -v ctest)"
    run_suite ctest-agent "$ctest_bin" --test-dir "$BUILD_ROOT" -C "$CONFIGURATION" -L agent --output-on-failure
fi

if [[ -n "$CHAT_MODEL" ]]; then
    daemon_smoke="$REPO_ROOT/scripts/test-agent-daemon-smoke.sh"
    run_suite daemon-foreground "$daemon_smoke" \
        --build-dir "$BUILD_DIR" \
        --configuration "$CONFIGURATION" \
        --chat-model "$CHAT_MODEL"
else
    echo "suite=daemon-foreground status=skipped reason=LLAMA_AGENT_CHAT_MODEL_not_set"
fi

failed=0
for result in "${RESULTS[@]}"; do [[ "$result" == *:failed ]] && failed=$((failed + 1)); done
echo "agent_test_pack_suites=${#RESULTS[@]}"
echo "agent_test_pack_failed=$failed"
if [[ "$failed" -ne 0 ]]; then echo "agent_test_pack=failed"; exit 1; fi
echo "agent_test_pack=passed"
