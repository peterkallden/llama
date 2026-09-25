#!/usr/bin/env bash
set -euo pipefail

agent_smoke_repo_root() { cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd; }
agent_smoke_build_dir() { printf '%s\n' "${LLAMA_AGENT_BUILD_DIR:-build-agent-packaging}"; }
agent_smoke_model() {
    local model="${LLAMA_AGENT_MODEL:-models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf}"
    if [[ "$model" = /* ]]; then printf '%s\n' "$model"; else printf '%s/%s\n' "$(agent_smoke_repo_root)" "$model"; fi
}
agent_smoke_build_path() {
    local build_dir
    build_dir=$(agent_smoke_build_dir)
    if [[ "$build_dir" = /* ]]; then printf '%s\n' "$build_dir"; else printf '%s/%s\n' "$(agent_smoke_repo_root)" "$build_dir"; fi
}
agent_smoke_binary() {
    local build_dir="$1"
    local binary_name="$2"
    if [[ "$build_dir" = /* ]]; then
        printf '%s/bin/%s\n' "$build_dir" "$binary_name"
    else
        printf '%s/%s/bin/%s\n' "$(agent_smoke_repo_root)" "$build_dir" "$binary_name"
    fi
}
agent_smoke_require_file() {
    if [[ ! -f "$1" ]]; then echo "model smoke not-run: $2 not found: $1" >&2; exit 77; fi
}
agent_smoke_require_executable() {
    if [[ ! -x "$1" ]]; then echo "model smoke not-run: $2 not found: $1" >&2; exit 77; fi
}
agent_smoke_prepare_workdir() {
    if [[ -n "${LLAMA_AGENT_SMOKE_WORKDIR:-}" ]]; then mkdir -p "$LLAMA_AGENT_SMOKE_WORKDIR"; printf '%s\n' "$LLAMA_AGENT_SMOKE_WORKDIR"; else mktemp -d "${TMPDIR:-/tmp}/llama-agent-$1.XXXXXX"; fi
}
agent_smoke_run_logged() {
    local log_path="$1"; shift
    echo "running: $*"
    set +e; timeout --foreground "${LLAMA_AGENT_TIMEOUT_SECONDS:-180}" "$@" >"$log_path" 2>&1; local status=$?; set -e
    cat "$log_path"
    if [[ $status -ne 0 ]]; then echo "model smoke failed: exit=${status}; log=${log_path}" >&2; return "$status"; fi
}
agent_smoke_build_if_requested() {
    if [[ "${LLAMA_AGENT_BUILD:-0}" == 1 ]]; then cmake --build "$(agent_smoke_build_path)" --target llama-agent --parallel "${LLAMA_AGENT_BUILD_JOBS:-1}"; fi
}
