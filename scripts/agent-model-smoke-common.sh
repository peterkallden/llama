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
agent_smoke_bootstrap_args() {
    local package="${LLAMA_AGENT_WORKFLOW_PACKAGE:-$(agent_smoke_repo_root)/docs/examples/agent-bootstrap-workflows-v1.json}"
    agent_smoke_require_file "$package" "workflow starter package"
    printf '%s\n' --agent-import "$package"
}
agent_smoke_run_logged() {
    local log_path="$1"; shift
    local command=("$@") has_import=0 arg index
    # Full CLI agent turns should exercise the installed starter workflows.
    # Lower-level model binaries do not accept CLI orchestration flags.
    if [[ "${command[0]##*/}" == llama-agent && "${command[1]:-}" == run ]]; then
        for ((index=0; index<${#command[@]}; ++index)); do
            arg=${command[index]}
            if [[ "$arg" == --agent-import ]]; then
                [[ $((index + 1)) -lt ${#command[@]} ]] || { echo "model smoke bootstrap: --agent-import requires a path" >&2; return 1; }
                agent_smoke_require_file "${command[index+1]}" "workflow starter package"
                has_import=1; break
            elif [[ "$arg" == --agent-import=* ]]; then
                agent_smoke_require_file "${arg#--agent-import=}" "workflow starter package"
                has_import=1; break
            fi
        done
        if [[ $has_import == 0 ]]; then
            local bootstrap_args=()
            mapfile -t bootstrap_args < <(agent_smoke_bootstrap_args)
            [[ ${#bootstrap_args[@]} == 2 ]] || { echo "model smoke bootstrap unavailable" >&2; return 1; }
            command+=("${bootstrap_args[@]}")
        fi
    fi
    set -- "${command[@]}"
    echo "running: $*"
    local status
    set +e
    if [[ -n "${LLAMA_AGENT_TIMEOUT_SECONDS:-}" ]]; then
        timeout --foreground "$LLAMA_AGENT_TIMEOUT_SECONDS" "$@" >"$log_path" 2>&1
        status=$?
    else
        "$@" >"$log_path" 2>&1
        status=$?
    fi
    set -e
    cat "$log_path"
    if [[ $status -eq 124 && -n "${LLAMA_AGENT_TIMEOUT_SECONDS:-}" ]]; then
        echo "model_smoke_budget phase=process reason=wall_clock_timeout limit_seconds=${LLAMA_AGENT_TIMEOUT_SECONDS} evaluation_status=incomplete log=${log_path}" >&2
    fi
    if [[ $status -ne 0 ]]; then echo "model smoke failed: exit=${status}; log=${log_path}" >&2; return "$status"; fi
}
agent_smoke_build_if_requested() {
    if [[ "${LLAMA_AGENT_BUILD:-0}" == 1 ]]; then
        local targets=("$@")
        if [[ ${#targets[@]} -eq 0 ]]; then targets=(llama-agent); fi
        cmake --build "$(agent_smoke_build_path)" --target "${targets[@]}" --parallel "${LLAMA_AGENT_BUILD_JOBS:-1}"
    fi
}
