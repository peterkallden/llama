#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/agent-model-smoke-common.sh"

repo_root=$(agent_smoke_repo_root)
build_dir=$(agent_smoke_build_dir)
model_override=
suite_override=
guidance_mode=
scenario_override=
n_parallel="${LLAMA_AGENT_N_PARALLEL:-2}"
n_ctx_per_sequence="${LLAMA_AGENT_N_CTX_PER_SEQUENCE:-6144}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            [[ $# -ge 2 ]] || { echo "--model requires a path" >&2; exit 2; }
            model_override="$2"
            shift 2
            ;;
        --suite)
            [[ $# -ge 2 ]] || { echo "--suite requires a path" >&2; exit 2; }
            suite_override="$2"
            shift 2
            ;;
        --guidance-mode)
            [[ $# -ge 2 ]] || { echo "--guidance-mode requires off|shadow|experimental|all" >&2; exit 2; }
            guidance_mode="$2"
            shift 2
            ;;
        --scenario-id)
            [[ $# -ge 2 ]] || { echo "--scenario-id requires an id" >&2; exit 2; }
            scenario_override="$2"
            shift 2
            ;;
        --n-parallel)
            [[ $# -ge 2 ]] || { echo "--n-parallel requires a positive integer" >&2; exit 2; }
            n_parallel="$2"
            shift 2
            ;;
        --n-ctx-per-sequence)
            [[ $# -ge 2 ]] || { echo "--n-ctx-per-sequence requires a positive integer" >&2; exit 2; }
            n_ctx_per_sequence="$2"
            shift 2
            ;;
        --help|-h)
            echo "usage: $0 [--model PATH] [--suite PATH] [--guidance-mode off|shadow|experimental|all] [--scenario-id ID] [--n-parallel N] [--n-ctx-per-sequence N]"
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            echo "usage: $0 [--model PATH] [--suite PATH] [--guidance-mode off|shadow|experimental|all] [--scenario-id ID] [--n-parallel N] [--n-ctx-per-sequence N]" >&2
            exit 2
            ;;
    esac
done

model="${model_override:-$(agent_smoke_model)}"
suite="${suite_override:-${LLAMA_AGENT_DATASET_QUESTION_SUITE:-${repo_root}/docs/examples/agent-flydelta-dataset-question-suite.json}}"
if [[ "$model" != /* ]]; then model="${repo_root}/${model}"; fi
if [[ "$suite" != /* ]]; then suite="${repo_root}/${suite}"; fi
work_dir=$(agent_smoke_prepare_workdir flydelta-dataset-question-repair-model)

smoke_bin=$(agent_smoke_binary "$build_dir" llama-agent-flydelta-dataset-question-repair-model-smoke)
agent_smoke_build_if_requested llama-agent-flydelta-dataset-question-repair-model-smoke

agent_smoke_require_file "$model" "chat model"
agent_smoke_require_file "$suite" "FlyDelta dataset question suite"
agent_smoke_require_executable "$smoke_bin" "FlyDelta dataset question repair model smoke"

args=(
    --model "$model"
    --suite "$suite"
    --threads "${LLAMA_AGENT_THREADS:-3}"
    --n-predict "${LLAMA_AGENT_N_PREDICT:-128}"
    --n-gpu-layers "${LLAMA_AGENT_GPU_LAYERS:-0}"
    --n-parallel "$n_parallel"
    --n-ctx-per-sequence "$n_ctx_per_sequence"
)
if [[ -n "${LLAMA_AGENT_CONFIG:-}" ]]; then
    args+=(--config "$LLAMA_AGENT_CONFIG")
fi

log_path="$work_dir/dataset-question-repair-model.log"
if [[ -n "$scenario_override" ]]; then args+=(--scenario-id "$scenario_override"); fi

verify_guidance_mode_log() {
    local mode="$1"
    local path="$2"
    grep -Fq "flydelta_guidance_smoke requested_mode=${mode} provider_mode=${mode}" "$path"
    grep -Fq "flydelta_guidance_smoke_whirlpool_trace mode=${mode}" "$path"
    if [[ "$mode" == "off" ]]; then
        if grep -Fq '[flydelta] outcome_guidance ' "$path"; then
            echo "off mode unexpectedly invoked the guidance adviser: ${path}" >&2
            return 1
        fi
    else
        grep -Fq "[flydelta] outcome_guidance " "$path"
        grep -Fq "\"mode\":\"${mode}\"" "$path"
        grep -Fq '"stage":"whirlpool_probe"' "$path"
    fi
}

case "$guidance_mode" in
    "")
        agent_smoke_run_logged "$log_path" "$smoke_bin" "${args[@]}"
        grep -Fq 'flydelta_dataset_question_repair_model_smoke' "$log_path"
        echo "flydelta_dataset_question_repair_model_smoke=passed"
        echo "log=${log_path}"
        ;;
    off|shadow|experimental)
        args+=(--outcome-guidance-mode "$guidance_mode")
        agent_smoke_run_logged "$log_path" "$smoke_bin" "${args[@]}"
        grep -Fq 'flydelta_dataset_question_repair_model_smoke' "$log_path"
        verify_guidance_mode_log "$guidance_mode" "$log_path"
        echo "flydelta_guidance_model_smoke mode=${guidance_mode} passed"
        echo "log=${log_path}"
        ;;
    all)
        for mode in off shadow experimental; do
            mode_args=("${args[@]}")
            mode_args+=(--outcome-guidance-mode "$mode")
            mode_log="$work_dir/dataset-question-repair-model-${mode}.log"
            echo "flydelta_guidance_model_smoke_start mode=${mode}"
            agent_smoke_run_logged "$mode_log" "$smoke_bin" "${mode_args[@]}"
            grep -Fq 'flydelta_dataset_question_repair_model_smoke' "$mode_log"
            verify_guidance_mode_log "$mode" "$mode_log"
            echo "flydelta_guidance_model_smoke mode=${mode} passed"
            echo "log=${mode_log}"
        done
        ;;
    *)
        echo "invalid --guidance-mode: ${guidance_mode}" >&2
        exit 2
        ;;
esac
