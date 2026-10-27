#!/usr/bin/env bash
# Shared execution plumbing; dose/search policies remain in the native smoke.
source "$(dirname "${BASH_SOURCE[0]}")/agent-model-smoke-common.sh"

agent_dose_run_profile() {
    local profile="$1" repo_root build_dir model suite repair_bin work_parent work_dir
    local mode mode_dir child_pid='' interrupted=0 condition_status overall_status=0 summary_status=0
    local -a modes args
    repo_root=$(agent_smoke_repo_root)
    build_dir=$(agent_smoke_build_path)
    model=$(agent_smoke_model)
    suite="${LLAMA_AGENT_DATASET_QUESTION_SUITE:-$repo_root/docs/examples/agent-flydelta-dataset-repair-echo-replay.jsonl}"
    [[ "$suite" == /* ]] || suite="$repo_root/$suite"
    repair_bin=$(agent_smoke_binary "$build_dir" llama-agent-flydelta-dataset-question-repair-model-smoke)
    agent_smoke_require_file "$model" "dose comparison model"
    agent_smoke_require_file "$suite" "dose comparison suite"
    agent_smoke_require_executable "$repair_bin" "dose comparison smoke"
    case "$profile" in
        paired) modes=(raw direction-dose) ;;
        three-condition) modes=(raw normalized direction-dose) ;;
        *) echo "Unknown dose comparison profile: $profile" >&2; return 2 ;;
    esac
    work_parent=$(agent_smoke_prepare_workdir flydelta-dose-comparison)
    work_dir=$(mktemp -d "$work_parent/dose-run.XXXXXX")
    export LLAMA_AGENT_RESIDENT_TRACE=1
    export LLAMA_AGENT_FORCE_TFO_LITE=0 LLAMA_AGENT_LOW_RANK_TFO_LITE=0 LLAMA_AGENT_LOW_RANK_RANK_SWEEP=0
    export LLAMA_AGENT_MODEL="$model" LLAMA_AGENT_DATASET_QUESTION_SUITE="$suite"
    local model_digest
    model_digest=$(sha256sum -- "$model")
    export LLAMA_AGENT_DOSE_MODEL_FINGERPRINT="sha256:${model_digest%% *}"
    args=(--model "$model" --suite "$suite" --dose-comparison --production-bindings
        --n-predict "${LLAMA_AGENT_N_PREDICT:-512}" --threads "${LLAMA_AGENT_THREADS:-3}"
        --n-gpu-layers "${LLAMA_AGENT_GPU_LAYERS:--1}"
        --n-parallel "${LLAMA_AGENT_N_PARALLEL:-2}"
        --n-ctx-per-sequence "${LLAMA_AGENT_N_CTX_PER_SEQUENCE:-6144}")
    if [[ "$profile" == paired ]]; then
        args+=(--behavior-key tool_use/dataset/data_aggregate --max-scenarios 2)
    else
        args+=(--max-scenarios "${LLAMA_AGENT_LOW_RANK_MAX_SCENARIOS:-8}")
    fi
    # Record the fully resolved invocation before the first expensive model call.
    python3 "$repo_root/scripts/agent-flydelta-dose-summary.py" --manifest "$work_dir" \
        "$profile" "$repo_root" "$repair_bin" "${args[@]}" || return $?
    trap 'interrupted=1; if [[ -n "$child_pid" ]]; then kill -TERM "$child_pid" 2>/dev/null || true; fi' INT TERM
    for mode in "${modes[@]}"; do
        [[ "$interrupted" == 0 ]] || break
        mode_dir="$work_dir/$mode"
        mkdir -p "$mode_dir"
        printf 'condition_started=%s mode=%s\n' "$(date -Is)" "$mode" > "$mode_dir/status.log"
        LLAMA_AGENT_DOSE_EXPERIMENT="$mode" LLAMA_AGENT_LEARNING_LEDGER="$mode_dir/learning-ledger.jsonl" \
            "$repair_bin" "${args[@]}" > "$mode_dir/low-rank-rank-sweep.log" 2>&1 &
        child_pid=$!
        printf '%s\n' "$child_pid" > "$mode_dir/pid"
        condition_status=0
        wait "$child_pid" || condition_status=$?
        if [[ "$interrupted" != 0 ]]; then
            wait "$child_pid" 2>/dev/null || true
            condition_status=130
        fi
        child_pid=''
        printf '%s\n' "$condition_status" > "$mode_dir/process.exit"
        printf 'condition_finished=%s mode=%s exit=%s\n' "$(date -Is)" "$mode" "$condition_status" >> "$mode_dir/status.log"
        if [[ "$condition_status" != 0 ]]; then overall_status=1; fi
        if [[ "$condition_status" == 130 || "$condition_status" == 143 ]]; then interrupted=1; break; fi
    done
    trap - INT TERM
    if [[ "$interrupted" != 0 ]]; then
        printf '%s\n' user_interrupted > "$work_dir/interrupted"
        overall_status=130
    fi
    python3 "$repo_root/scripts/agent-flydelta-dose-summary.py" "$work_dir" --modes "${modes[@]}" || summary_status=$?
    if [[ "$overall_status" == 0 && "$summary_status" != 0 ]]; then overall_status=$summary_status; fi
    printf '%s\n' "$overall_status" > "$work_dir/exit.status"
    printf 'dose_comparison_artifacts=%s\n' "$work_dir"
    return "$overall_status"
}
