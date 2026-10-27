#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/agent-model-smoke-common.sh"
repo_root=$(agent_smoke_repo_root)
build_dir=$(agent_smoke_build_path)
model=$(agent_smoke_model)
suite="${LLAMA_AGENT_DATASET_QUESTION_SUITE:-$repo_root/docs/examples/agent-flydelta-dataset-repair-echo-replay.jsonl}"
if [[ "$suite" != /* ]]; then suite="$repo_root/$suite"; fi
agent_smoke_require_file "$model" "dose comparison model"
agent_smoke_require_file "$suite" "dose comparison fixture suite"
repair_bin=$(agent_smoke_binary "$build_dir" llama-agent-flydelta-dataset-question-repair-model-smoke)
agent_smoke_require_executable "$repair_bin" "FlyDelta dataset question repair model smoke"

work_dir=$(agent_smoke_prepare_workdir flydelta-dose-paired)
export LLAMA_AGENT_RESIDENT_TRACE=1
export LLAMA_AGENT_LOW_RANK_TFO_LITE=0
export LLAMA_AGENT_FORCE_TFO_LITE=0
export LLAMA_AGENT_LOW_RANK_MAX_SCENARIOS=2
export LLAMA_AGENT_MODEL="$model"
export LLAMA_AGENT_DATASET_QUESTION_SUITE="$suite"
model_digest=$(sha256sum -- "$model")
export LLAMA_AGENT_DOSE_MODEL_FINGERPRINT="sha256:${model_digest%% *}"

printf '%s\n' \
    "model=$model" \
    "model_fingerprint=$LLAMA_AGENT_DOSE_MODEL_FINGERPRINT" \
    "suite=$suite" \
    "suite_fingerprint=$(sha256sum -- "$suite" | cut -d ' ' -f 1)" \
    "build_dir=$build_dir" \
    "gpu_layers=${LLAMA_AGENT_GPU_LAYERS:-99}" \
    "behavior_key=tool_use/dataset/data_aggregate" \
    "fixed_rank=1" \
    "trace=resident" \
    "timeout=none" > "$work_dir/comparison-inputs.txt"

overall_status=0
for mode in raw direction-dose; do
    mode_dir="$work_dir/$mode"
    mkdir -p "$mode_dir"
    printf 'condition_started=%s mode=%s\n' "$(date -Is)" "$mode" | tee "$mode_dir/status.log"
    set +e
    LLAMA_AGENT_DOSE_EXPERIMENT="$mode" \
    LLAMA_AGENT_LEARNING_LEDGER="$mode_dir/learning-ledger.jsonl" \
        "$repair_bin" \
        --model "$model" \
        --suite "$suite" \
        --behavior-key tool_use/dataset/data_aggregate \
        --max-scenarios 2 \
        --dose-comparison \
        --production-bindings \
        --n-predict "${LLAMA_AGENT_N_PREDICT:-512}" \
        --threads "${LLAMA_AGENT_THREADS:-3}" \
        --n-gpu-layers "${LLAMA_AGENT_GPU_LAYERS:-99}" \
        > "$mode_dir/trace.log" 2>&1
    condition_status=$?
    set -e
    printf '%s\n' "$condition_status" > "$mode_dir/process.exit"
    printf 'condition_finished=%s mode=%s exit=%s\n' "$(date -Is)" "$mode" "$condition_status" | tee -a "$mode_dir/status.log"
    if [[ "$condition_status" != 0 ]]; then overall_status=1; fi
done

printf '%s\n' "$overall_status" > "$work_dir/exit.status"
printf 'dose_paired_comparison_artifacts=%s\n' "$work_dir"
exit "$overall_status"
