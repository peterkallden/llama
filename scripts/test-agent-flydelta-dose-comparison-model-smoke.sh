#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/agent-model-smoke-common.sh"
repo_root=$(agent_smoke_repo_root)
model=$(agent_smoke_model)
agent_smoke_require_file "$model" "dose comparison model"
work_parent=$(agent_smoke_prepare_workdir flydelta-dose-comparison)
# Never mix a new attempt with stale logs from a previous interrupted run.
work_dir=$(mktemp -d "$work_parent/dose-run.XXXXXX")
suite="${LLAMA_AGENT_DATASET_QUESTION_SUITE:-$repo_root/docs/examples/agent-flydelta-dataset-repair-echo-replay.jsonl}"
if [[ "$suite" != /* ]]; then suite="$repo_root/$suite"; fi
agent_smoke_require_file "$suite" "dose comparison fixture suite"
export LLAMA_AGENT_DATASET_QUESTION_SUITE="$suite"
# Capture the actual model identity; do not reuse the historical smoke label.
model_digest=$(sha256sum -- "$model")
export LLAMA_AGENT_DOSE_MODEL_FINGERPRINT="sha256:${model_digest%% *}"
export LLAMA_AGENT_RESIDENT_TRACE=1
export LLAMA_AGENT_LOW_RANK_TFO_LITE=0
export LLAMA_AGENT_FORCE_TFO_LITE=0
export LLAMA_AGENT_LOW_RANK_MAX_SCENARIOS="${LLAMA_AGENT_LOW_RANK_MAX_SCENARIOS:-8}"
export LLAMA_AGENT_MODEL="$model"
printf '%s\n' "model=$model" "model_fingerprint=$LLAMA_AGENT_DOSE_MODEL_FINGERPRINT" \
    "build_dir=$(agent_smoke_build_dir)" "gpu_layers=${LLAMA_AGENT_GPU_LAYERS:-99}" \
    "suite=${LLAMA_AGENT_DATASET_QUESTION_SUITE:-docs/examples/agent-flydelta-dataset-repair-echo-replay.jsonl}" \
    > "$work_dir/comparison-inputs.txt"
suite_digest=$(sha256sum -- "$suite")
printf 'suite_fingerprint=sha256:%s\n' "${suite_digest%% *}" >> "$work_dir/comparison-inputs.txt"
for mode in raw normalized direction-dose; do
    mkdir -p "$work_dir/$mode"
    condition_exit=0
    LLAMA_AGENT_DOSE_EXPERIMENT="$mode" \
    LLAMA_AGENT_SMOKE_WORKDIR="$work_dir/$mode" \
        bash "$repo_root/scripts/test-agent-flydelta-low-rank-model-smoke.sh" || condition_exit=$?
    printf '%s\n' "$condition_exit" > "$work_dir/$mode/process.exit"
    # An interrupted comparison must not start another expensive model run.
    if [[ "$condition_exit" == 130 || "$condition_exit" == 143 ]]; then break; fi
done
echo "dose_comparison_artifacts=$work_dir"
python3 "$repo_root/scripts/agent-flydelta-dose-summary.py" "$work_dir"
