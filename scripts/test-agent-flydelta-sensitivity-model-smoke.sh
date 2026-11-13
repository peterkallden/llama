#!/usr/bin/env bash
# Matched TFO control/treatment runs. Always retain model and component traces.
set -euo pipefail
source "$(dirname "$0")/agent-model-smoke-common.sh"
repo_root=$(agent_smoke_repo_root)
build_dir=$(agent_smoke_build_dir)
model="${LLAMA_AGENT_MODEL:-}"
if [[ $# -eq 2 && "$1" == --model ]]; then model="$2";
elif [[ $# -ne 0 ]]; then echo "usage: $0 [--model PATH]" >&2; exit 2; fi
if [[ -z "$model" ]]; then echo 'sensitivity_model=skipped_no_model'; exit 77; fi
agent_smoke_require_file "$model" model
binary=$(agent_smoke_binary "$build_dir" llama-agent-flydelta-dataset-question-repair-model-smoke)
agent_smoke_require_executable "$binary" 'FlyDelta model smoke'
work_dir=$(agent_smoke_prepare_workdir flydelta-sensitivity-model)
export LLAMA_AGENT_RESIDENT_TRACE=1
export LLAMA_AGENT_TRACE=1
unset LLAMA_AGENT_DOSE_EXPERIMENT LLAMA_AGENT_LOW_RANK_RANK_SWEEP
suite="${LLAMA_AGENT_DATASET_QUESTION_SUITE:-$repo_root/docs/examples/agent-flydelta-dataset-repair-echo-replay.jsonl}"
sha256sum -- "$model" "$suite" > "$work_dir/inputs.sha256"
status=0
for condition in control enabled; do
    "$binary" --model "$model" --suite "$suite" --production-bindings \
        "--sensitivity-$condition" --max-scenarios "${LLAMA_AGENT_LOW_RANK_MAX_SCENARIOS:-4}" \
        --baseline-repeats "${LLAMA_AGENT_BASELINE_REPEATS:-3}" \
        --n-predict "${LLAMA_AGENT_N_PREDICT:-512}" --threads "${LLAMA_AGENT_THREADS:-3}" \
        --n-gpu-layers "${LLAMA_AGENT_GPU_LAYERS:--1}" \
        --n-parallel "${LLAMA_AGENT_N_PARALLEL:-2}" \
        --n-ctx-per-sequence "${LLAMA_AGENT_N_CTX_PER_SEQUENCE:-6144}" \
        --learning-ledger "$work_dir/$condition-ledger.jsonl" \
        > "$work_dir/$condition.log" 2>&1 || status=1
done
python3 "$repo_root/scripts/agent-flydelta-sensitivity-summary.py" "$work_dir" || status=$?
echo "sensitivity_artifacts=$work_dir"
exit "$status"
