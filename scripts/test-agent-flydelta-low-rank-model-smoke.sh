#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/agent-model-smoke-common.sh"

repo_root=$(agent_smoke_repo_root)
build_dir=$(agent_smoke_build_dir)
model="${LLAMA_AGENT_MODEL:-$(agent_smoke_model)}"
suite="${LLAMA_AGENT_DATASET_QUESTION_SUITE:-${repo_root}/docs/examples/agent-flydelta-dataset-repair-echo-replay.jsonl}"
max_scenarios="${LLAMA_AGENT_LOW_RANK_MAX_SCENARIOS:-4}"
if [[ "$model" != /* ]]; then model="${repo_root}/${model}"; fi
if [[ "$suite" != /* ]]; then suite="${repo_root}/${suite}"; fi
work_dir=$(agent_smoke_prepare_workdir flydelta-low-rank-model)

repair_bin=$(agent_smoke_binary "$build_dir" llama-agent-flydelta-dataset-question-repair-model-smoke)
cvec_bin=$(agent_smoke_binary "$build_dir" llama-agent-flydelta-cvec-batch-model-smoke)
agent_smoke_require_file "$model" "Qwen Instruct model"
agent_smoke_require_file "$suite" "FlyDelta dataset repair replay"
agent_smoke_require_executable "$repair_bin" "FlyDelta dataset question repair model smoke"
agent_smoke_require_executable "$cvec_bin" "FlyDelta native cvec batch smoke"

agent_smoke_build_if_requested

rank_log="$work_dir/low-rank-rank-sweep.log"
rank_exit=0
if LLAMA_AGENT_RESIDENT_TRACE="${LLAMA_AGENT_RESIDENT_TRACE:-1}" \
        LLAMA_AGENT_TIMEOUT_SECONDS="${LLAMA_AGENT_TIMEOUT_SECONDS:-1800}" \
        agent_smoke_run_logged "$rank_log" "$repair_bin" \
            --model "$model" --suite "$suite" \
            --max-scenarios "$max_scenarios" \
            --low-rank-rank-sweep --n-predict "${LLAMA_AGENT_N_PREDICT:-32}" \
            --threads "${LLAMA_AGENT_THREADS:-3}" \
            --n-gpu-layers "${LLAMA_AGENT_GPU_LAYERS:-99}"; then
    rank_exit=0
else
    rank_exit=$?
fi
if [[ "$rank_exit" -eq 0 ]] && grep -Fq 'flydelta_low_rank_rank_sweep group=' "$rank_log"; then
    grep -Fq 'requested_rank=1' "$rank_log"
    grep -Fq 'requested_rank=2' "$rank_log"
    grep -Fq 'requested_rank=4' "$rank_log"
    grep -Fq 'execution_batch=native' "$rank_log"
    rank_status=completed
elif [[ "$rank_exit" -eq 0 ]] && grep -Fq 'flydelta_low_rank_rank_sweep_skipped' "$rank_log"; then
    echo "rank_comparison=unavailable reason=insufficient_compatible_directions"
    rank_status=unavailable
elif [[ "$rank_exit" -eq 124 ]] && grep -Fq 'flydelta_low_rank_rank_sweep group=' "$rank_log"; then
    completed_ranks=$(grep -c '^flydelta_low_rank_rank_sweep group=' "$rank_log")
    echo "rank_comparison=partial_timeout completed_ranks=${completed_ranks}"
    rank_status=partial_timeout
else
    echo "rank_comparison=failed exit=${rank_exit}; log=${rank_log}" >&2
    rank_status=failed
fi

# The rank sweep keeps its existing Deep runner intentionally visible in the
# report.  This separate resident check proves that the same model/backend
# supports the physical per-sequence cvec batch path before any future
# basis-plus-coefficient runtime representation is proposed.
cvec_log="$work_dir/native-cvec-batch.log"
LLAMA_AGENT_TIMEOUT_SECONDS="${LLAMA_AGENT_TIMEOUT_SECONDS:-1800}" \
agent_smoke_run_logged "$cvec_log" "$cvec_bin" \
    --model "$model" --n-predict "${LLAMA_AGENT_CVEC_N_PREDICT:-8}" \
    --threads "${LLAMA_AGENT_THREADS:-3}" \
    --n-gpu-layers "${LLAMA_AGENT_GPU_LAYERS:-99}" --factorized
grep -Fq 'flydelta_cvec_batch_model_smoke=passed' "$cvec_log"
grep -Fq 'batched_execution=yes' "$cvec_log"
grep -Fq 'overlay_representation=factorized' "$cvec_log"

echo "flydelta_low_rank_model_smoke=completed"
echo "rank_comparison=${rank_status}"
echo "rank_log=${rank_log}"
echo "cvec_log=${cvec_log}"
if [[ "$rank_exit" -ne 0 ]]; then exit "$rank_exit"; fi
