#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/agent-model-smoke-common.sh"

repo_root=$(agent_smoke_repo_root)
build_dir=$(agent_smoke_build_dir)
model="${LLAMA_AGENT_MODEL:-$(agent_smoke_model)}"
suite="${LLAMA_AGENT_DATASET_QUESTION_SUITE:-${repo_root}/docs/examples/agent-flydelta-dataset-repair-echo-replay.jsonl}"
max_scenarios="${LLAMA_AGENT_LOW_RANK_MAX_SCENARIOS:-4}"
n_parallel="${LLAMA_AGENT_N_PARALLEL:-2}"
n_ctx_per_sequence="${LLAMA_AGENT_N_CTX_PER_SEQUENCE:-6144}"
cvec_arm_count="${LLAMA_AGENT_CVEC_ARM_COUNT:-4}"
cvec_warmup_count="${LLAMA_AGENT_CVEC_WARMUP_COUNT:-1}"
cvec_repeat_count="${LLAMA_AGENT_CVEC_REPEAT_COUNT:-4}"
if [[ "$model" != /* ]]; then model="${repo_root}/${model}"; fi
if [[ "$suite" != /* ]]; then suite="${repo_root}/${suite}"; fi
work_dir=$(agent_smoke_prepare_workdir flydelta-low-rank-model)

repair_bin=$(agent_smoke_binary "$build_dir" llama-agent-flydelta-dataset-question-repair-model-smoke)
cvec_bin=$(agent_smoke_binary "$build_dir" llama-agent-flydelta-cvec-batch-model-smoke)
agent_smoke_build_if_requested llama-agent-flydelta-dataset-question-repair-model-smoke llama-agent-flydelta-cvec-batch-model-smoke
agent_smoke_require_file "$model" "Qwen Instruct model"
agent_smoke_require_file "$suite" "FlyDelta dataset repair replay"
agent_smoke_require_executable "$repair_bin" "FlyDelta dataset question repair model smoke"
agent_smoke_require_executable "$cvec_bin" "FlyDelta native cvec batch smoke"

rank_log="$work_dir/low-rank-rank-sweep.log"
rank_exit=0
rank_args=()
if [[ "${LLAMA_AGENT_LOW_RANK_PRODUCTION_BINDINGS:-1}" == "1" ]]; then
    rank_args+=(--production-bindings)
fi
if [[ "${LLAMA_AGENT_LOW_RANK_PRODUCTION_WHERE:-0}" == "1" ]]; then
    rank_args+=(--production-where)
fi
factorized_disabled="${LLAMA_AGENT_LOW_RANK_DISABLE_FACTORIZED:-}"
if [[ -z "$factorized_disabled" ]]; then
    # Backward-compatible interpretation of the previous positive toggle.
    if [[ "${LLAMA_AGENT_LOW_RANK_FACTORIZED:-1}" == "0" ]]; then
        factorized_disabled=1
    else
        factorized_disabled=0
    fi
fi
if [[ "$factorized_disabled" == "1" ]]; then
    rank_args+=(--disable-factorized-cvec-batch)
fi
if [[ "${LLAMA_AGENT_LOW_RANK_TFO_LITE:-0}" == "1" ]]; then
    rank_args+=(--force-tfo-lite)
fi
if [[ -n "${LLAMA_AGENT_LOW_RANK_BEHAVIOR_KEY:-}" ]]; then
    rank_args+=(--behavior-key "$LLAMA_AGENT_LOW_RANK_BEHAVIOR_KEY")
fi
if LLAMA_AGENT_RESIDENT_TRACE="${LLAMA_AGENT_RESIDENT_TRACE:-1}" \
        LLAMA_AGENT_LEARNING_LEDGER="$work_dir/learning-ledger.jsonl" \
        LLAMA_AGENT_TIMEOUT_SECONDS="${LLAMA_AGENT_TIMEOUT_SECONDS:-1800}" \
        LLAMA_AGENT_N_PARALLEL="$n_parallel" \
        LLAMA_AGENT_N_CTX_PER_SEQUENCE="$n_ctx_per_sequence" \
        agent_smoke_run_logged "$rank_log" "$repair_bin" \
            --model "$model" --suite "$suite" \
            --max-scenarios "$max_scenarios" \
            --low-rank-rank-sweep "${rank_args[@]}" --n-predict "${LLAMA_AGENT_N_PREDICT:-512}" \
            --threads "${LLAMA_AGENT_THREADS:-3}" \
            --n-gpu-layers "${LLAMA_AGENT_GPU_LAYERS:-99}"; then
    rank_exit=0
else
    rank_exit=$?
fi
if [[ "$rank_exit" -eq 0 ]] && grep -Fq 'flydelta_low_rank_rank_sweep group=' "$rank_log"; then
    grep -Fq 'requested_rank=1' "$rank_log"
    for rank in 2 4; do
        if ! grep -Fq " requested_rank=${rank} " "$rank_log" &&
                ! grep -Eq "flydelta_low_rank_rank_unavailable group=.* requested_rank=${rank} " "$rank_log"; then
            echo "rank comparison omitted rank=${rank} without an unavailable record" >&2
            exit 1
        fi
    done
    grep -Eq 'execution_batch=(device_batch|backend_batch)' "$rank_log"
    rank_status=completed
elif [[ "$rank_exit" -eq 0 ]] && grep -Fq 'flydelta_low_rank_rank_sweep_skipped' "$rank_log"; then
    grep -F 'flydelta_low_rank_rank_sweep_skipped' "$rank_log"
    echo "rank_comparison=unavailable reason=see_rank_skip_record"
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
    --n-gpu-layers "${LLAMA_AGENT_GPU_LAYERS:-99}" \
    --n-parallel "$n_parallel" --n-ctx-per-sequence "$n_ctx_per_sequence" \
    --arm-count "$cvec_arm_count" --warmup-count "$cvec_warmup_count" \
    --repeat-count "$cvec_repeat_count" --factorized
grep -Fq 'flydelta_cvec_batch_model_smoke=passed' "$cvec_log"
grep -Fq 'batched_execution=yes' "$cvec_log"
grep -Fq 'overlay_representation=factorized' "$cvec_log"
grep -Fq 'factorized_binding_used=yes' "$cvec_log"

# Run the identical deterministic prompt/fixture through the legacy sparse
# path, then compare the overlaid sequence output against the factorized run.
sparse_log="$work_dir/sparse-cvec-batch.log"
LLAMA_AGENT_TIMEOUT_SECONDS="${LLAMA_AGENT_TIMEOUT_SECONDS:-1800}" \
agent_smoke_run_logged "$sparse_log" "$cvec_bin" \
    --model "$model" --n-predict "${LLAMA_AGENT_CVEC_N_PREDICT:-8}" \
    --threads "${LLAMA_AGENT_THREADS:-3}" \
    --n-gpu-layers "${LLAMA_AGENT_GPU_LAYERS:-99}" \
    --n-parallel "$n_parallel" --n-ctx-per-sequence "$n_ctx_per_sequence" \
    --arm-count "$cvec_arm_count" --warmup-count "$cvec_warmup_count" \
    --repeat-count "$cvec_repeat_count"
grep -Fq 'flydelta_cvec_batch_model_smoke=passed' "$sparse_log"
factorized_output=$(grep '^cvec_smoke_arm_output arm="flydelta://arm/cvec-batch-smoke/1"' "$cvec_log" | tail -n 1)
sparse_output=$(grep '^cvec_smoke_arm_output arm="flydelta://arm/cvec-batch-smoke/1"' "$sparse_log" | tail -n 1)
if [[ -z "$factorized_output" || "$factorized_output" != "$sparse_output" ]]; then
    echo "Q1 output mismatch between sparse and factorized cvec runs" >&2
    echo "sparse: $sparse_output" >&2
    echo "factorized: $factorized_output" >&2
    exit 1
fi
factorized_margin=$(grep '^cvec_smoke_teacher_margin arm="flydelta://arm/cvec-batch-smoke/1"' "$cvec_log" | tail -n 1 | sed -E 's/.* total=([^ ]+).*/\1/')
sparse_margin=$(grep '^cvec_smoke_teacher_margin arm="flydelta://arm/cvec-batch-smoke/1"' "$sparse_log" | tail -n 1 | sed -E 's/.* total=([^ ]+).*/\1/')
if [[ -z "$factorized_margin" || -z "$sparse_margin" ]]; then
    echo "Q1 teacher-forced margin missing" >&2
    exit 1
fi
margin_delta=$(awk -v factorized="$factorized_margin" -v sparse="$sparse_margin" \
    'BEGIN { delta = factorized - sparse; if (delta < 0) delta = -delta; printf "%.9g", delta }')
if ! awk -v delta="$margin_delta" 'BEGIN { exit !(delta <= 1e-4) }'; then
    echo "Q1 teacher margin mismatch: sparse=${sparse_margin} factorized=${factorized_margin} delta=${margin_delta}" >&2
    exit 1
fi
echo "q1_generated_output_match=yes"
echo "q1_teacher_margin_delta=${margin_delta}"
if grep -Fq 'device_resident=yes' "$cvec_log"; then
    echo "q1_execution_device=device_batch"
else
    echo "q1_execution_device=backend_batch_non_device"
fi

echo "q1_repeats=${cvec_repeat_count}"
echo "flydelta_low_rank_model_smoke=completed"
echo "rank_comparison=${rank_status}"
echo "rank_log=${rank_log}"
echo "cvec_log=${cvec_log}"
echo "sparse_log=${sparse_log}"
if [[ "$rank_exit" -ne 0 ]]; then exit "$rank_exit"; fi
