#!/usr/bin/env bash
# FlyDelta progress/advice smoke. Does not build or start a daemon.
set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
bin_dir=${LLAMA_AGENT_BUILD_DIR:-"$repo_dir/build-agent-vulkan-cozo"}/bin
trace_dir=${FLYDELTA_SMOKE_TRACE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/flydelta-progress-smoke.XXXXXXXX")}
mkdir -p -- "$trace_dir" || exit 1
trace=$trace_dir/trace.jsonl
: > "$trace" || exit 1
status=0

model=${LLAMA_AGENT_MODEL:-}
suite=${LLAMA_AGENT_DATASET_QUESTION_SUITE:-"$repo_dir/docs/examples/agent-flydelta-dataset-question-suite-incremental-v3.json"}
mode=experimental
while (($#)); do
    case "$1" in
        --model)
            (($# >= 2)) || { echo "missing value for --model" >&2; exit 2; }
            model=$2; shift 2 ;;
        --suite)
            (($# >= 2)) || { echo "missing value for --suite" >&2; exit 2; }
            suite=$2; shift 2 ;;
        --mode)
            (($# >= 2)) || { echo "missing value for --mode" >&2; exit 2; }
            mode=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$model" ]]; then
    printf '%s\n' '{"event":"start","type":"model_free","model_inference":false,"gpu":"not_used","note":"Separate component checks; no daemon restart or end-to-end queue-to-adviser execution"}' >> "$trace"
else
    printf '%s\n' '{"event":"start","type":"model_backed","model_inference":true,"gpu":"auto-detected","note":"Deterministic component checks plus a traced host-verified model-backed FlyDelta wave"}' >> "$trace"
fi

for name in test-agent-flydelta-deep-search test-agent-flydelta-collection test-agent-flydelta-outcome-guidance; do
    printf '{"event":"check_start","name":"%s"}\n' "$name" >> "$trace"
    if [[ ! -x "$bin_dir/$name" ]]; then
        printf '{"event":"check_end","name":"%s","status":"missing_binary"}\n' "$name" >> "$trace"
        status=1
        continue
    fi
    # Deep-search trace is JSONL. Other tests may print ordinary prose.
    output="$trace_dir/$name.stdout.log"
    [[ "$name" == test-agent-flydelta-deep-search ]] && output="$trace"
    if "$bin_dir/$name" >> "$output" 2>> "$trace_dir/stderr.log"; then
        printf '{"event":"check_end","name":"%s","status":"passed"}\n' "$name" >> "$trace"
    else
        code=$?
        printf '{"event":"check_end","name":"%s","status":"failed","exit_code":%d}\n' "$name" "$code" >> "$trace"
        status=1
    fi
done

if [[ -n "$model" ]]; then
    model_binary="$bin_dir/llama-agent-flydelta-dataset-question-repair-model-smoke"
    model_log="$trace_dir/model-smoke.log"
    printf '{"event":"model_start","model":"%s","suite":"%s","mode":"%s"}\n' \
        "$model" "$suite" "$mode" >> "$trace"
    if [[ ! -x "$model_binary" ]]; then
        printf '%s\n' '{"event":"model_end","status":"missing_binary"}' >> "$trace"
        status=1
    elif [[ ! -f "$model" || ! -f "$suite" ]]; then
        printf '%s\n' '{"event":"model_end","status":"missing_model_or_suite"}' >> "$trace"
        status=1
    else
        gpu_available=false
        gpu_inventory=
        if command -v nvidia-smi >/dev/null 2>&1; then
            gpu_inventory=$(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null) &&
                [[ -n "$gpu_inventory" ]] && gpu_available=true
        fi
        if [[ "$gpu_available" == true ]]; then
            device_lister="$bin_dir/llama-cli"
            vulkan_devices=
            if [[ -x "$device_lister" ]]; then
                vulkan_devices=$("$device_lister" --list-devices 2>&1)
            fi
            nvidia_vulkan_index=$(sed -nE 's/^[[:space:]]*Vulkan([0-9]+):.*NVIDIA.*/\1/p' <<<"$vulkan_devices" | head -n 1)
            visible_devices=${GGML_VK_VISIBLE_DEVICES:-$nvidia_vulkan_index}
            selected_devices=
            if [[ -n "$visible_devices" ]]; then
                selected_devices=$(GGML_VK_VISIBLE_DEVICES="$visible_devices" "$device_lister" --list-devices 2>&1)
            fi
            if [[ -z "$nvidia_vulkan_index" || -z "$visible_devices" ||
                    ! "$selected_devices" =~ NVIDIA ]]; then
                printf '%s\n' '{"event":"model_end","status":"nvidia_vulkan_device_not_selectable"}' >> "$trace"
                printf '%s\n' "$vulkan_devices" > "$trace_dir/vulkan-devices.log"
                status=1
            else
            printf '{"event":"model_device","selected":"nvidia-vulkan-auto-fit","visible_devices":"%s","inventory":"%s"}\n' \
                "$visible_devices" "${gpu_inventory//$'\n'/; }" >> "$trace"
            GGML_VK_VISIBLE_DEVICES="$visible_devices" LLAMA_AGENT_EXPECTED_GPU=NVIDIA \
                    LLAMA_AGENT_RESIDENT_TRACE=1 \
                    "$model_binary" --model "$model" --suite "$suite" \
                    --progress-guidance-smoke "$mode" --behavior-key tool_use/dataset/data_filter \
                    --max-scenarios 3 --require-gpu >"$model_log" 2>&1 &
            model_pid=$!
            # Observe this process, rather than relying on optional loader log wording
            # or unrelated activity on the device. Preserve the matching sample.
            (
                while kill -0 "$model_pid" 2>/dev/null; do
                    sample=$(nvidia-smi 2>/dev/null || true)
                    if grep -Eq "[[:space:]]${model_pid}[[:space:]].*[1-9][0-9]*MiB" <<<"$sample"; then
                        printf '%s\n' "$sample" > "$trace_dir/gpu-process.log"
                        break
                    fi
                    sleep 2
                done
            ) &
            monitor_pid=$!
            if wait "$model_pid"; then
                model_exit=0
            else
                model_exit=$?
            fi
            kill "$monitor_pid" 2>/dev/null || true
            wait "$monitor_pid" 2>/dev/null || true
            if [[ "$model_exit" -eq 0 ]]; then
                if ! grep -q 'event=visible-device .*description=.*NVIDIA' "$model_log" ||
                        [[ ! -s "$trace_dir/gpu-process.log" ]] ||
                        ! grep -q '^flydelta_model_progress_guidance ' "$model_log" ||
                        ! grep -q '^flydelta_model_exchange .*"phase":"baseline"' "$model_log" ||
                        ! grep -q '^flydelta_model_exchange .*"phase":"deep_full"' "$model_log" ||
                        ! grep -q '^flydelta_model_exchange .*"overlay_applied":true' "$model_log"; then
                    printf '%s\n' '{"event":"model_end","status":"incomplete_gpu_or_trace_evidence"}' >> "$trace"
                    status=1
                elif grep -q '^flydelta_model_progress_guidance .*"status":"waves_executed".*"posterior_position_executed":true' "$model_log"; then
                    if grep -q '^flydelta_progress_control ' "$model_log" &&
                            grep -q '^flydelta_progress_adviser ' "$model_log" &&
                            grep -q '^flydelta_progress_continuation ' "$model_log" &&
                            grep -q '^flydelta_model_exchange .*"phase":"resume_full"' "$model_log"; then
                        if grep -q '^flydelta_model_progress_guidance .*"anchor_found":true.*"continued_improved":true' "$model_log" ||
                                grep -q '^flydelta_model_progress_guidance .*"anchor_found":true.*"continued_solved":true' "$model_log"; then
                            printf '%s\n' '{"event":"model_end","status":"passed","device":"gpu","fit":"auto","trace":"complete"}' >> "$trace"
                        else
                            printf '%s\n' '{"event":"model_end","status":"inconclusive_anchor_or_further_progress","integration":"posterior_arm_executed","device":"gpu"}' >> "$trace"
                            [[ $status -eq 0 ]] && status=2
                        fi
                    else
                        printf '%s\n' '{"event":"model_end","status":"incomplete_continuation_trace","device":"gpu","fit":"auto"}' >> "$trace"
                        status=1
                    fi
                else
                    printf '%s\n' '{"event":"model_end","status":"inconclusive_no_posterior_execution","device":"gpu","fit":"auto"}' >> "$trace"
                    [[ $status -eq 0 ]] && status=2
                fi
            else
                printf '{"event":"model_end","status":"failed","device":"gpu","exit_code":%d}\n' "$model_exit" >> "$trace"
                status=1
            fi
            fi
        else
            printf '%s\n' '{"event":"model_device","selected":"cpu","reason":"no NVIDIA GPU is visible to this process"}' >> "$trace"
            if LLAMA_AGENT_RESIDENT_TRACE=1 "$model_binary" --model "$model" --suite "$suite" \
                    --progress-guidance-smoke "$mode" --behavior-key tool_use/dataset/data_filter \
                    --max-scenarios 3 --n-gpu-layers 0 >"$model_log" 2>&1; then
                if ! grep -q '^flydelta_smoke_device .*"configured_gpu_layers":0' "$model_log" ||
                        ! grep -q '^flydelta_model_progress_guidance ' "$model_log" ||
                        ! grep -q '^flydelta_model_exchange .*"phase":"baseline"' "$model_log" ||
                        ! grep -q '^flydelta_model_exchange .*"phase":"deep_full"' "$model_log" ||
                        ! grep -q '^flydelta_model_exchange .*"overlay_applied":true' "$model_log"; then
                    printf '%s\n' '{"event":"model_end","status":"incomplete_cpu_or_trace_evidence"}' >> "$trace"
                    status=1
                elif grep -q '^flydelta_model_progress_guidance .*"status":"waves_executed".*"posterior_position_executed":true' "$model_log"; then
                    if grep -q '^flydelta_progress_control ' "$model_log" &&
                            grep -q '^flydelta_progress_adviser ' "$model_log" &&
                            grep -q '^flydelta_progress_continuation ' "$model_log" &&
                            grep -q '^flydelta_model_exchange .*"phase":"resume_full"' "$model_log"; then
                        if grep -q '^flydelta_model_progress_guidance .*"anchor_found":true.*"continued_improved":true' "$model_log" ||
                                grep -q '^flydelta_model_progress_guidance .*"anchor_found":true.*"continued_solved":true' "$model_log"; then
                            printf '%s\n' '{"event":"model_end","status":"passed","device":"cpu","trace":"complete"}' >> "$trace"
                        else
                            printf '%s\n' '{"event":"model_end","status":"inconclusive_anchor_or_further_progress","integration":"posterior_arm_executed","device":"cpu"}' >> "$trace"
                            [[ $status -eq 0 ]] && status=2
                        fi
                    else
                        printf '%s\n' '{"event":"model_end","status":"incomplete_continuation_trace","device":"cpu"}' >> "$trace"
                        status=1
                    fi
                else
                    printf '%s\n' '{"event":"model_end","status":"inconclusive_no_posterior_execution","device":"cpu"}' >> "$trace"
                    [[ $status -eq 0 ]] && status=2
                fi
            else
                code=$?
                printf '{"event":"model_end","status":"failed","device":"cpu","exit_code":%d}\n' "$code" >> "$trace"
                status=1
            fi
        fi
    fi
    printf '{"event":"model_log","path":"%s"}\n' "$model_log" >> "$trace"
fi

if [[ -n "$model" ]]; then
    printf '%s\n' '{"event":"coverage_limit","checkpoint":"separate_queue_contract","resume":"in_memory_deep_search_if_anchor","bayesian_history":"in_memory_trial_evidence","exact_original_baseline_identity":"not_verified","dose_comparison":"not_run","adviser_controls_search":"conditional","independent_regression_controls":"one_fixture","daemon_queue":"not_run","guidance_qualification":"not_run_direct_smoke"}' >> "$trace"
else
    printf '%s\n' '{"event":"coverage_limit","checkpoint":"separate_queue_contract","resume":"in_memory_deep_search","bayesian_history":"in_memory_trial_evidence","exact_original_baseline_identity":"not_verified","real_model":"not_run","dose_comparison":"not_run","adviser_controls_search":false,"independent_regression_controls":"not_run"}' >> "$trace"
fi
finish_status=failed
if [[ $status -eq 0 ]]; then finish_status=passed; elif [[ $status -eq 2 ]]; then finish_status=inconclusive; fi
printf '{"event":"finish","status":"%s"}\n' "$finish_status" >> "$trace"
printf 'Trace: %s\nDiagnostics: %s\n' "$trace" "$trace_dir/stderr.log"
exit "$status"
