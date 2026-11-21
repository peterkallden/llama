#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/agent-model-smoke-common.sh"
repo_root=$(agent_smoke_repo_root)
build_dir=$(agent_smoke_build_dir)
model=$(agent_smoke_model)
embedding_model="${LLAMA_AGENT_EMBEDDING_MODEL:-${HOME}/models/nomic-embed-text-v1.5.Q4_K_M.gguf}"
work_dir=$(agent_smoke_prepare_workdir qwen-workflow-action)
agent_bin=$(agent_smoke_binary "$build_dir" llama-agent)
seed_bin=$(agent_smoke_binary "$build_dir" llama-agent-data-store-cozo-seed)
agent_smoke_require_file "$model" "chat model"
agent_smoke_require_file "$embedding_model" "embedding model"
agent_smoke_require_executable "$agent_bin" "llama-agent"
agent_smoke_require_executable "$seed_bin" "Cozo seed executable"
agent_smoke_build_if_requested

orders="$work_dir/orders.csv"
customers="$work_dir/customers.csv"
data_db="$work_dir/data.cozo"
seed_log="$work_dir/seed.log"
agent_log="$work_dir/workflow-action.log"
printf '%s\n' 'order_id,customer_id,amount' '1,10,12' '2,10,8' '3,11,20' > "$orders"
printf '%s\n' 'customer_id,segment' '10,enterprise' '11,consumer' > "$customers"
agent_smoke_run_logged "$seed_log" "$seed_bin" --db "$data_db" --orders "$orders" --customers "$customers"
grep -Eq 'seeded_orders=[1-9]' "$seed_log"

gpu_layers="${LLAMA_AGENT_GPU_LAYERS:-99}"
prompt='Describe only the amount column in the orders dataset using the available dataset-analysis workflow. Count and mean are requested statistics, not column names. Report the count and mean from the executed result; do not guess.'
scenario="${LLAMA_AGENT_WORKFLOW_ACTION_SCENARIO:-overall}"
case "$scenario" in
    overall) ;;
    grouped) prompt='Describe the amount column in orders separately for each customer_id. Report count and mean for every observed customer group from the executed result.' ;;
    *) echo "unsupported workflow action scenario: $scenario" >&2; exit 2 ;;
esac
run_status=0
if LLAMA_AGENT_RESIDENT_TRACE="${LLAMA_AGENT_RESIDENT_TRACE:-1}" \
    LLAMA_AGENT_TIMEOUT_SECONDS="${LLAMA_AGENT_TIMEOUT_SECONDS:-900}" \
    agent_smoke_run_logged "$agent_log" "$agent_bin" run \
    --backend cozo \
    --memory-db "$work_dir/memory.cozo" \
    --plan-backend cozo \
    --plan-db "$work_dir/plan.cozo" \
    --data-backend cozo \
    --data-db "$data_db" \
    --model "$model" \
    --embedding-model "$embedding_model" \
    --agent-profile default \
    --tool-profile analysis \
    --memory-scope session \
    --plan-scope session \
    --tool-output-format "${LLAMA_AGENT_TOOL_OUTPUT_FORMAT:-dsl}" \
    --agent-bootstrap none \
    --agent-import "$repo_root/docs/examples/agent-bootstrap-workflows-v1.json" \
    --agent-blueprint dataset-analysis-v2 \
    --agent-inference-backend server-context \
    --max-tool-rounds 4 \
    --require-tool-execution \
    --agent-trace \
    --generation-trace \
    --plan-show-summary \
    --prompt "$prompt" \
    --n-predict "${LLAMA_AGENT_N_PREDICT:-128}" \
    --context-size "${LLAMA_AGENT_CONTEXT_SIZE:-4096}" \
    --threads "${LLAMA_AGENT_THREADS:-4}" \
    -ngl "$gpu_layers"; then
    :
else
    run_status=$?
fi

if [[ "$gpu_layers" =~ ^[0-9]+$ ]] && (( gpu_layers >= 99 )); then
    grep -Fq "requested_gpu_layers=$gpu_layers effective_gpu_layers=-1 fit_params=true workspace_reservation=true legacy_full_offload_fit=true" "$agent_log" || {
        echo "workflow action smoke did not prove the resident GPU auto-fit path" >&2
        exit 1
    }
fi
grep -Fq 'workflow_action_selected' "$agent_log"
grep -Fq 'analysis_result_requirement_bound' "$agent_log"
grep -Eq 'stage=tool kind=(succeeded|completed).*tool=statistics.describe' "$agent_log"
grep -Eq 'workflow-action trace: attempt=[0-9]+ parse=ok tool=statistics[.]describe canonical_args=.*"columns":\["amount"\]' "$agent_log" || {
    echo "workflow action smoke did not normalize the selected arguments to columns=[amount]" >&2
    exit 1
}
# Schema compliance is separate from whether the requested measures were proven.
python3 - "$agent_log" "$scenario" <<'PY_VERIFY'
import json
import math
import re
import sys
proofs = []
contract_phases = set()
with open(sys.argv[1], encoding="utf-8") as trace:
    for line in trace:
        match = re.search(r"workflow_verification (\{.*\})", line)
        if not match:
            continue
        try:
            record = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if record.get("check") == "analysis-result-contract" and record.get("status") == "passed":
            contract_phases.add(record.get("phase"))
        if record.get("kind") == "declared_numeric_claims" and record.get("status") == "passed":
            proofs.extend(record.get("claims", []))
def proved(pointer, expected, group=None):
    return any(item.get("status") == "passed" and
        item.get("source_location") == pointer and
        (group is None or item.get("group", {}).get("customer_id") == group) and
        isinstance(item.get("value"), (int, float)) and not isinstance(item["value"], bool) and
        math.isclose(item["value"], expected, rel_tol=1e-9, abs_tol=1e-9)
        for item in proofs)
if contract_phases != {"before_tool", "after_tool", "before_answer"}:
    raise SystemExit("workflow action smoke lacks argument, result or answer-coverage proof")
if sys.argv[2] == "overall":
    expected = [("/result/columns/0/count", 3), ("/result/columns/0/mean", 40 / 3)]
else:
    # The fixture has two groups. Group ordering is intentionally not assumed.
    pairs = []
    for group in range(2):
        prefix = f"/result/groups/{group}/columns/0/"
        pairs.append((proved(prefix + "count", 2, 10) and proved(prefix + "mean", 10, 10),
                      proved(prefix + "count", 1, 11) and proved(prefix + "mean", 20, 11)))
    if not (any(a for a, b in pairs) and any(b for a, b in pairs)):
        raise SystemExit("workflow action smoke lacks both fixture customer-group measures")
    expected = []
if any(not proved(pointer, value) for pointer, value in expected):
    raise SystemExit("workflow action smoke lacks host-verified requested count/mean claims")
print("qwen_nomic_workflow_action_numeric_evidence=passed")
PY_VERIFY
echo "qwen_nomic_workflow_action_execution=passed"
if (( run_status != 0 )); then
    echo "full_turn=failed (exit=${run_status}; see log)" >&2
    exit "$run_status"
fi
echo "log=${agent_log}"
