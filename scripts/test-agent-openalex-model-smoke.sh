#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/agent-model-smoke-common.sh"

repo_root=$(agent_smoke_repo_root)
build_dir=$(agent_smoke_build_dir)
model="${LLAMA_AGENT_MODEL:-${HOME}/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf}"
embedding_model="${LLAMA_AGENT_EMBEDDING_MODEL:-${HOME}/models/nomic-embed-text-v1.5.Q4_K_M.gguf}"
spec="${LLAMA_AGENT_OPENALEX_SPEC:-${repo_root}/docs/examples/openalex-openapi.json}"
provider_fragment="${LLAMA_AGENT_OPENALEX_PROVIDER:-${repo_root}/docs/examples/openalex-provider.d/openalex.json}"
workflow_package="${LLAMA_AGENT_WORKFLOW_PACKAGE:-${repo_root}/docs/examples/agent-bootstrap-workflows-v1.json}"
work_dir=$(agent_smoke_prepare_workdir openalex-model)
agent_bin=$(agent_smoke_binary "$build_dir" llama-agent)
gpu_layers="${LLAMA_AGENT_GPU_LAYERS:-99}"

agent_smoke_require_file "$model" "chat model"
agent_smoke_require_file "$embedding_model" "embedding model"
agent_smoke_require_file "$spec" "generated OpenAlex OpenAPI spec"
agent_smoke_require_file "$provider_fragment" "generated OpenAlex provider fragment"
agent_smoke_require_file "$workflow_package" "workflow starter package"
agent_smoke_require_executable "$agent_bin" "llama-agent"
agent_smoke_build_if_requested

config="$work_dir/openalex-host-config.json"
python3 - "$config" "$model" "$spec" "$provider_fragment" <<'PY'
import json
import os
import pathlib
import sys

output, model, spec, provider_fragment = sys.argv[1:]
provider = json.loads(pathlib.Path(provider_fragment).read_text(encoding="utf-8"))
provider["spec_path"] = str(pathlib.Path(spec).resolve())
operations = provider.setdefault("policy", {}).setdefault("operations", {})
list_works = operations["listWorks"]
provider["policy"]["operations"] = {"listWorks": list_works}
list_works["bound_arguments"] = {
    "search": "machine learning",
    "per_page": 1,
    "select": "id,display_name,publication_year,primary_topic,topics",
}
pathlib.Path(output).write_text(json.dumps({
    "schema_version": 1,
    "model": {"backend": "server-context", "path": model},
    "tools": {
        "profile": "openalex-smoke",
        "families": {
            "openalex": {
                "description": "Search OpenAlex works by title, author, institution, topic and year"
            }
        },
        "profiles": {
            "openalex-smoke": {
                "allow_network": True,
                "allow_policy_gated_writes": False
            }
        },
        "providers": [provider],
    },
    # The generic CLI default is intentionally short for local tools.  Make
    # the network smoke's budget explicit so the OpenAPI request timeout is
    # not defeated by the outer tool-execution deadline.
    "limits": {
        "tool_timeout_ms": int(os.environ.get("LLAMA_AGENT_TOOL_TIMEOUT_MS", "18000"))
    },
}) + "\n", encoding="utf-8")
PY

log_path="$work_dir/openalex-model.log"
prompt="Call exactly openalex.listWorks (same lowercase spelling), and no other tool, to search for machine learning. The host already supplies search, page size and projection; emit no fields. After it succeeds, answer with the first work id and display name. Do not invent a result."
args=(
    run --config "$config" --model "$model"
    --agent-profile default --tool-profile openalex-smoke
    --plan-backend in-memory
    --agent-bootstrap none --agent-import "$workflow_package" --agent-blueprint openalex-work-search-v1
    --thinking-mode deliberate
    --max-tool-rounds 4
    --require-tool-execution --agent-trace --generation-trace
    --agent-inference-backend server-context
    --prompt "$prompt"
    --n-predict "${LLAMA_AGENT_N_PREDICT:-256}"
    --context-size "${LLAMA_AGENT_CONTEXT_SIZE:-4096}"
    --threads "${LLAMA_AGENT_THREADS:-4}" -ngl "$gpu_layers"
)
if [[ -n "${LLAMA_AGENT_PLANNER_N_PREDICT:-}" ]]; then
    args+=(--planner-n-predict "$LLAMA_AGENT_PLANNER_N_PREDICT")
fi
if [[ -n "$embedding_model" ]]; then
    args+=(--embedding-model "$embedding_model")
fi

LLAMA_AGENT_RESIDENT_TRACE="${LLAMA_AGENT_RESIDENT_TRACE:-1}" \
    agent_smoke_run_logged "$log_path" "$agent_bin" "${args[@]}"
if [[ "$gpu_layers" =~ ^[0-9]+$ ]] && (( gpu_layers >= 99 )); then
    grep -Fq "requested_gpu_layers=$gpu_layers effective_gpu_layers=-1 fit_params=true workspace_reservation=true legacy_full_offload_fit=true" "$log_path" || {
        echo "OpenAlex model smoke did not prove the resident GPU auto-fit path; log=${log_path}" >&2
        exit 1
    }
fi
grep -Eq 'stage=tool kind=(succeeded|completed).*tool=openalex\.listWorks' "$log_path" || {
    echo "OpenAlex model smoke did not execute openalex.listWorks; log=${log_path}" >&2
    exit 1
}
grep -Eq 'route_selection_evaluated.*openalex-work-search|selected_route_id.*openalex-work-search' "$log_path" || {
    echo "OpenAlex model smoke did not select the imported OpenAlex workflow route; log=${log_path}" >&2
    exit 1
}
echo "openalex_model_smoke=passed"
echo "log=${log_path}"
