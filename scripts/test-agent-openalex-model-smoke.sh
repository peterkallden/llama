#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/agent-model-smoke-common.sh"

repo_root=$(agent_smoke_repo_root)
build_dir=$(agent_smoke_build_dir)
model="${LLAMA_AGENT_MODEL:-${HOME}/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf}"
embedding_model="${LLAMA_AGENT_EMBEDDING_MODEL:-}"
spec="${LLAMA_AGENT_OPENALEX_SPEC:-${repo_root}/docs/examples/openalex-works-openapi.json}"
work_dir=$(agent_smoke_prepare_workdir openalex-model)
if [[ "$build_dir" = /* ]]; then
    agent_bin="${build_dir}/bin/llama-agent"
else
    agent_bin="${repo_root}/${build_dir}/bin/llama-agent"
fi

agent_smoke_require_file "$model" "chat model"
agent_smoke_require_file "$spec" "OpenAlex OpenAPI smoke contract"
agent_smoke_require_executable "$agent_bin" "llama-agent"
agent_smoke_build_if_requested

config="$work_dir/openalex-host-config.json"
python3 - "$config" "$model" "$spec" <<'PY'
import json
import pathlib
import sys

output, model, spec = sys.argv[1:]
pathlib.Path(output).write_text(json.dumps({
    "schema_version": 1,
    "model": {"backend": "server-context", "path": model},
    "tools": {
        "profile": "openalex-smoke",
        "families": {
            "openalex": {
                "description": "Search and retrieve scholarly works from OpenAlex"
            }
        },
        "profiles": {
            "openalex-smoke": {
                "allow_network": True,
                "allow_policy_gated_writes": False
            }
        },
        "providers": [{
            "type": "openapi",
            "id": "openalex",
            "enabled": True,
            "required": True,
            "spec_path": str(pathlib.Path(spec).resolve()),
            "base_url": "https://api.openalex.org",
            "prefix": "openalex",
            "policy": {"access": "read_only", "exposure": "auto"},
            "auth": {"type": "none"},
        }],
    },
}) + "\n", encoding="utf-8")
PY

log_path="$work_dir/openalex-model.log"
prompt="Use only the openalex.listWorks tool. Call it with the search argument set to machine learning, per_page set to 1, and select set to id,display_name. After the tool succeeds, answer with the first work id and display name. Do not invent a result."
args=(
    run --config "$config" --model "$model"
    --agent-profile default --tool-profile openalex-smoke
    --thinking-mode deliberate --agent-plan off
    --max-tool-rounds 4
    --require-tool-execution --agent-trace --generation-trace
    --prompt "$prompt"
    --n-predict "${LLAMA_AGENT_N_PREDICT:-256}"
    --context-size "${LLAMA_AGENT_CONTEXT_SIZE:-4096}"
    --threads "${LLAMA_AGENT_THREADS:-4}" -ngl "${LLAMA_AGENT_GPU_LAYERS:-0}"
)
if [[ -n "$embedding_model" ]]; then
    args+=(--embedding-model "$embedding_model")
fi

agent_smoke_run_logged "$log_path" "$agent_bin" "${args[@]}"
grep -Eq 'stage=tool kind=(succeeded|completed).*tool=openalex\.listWorks' "$log_path" || {
    echo "OpenAlex model smoke did not execute openalex.listWorks; log=${log_path}" >&2
    exit 1
}
echo "openalex_model_smoke=passed"
echo "log=${log_path}"
