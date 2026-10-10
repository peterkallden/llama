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
thinking_mode="${LLAMA_AGENT_THINKING_MODE:-reflective}"
case "$thinking_mode" in
    reflective|deliberate|research) ;;
    *) echo "thinking mode must be reflective, deliberate or research" >&2; exit 2 ;;
esac
work_dir=$(agent_smoke_prepare_workdir "openalex-fulltext-${thinking_mode}")
agent_bin=$(agent_smoke_binary "$build_dir" llama-agent)
gpu_layers="${LLAMA_AGENT_GPU_LAYERS:-99}"

agent_smoke_require_file "$model" "chat model"
agent_smoke_require_file "$embedding_model" "embedding model"
agent_smoke_require_file "$spec" "generated OpenAlex OpenAPI spec"
agent_smoke_require_file "$provider_fragment" "generated OpenAlex provider fragment"
agent_smoke_require_file "$workflow_package" "workflow starter package"
agent_smoke_require_executable "$agent_bin" "llama-agent"
agent_smoke_build_if_requested

config="$work_dir/openalex-fulltext-host-config.json"
python3 - "$config" "$model" "$spec" "$provider_fragment" <<'PY'
import json
import pathlib
import sys

output, model, spec, provider_fragment = sys.argv[1:]
provider = json.loads(pathlib.Path(provider_fragment).read_text(encoding="utf-8"))
provider["spec_path"] = str(pathlib.Path(spec).resolve())
operations = provider.setdefault("policy", {}).setdefault("operations", {})
get_work = operations["getWork"]
provider["policy"]["operations"] = {"getWork": get_work}
work_root = pathlib.Path(output).with_name("resource-workspaces")
processor_sandbox = {
    "timeout_ms": 120000,
    "max_output_bytes": 4194304,
    "network": "none",
    "filesystem": "artifact_write",
    "allow_artifacts": True,
}
pathlib.Path(output).write_text(json.dumps({
    "schema_version": 1,
    "model": {"backend": "server-context", "path": model},
    "tools": {
        "profile": "openalex-fulltext-smoke",
        "capabilities": {
            "web.fetch": ["web.fetch"],
            "resource.inspect": ["resource.inspect"],
            "resource.read": ["resource.read"]
        },
        "families": {
            "openalex": {
                "description": "Retrieve verified OpenAlex work metadata, abstract evidence and declared open-access locations"
            }
        },
        "profiles": {
            "openalex-fulltext-smoke": {
                "include_capabilities": ["web.fetch", "resource.inspect", "resource.read"],
                "allow_network": True,
                "allow_policy_gated_writes": False
            }
        },
        "providers": [provider],
    },
    "resources": {
        "processor_policies": {
            "pdf.page_image": {
                "execution": "local_required",
                "backend": "local",
                "executable": "mutool",
                "max_pages": 20
            },
            "ocr.tesseract": {
                "execution": "local_required",
                "backend": "local",
                "executable": "tesseract"
            }
        }
    },
    "sandbox": {
        "workspace": {
            "root": str(work_root / "workspaces"),
            "artifact_root": str(work_root / "artifacts"),
        },
        "classes": {
            "resource.processor.pdf.page_image": processor_sandbox,
            "resource.processor.ocr.tesseract": processor_sandbox,
        },
    },
    "limits": {"tool_timeout_ms": 300000, "max_continuations": 8},
}) + "\n", encoding="utf-8")
PY

log_path="$work_dir/openalex-fulltext-model.log"
prompt='Use the configured OpenAlex full-text workflow for work W3038568908. State whether the article full text was obtained and report one concrete result from the article, based on its extracted text rather than only OpenAlex metadata.'
args=(
    run --config "$config" --model "$model"
    --agent-profile default --tool-profile openalex-fulltext-smoke
    --plan-backend in-memory
    --agent-bootstrap none --agent-import "$workflow_package" --agent-blueprint openalex-full-text-evidence-v2
    --thinking-mode "$thinking_mode"
    --max-tool-rounds 12
    --require-tool-execution --agent-trace --generation-trace --plan-show-summary
    --agent-inference-backend server-context
    --prompt "$prompt"
    --n-predict "${LLAMA_AGENT_N_PREDICT:-512}"
    --context-size "${LLAMA_AGENT_CONTEXT_SIZE:-8192}"
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
        echo "OpenAlex full-text model smoke did not prove the resident GPU auto-fit path; log=$log_path" >&2
        exit 1
    }
fi
grep -Eq 'stage=tool kind=(succeeded|completed).*tool=openalex\.getWork' "$log_path" || {
    echo "OpenAlex full-text model smoke did not execute openalex.getWork; log=$log_path" >&2
    exit 1
}
grep -Eq 'stage=tool kind=(succeeded|completed).*tool=web\.fetch' "$log_path" || {
    echo "OpenAlex full-text model smoke did not execute web.fetch; log=$log_path" >&2
    exit 1
}
grep -Eq 'agent workflow-choice trace: transition=choose-evidence-depth.*selected_id=fetch_full_text' "$log_path" || {
    echo "OpenAlex full-text model smoke did not record the host-bounded full-text choice; log=$log_path" >&2
    exit 1
}
grep -Eq 'stage=tool kind=(succeeded|completed|failed).*tool=resource\.inspect' "$log_path" || {
    echo "OpenAlex full-text model smoke did not attempt resource.inspect; log=$log_path" >&2
    exit 1
}
grep -Eq 'stage=tool kind=(succeeded|completed).*tool=resource\.read' "$log_path" || {
    echo "OpenAlex full-text model smoke did not successfully read a text representation; log=$log_path" >&2
    exit 1
}
grep -Fq 'detail=bounded resource chunk plan attached' "$log_path" || {
    echo "OpenAlex full-text smoke did not hand the workflow-read text resource to the chunker; log=$log_path" >&2
    exit 1
}
grep -Fq 'detail=bounded resource chunk processed' "$log_path" || {
    echo "OpenAlex full-text smoke did not process any bounded text chunks; log=$log_path" >&2
    exit 1
}
grep -Fq 'processed 7 of 7 pages.' "$log_path" || {
    echo "OpenAlex full-text smoke did not OCR all configured PDF pages; log=$log_path" >&2
    exit 1
}
python3 - "$log_path" <<'PY'
import pathlib
import re
import sys

log = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
counts = re.findall(r"chunk_count=(\d+)", log)
if not counts:
    raise SystemExit("smoke trace did not expose the host's planned chunk count")
expected = int(counts[0])
processed = log.count("detail=bounded resource chunk processed")
extraction_blocks = re.findall(
    r"agent generation trace prompt: purpose=resource_chunk_extraction.*?\n(.*?)\nagent generation trace prompt: end",
    log,
    re.S,
)
extraction_prompts = len(extraction_blocks)
if expected <= 0 or processed != expected:
    raise SystemExit(
        f"validated chunk count mismatch: planned={expected}, processed={processed}")
if extraction_prompts < expected:
    raise SystemExit(
        f"host chunk text was not submitted for every extraction: planned={expected}, prompts={extraction_prompts}")
if "Resource text (untrusted data):" not in log:
    raise SystemExit("chunk extraction trace omitted the actual host-read text context")
for block in extraction_blocks:
    lowered = block.lower()
    if "resource.read" in lowered or "read this slice" in lowered or "read the chunk" in lowered:
        raise SystemExit("tool-free extraction prompt asks the model to read a chunk itself")
    if "chunk_index" in lowered or "cursor" in lowered:
        raise SystemExit("tool-free extraction prompt exposes host-owned chunk identity or cursor")

decoded = list(re.finditer(r"(?m)^decoded \d+ tokens$", log))
if not decoded:
    raise SystemExit("smoke log did not contain the final model response")
answer = log[decoded[-1].end():].strip()
answer_lower = answer.lower()
if not any(term in answer_lower for term in ("borated", "mcnp", "anneal", "bright spot", "radiation flux")):
    raise SystemExit("final answer did not include an identifiable full-text-supported finding")
if re.search(r"(?:still need|needs? to be|must be|should be) fetched|need to fetch.*(?:pdf|full text)|fetch.*to obtain.*full.text", answer_lower):
    raise SystemExit("final answer incorrectly says the PDF/full text still needs fetching")
PY
grep -Eq 'route_selection_evaluated.*openalex-full-text-evidence|selected_route_id.*openalex-full-text-evidence' "$log_path" || {
    echo "OpenAlex full-text model smoke did not select the imported full-text workflow route; log=$log_path" >&2
    exit 1
}
if [[ "$thinking_mode" == research ]]; then
    grep -Fq 'research acquisition started' "$log_path" || {
        echo "OpenAlex full-text research smoke did not enter the research controller; log=$log_path" >&2
        exit 1
    }
    grep -Fq 'research source recorded' "$log_path" || {
        echo "OpenAlex full-text research smoke did not seed a workflow source; log=$log_path" >&2
        exit 1
    }
    grep -Fq 'research evidence recorded' "$log_path" || {
        echo "OpenAlex full-text research smoke did not produce research evidence; log=$log_path" >&2
        exit 1
    }
fi
echo "openalex_fulltext_model_smoke=passed"
echo "thinking_mode=$thinking_mode"
echo "log=$log_path"
