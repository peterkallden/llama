#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Generate a llama-agent daemon configuration.
Usage: agent-config-bootstrap.sh [options]

  --output PATH              Output JSON (default: agent-daemon-config.json)
  --model PATH               Generation model (default: models/model.gguf)
  --embedding-model PATH     Optional embedding model
  --cozo-root PATH           Cozo/data root (default: data)
  --repository-root PATH     Controlled workspace root (default: .)
  --tool-profile NAME        Tool profile (default: all-configured)
  --providers-file PATH      JSON provider array; use - to read stdin
  --threads N                Inference threads (default: 4)
  --gpu-layers N             GPU layers (default: 0)
  --worker-count N           Scheduler workers (default: 2)
  --queue-capacity N         Pending request capacity (default: 8)
  --inference-max-active N   Concurrent inference limit (default: 1)
  --default-mode MODE        chat or agent (default: agent)
  --thinking-mode MODE       auto, reflective, deliberate or research (default: auto)
  --sandbox BACKEND          none, docker or kubernetes (default: none)
  --sandbox-executable PATH  Container executable for the docker backend (default: docker)
  --transport NAME           stdio, mcp-http, jsonl-tcp or jsonl-unix (default: stdio)
  --listen ADDRESS           Bind address (default: 127.0.0.1)
  --port N                   MCP/JSONL TCP port (default: 8080)
  --unix-socket PATH         JSONL Unix socket path
  --auth-mode MODE           none, opaque or jwt (default: none)
  --enable-tools LIST        Comma-separated external tool allowlist
  --list-tools               List known external tools and internal capabilities
  --token-env NAME           Environment variable containing an opaque token
  --token-profile NAME       Tool profile for the opaque token
  --jwt-issuer URL            JWT issuer
  --jwt-audience NAME         JWT audience
  --jwt-jwks-uri URL          JWT JWKS URI
  --jwt-tool-profile NAME     Tool profile for JWT clients
  --jwt-scope NAME            Required JWT scope
  --pdf-page-image-execution MODE  disabled, local_preferred, local_required, sandbox_preferred or sandbox_required
  --pdf-page-image-backend BACKEND auto, local, docker or kubernetes
  --pdf-page-image-executable PATH  PDF renderer (default: mutool)
  --pdf-page-image-version VERSION  Expected renderer version
  --ocr-tesseract-execution MODE    disabled, local_preferred, local_required, sandbox_preferred or sandbox_required
  --ocr-tesseract-backend BACKEND   auto, local, docker or kubernetes
  --ocr-tesseract-executable PATH   OCR executable (default: tesseract)
  --ocr-tesseract-version VERSION   Expected OCR version
  --help                     Show this help
EOF
}

escape_json() {
    local value=$1
    [[ $value != *$'\n'* && $value != *$'\r'* ]] || exit 2
    printf '%s' "$value" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

positive() {
    [[ $2 =~ ^[1-9][0-9]*$ ]] || { echo "$1 must be positive" >&2; exit 2; }
}

output=agent-daemon-config.json
model=models/model.gguf
embedding_model=
cozo_root=data
repository_root=.
tool_profile=all-configured
providers_file=
threads=4
gpu_layers=0
worker_count=2
queue_capacity=8
inference_max_active=1
default_mode=agent
thinking_mode=auto
sandbox=none
sandbox_executable=docker
transport=stdio
listen=127.0.0.1
port=8080
unix_socket=run/llama-agent.sock
auth_mode=none
token_env=
token_profile=
jwt_issuer=
jwt_audience=
jwt_jwks_uri=
jwt_tool_profile=
jwt_scope=
enable_tools_csv=
enable_tools_set=false
pdf_page_image_execution=disabled
pdf_page_image_backend=auto
pdf_page_image_executable=mutool
pdf_page_image_version=
ocr_tesseract_execution=disabled
ocr_tesseract_backend=auto
ocr_tesseract_executable=tesseract
ocr_tesseract_version=

list_tools() {
    cat <<'EOF'
Internal agent capabilities (always available to the agent):
  memory, planning, deliberation, reflection, research, resources

External catalog tools (exposure is policy-controlled):
  calculator
  time_now
  memory_search, memory_get
  memory_inspect, memory_conflict_check
  memory_remember, memory_propose_update, memory_propose_forget
  memory_link, memory_compact_propose
  plan_get, plan_propose
  repository.list, repository.search, repository.read
  repository.diff, repository.log, repository.status, repository.changed_files
  workspace.list, workspace.read, workspace.search, workspace.patch
  diagnostics.compile, diagnostics.symbol, diagnostics.references
  diagnostics.call_hierarchy, diagnostics.test_failures
  diagnostics.format, diagnostics.include_graph
  dataset.list, dataset.inspect, dataset.schema, dataset.sample, dataset.validate
  data.query, data.filter, data.aggregate, data.join, data.transform
  statistics.describe, statistics.outliers, statistics.value_counts
  artifact.export
  resource_read
  web_search, web_fetch
  development.build, development.test

Use --enable-tools with a comma-separated subset when configuring an
authenticated MCP or JSONL client. Internal capabilities are not client tools.
EOF
}

while (($# > 0)); do
    option=$1
    if [[ $option != --help && $option != -h && $option != --list-tools && $# -lt 2 ]]; then
        echo "Missing value for $option" >&2
        exit 2
    fi
    case $option in
        --help|-h) usage; exit 0 ;;
        --list-tools) list_tools; exit 0 ;;
        --output) output=$2 ;;
        --model) model=$2 ;;
        --embedding-model) embedding_model=$2 ;;
        --cozo-root) cozo_root=$2 ;;
        --repository-root) repository_root=$2 ;;
        --tool-profile) tool_profile=$2 ;;
        --providers-file) providers_file=$2 ;;
        --threads) threads=$2 ;;
        --gpu-layers) gpu_layers=$2 ;;
        --worker-count) worker_count=$2 ;;
        --queue-capacity) queue_capacity=$2 ;;
        --inference-max-active) inference_max_active=$2 ;;
        --default-mode) default_mode=$2 ;;
        --thinking-mode) thinking_mode=$2 ;;
        --sandbox) sandbox=$2 ;;
        --sandbox-executable) sandbox_executable=$2 ;;
        --transport) transport=$2 ;;
        --listen) listen=$2 ;;
        --port) port=$2 ;;
        --unix-socket) unix_socket=$2 ;;
        --auth-mode) auth_mode=$2 ;;
        --enable-tools) enable_tools_csv=$2; enable_tools_set=true ;;
        --token-env) token_env=$2 ;;
        --token-profile) token_profile=$2 ;;
        --jwt-issuer) jwt_issuer=$2 ;;
        --jwt-audience) jwt_audience=$2 ;;
        --jwt-jwks-uri) jwt_jwks_uri=$2 ;;
        --jwt-tool-profile) jwt_tool_profile=$2 ;;
        --jwt-scope) jwt_scope=$2 ;;
        --pdf-page-image-execution) pdf_page_image_execution=$2 ;;
        --pdf-page-image-backend) pdf_page_image_backend=$2 ;;
        --pdf-page-image-executable) pdf_page_image_executable=$2 ;;
        --pdf-page-image-version) pdf_page_image_version=$2 ;;
        --ocr-tesseract-execution) ocr_tesseract_execution=$2 ;;
        --ocr-tesseract-backend) ocr_tesseract_backend=$2 ;;
        --ocr-tesseract-executable) ocr_tesseract_executable=$2 ;;
        --ocr-tesseract-version) ocr_tesseract_version=$2 ;;
        *) echo "Unknown option: $option" >&2; usage >&2; exit 2 ;;
    esac
    [[ $option == --help || $option == -h ]] || { [[ $# -ge 2 ]] || exit 2; shift 2; }
done

positive --threads "$threads"
positive --worker-count "$worker_count"
positive --queue-capacity "$queue_capacity"
positive --inference-max-active "$inference_max_active"
[[ $gpu_layers =~ ^[0-9]+$ && $port =~ ^[1-9][0-9]*$ ]] || exit 2
case $default_mode in chat|agent) ;; *) exit 2 ;; esac
case $thinking_mode in auto|reflective|deliberate|research) ;; *) exit 2 ;; esac
case $sandbox in none|docker|kubernetes) ;; *) exit 2 ;; esac
case $transport in stdio|mcp-http|jsonl-tcp|jsonl-unix) ;; *) exit 2 ;; esac
[[ -n $sandbox_executable && $sandbox_executable != *[[:space:]]* ]] || {
    echo "--sandbox-executable must be a non-empty executable name or path without whitespace" >&2
    exit 2
}
case $auth_mode in none|opaque|jwt) ;; *) exit 2 ;; esac
case $pdf_page_image_execution in disabled|local_preferred|local_required|sandbox_preferred|sandbox_required) ;; *) exit 2 ;; esac
case $ocr_tesseract_execution in disabled|local_preferred|local_required|sandbox_preferred|sandbox_required) ;; *) exit 2 ;; esac
case $pdf_page_image_backend in auto|local|docker|kubernetes) ;; *) exit 2 ;; esac
case $ocr_tesseract_backend in auto|local|docker|kubernetes) ;; *) exit 2 ;; esac
if [[ $auth_mode == opaque && ( -z $token_env || -z $token_profile ) ]]; then exit 2; fi
if [[ $auth_mode == jwt && ( -z $jwt_issuer || -z $jwt_audience || -z $jwt_jwks_uri || -z $jwt_tool_profile ) ]]; then exit 2; fi
if [[ $enable_tools_set == true && $auth_mode == none ]]; then
    echo "--enable-tools requires --auth-mode opaque or jwt" >&2
    exit 2
fi

inbound=false
jsonl_tcp=false
jsonl_unix=false
case $transport in
    mcp-http) inbound=true ;;
    jsonl-tcp) jsonl_tcp=true ;;
    jsonl-unix) jsonl_unix=true ;;
esac

model=$(escape_json "$model")
embedding_model=$(escape_json "$embedding_model")
cozo_root=$(escape_json "$cozo_root")
repository_root=$(escape_json "$repository_root")
tool_profile=$(escape_json "$tool_profile")
listen=$(escape_json "$listen")
unix_socket=$(escape_json "$unix_socket")
sandbox_executable=$(escape_json "$sandbox_executable")
pdf_page_image_executable=$(escape_json "$pdf_page_image_executable")
pdf_page_image_version=$(escape_json "$pdf_page_image_version")
ocr_tesseract_executable=$(escape_json "$ocr_tesseract_executable")
ocr_tesseract_version=$(escape_json "$ocr_tesseract_version")

processor_policies_json=''
if [[ $pdf_page_image_execution != disabled ]]; then
    processor_policies_json+="\"pdf.page_image\":{\"execution\":\"$pdf_page_image_execution\",\"backend\":\"$pdf_page_image_backend\",\"executable\":\"$pdf_page_image_executable\",\"expected_version\":\"$pdf_page_image_version\"}"
fi
if [[ $ocr_tesseract_execution != disabled ]]; then
    [[ -n $processor_policies_json ]] && processor_policies_json+=', '
    processor_policies_json+="\"ocr.tesseract\":{\"execution\":\"$ocr_tesseract_execution\",\"backend\":\"$ocr_tesseract_backend\",\"executable\":\"$ocr_tesseract_executable\",\"expected_version\":\"$ocr_tesseract_version\"}"
fi
processor_policies_suffix=
if [[ -n $processor_policies_json ]]; then
    processor_policies_suffix=",\"processor_policies\":{$processor_policies_json}"
fi

providers_json='[]'
if [[ -n $providers_file && $providers_file != none ]]; then
    if [[ $providers_file == - ]]; then
        providers_json=$(cat)
    else
        [[ -f $providers_file ]] || { echo "Provider file not found: $providers_file" >&2; exit 2; }
        providers_json=$(cat "$providers_file")
    fi
    providers_compact=$(printf '%s' "$providers_json" | tr -d '[:space:]')
    [[ $providers_compact == \[*\] ]] || {
        echo "Provider input must be a JSON array" >&2
        exit 2
    }
fi

authorization='"tokens": []'
allowed_tools_suffix=
if [[ $enable_tools_set == true ]]; then
    if [[ $enable_tools_csv == none ]]; then
        allowed_tools_suffix=',"allowed_tools":[]'
    else
        IFS=',' read -r -a enabled_tools <<< "$enable_tools_csv"
        allowed_tools_json=
        for tool in "${enabled_tools[@]}"; do
            [[ -n $tool && $tool != *[[:space:]]* ]] || { echo "Invalid --enable-tools entry" >&2; exit 2; }
            [[ -n $allowed_tools_json ]] && allowed_tools_json+=', '
            allowed_tools_json+="\"$(escape_json "$tool")\""
        done
        allowed_tools_suffix=",\"allowed_tools\":[$allowed_tools_json]"
    fi
fi
if [[ $auth_mode == opaque ]]; then
    authorization=$(printf '%s\n' \
        '      "tokens": [{"id":"bootstrap-client","token_env":"' \
        "$(escape_json "$token_env")" \
        '","audience":"llama-agent","namespace":"local","project":"default","tool_profile":"' \
        "$(escape_json "$token_profile")" \
        '","allow_writes":false}]' | tr -d '\n')
elif [[ $auth_mode == jwt ]]; then
    authorization='"authorization":{"mode":"jwt","issuer":"'"$(escape_json "$jwt_issuer")"'","audience":"'"$(escape_json "$jwt_audience")"'","jwks_uri":"'"$(escape_json "$jwt_jwks_uri")"'","allowed_algorithms":["RS256"],"required_scopes":["'"$(escape_json "$jwt_scope")"'"],"tool_profile":"'"$(escape_json "$jwt_tool_profile")"'","allowed_tools":[],"allow_writes":false}'
fi

if [[ $enable_tools_set == true ]]; then
    if [[ $auth_mode == jwt ]]; then
        jwt_allowed_tools=${allowed_tools_suffix#,}
        authorization=${authorization/\"allowed_tools\":[]/$jwt_allowed_tools}
    else
        authorization=${authorization/\"allow_writes\":false/\"allow_writes\":false$allowed_tools_suffix}
    fi
fi

mkdir -p "$(dirname "$output")"
cat > "$output" <<EOF
{
  "schema_version": 1,
  "model": {"backend":"server-context","path":"$model","embedding_model":"$embedding_model"},
  "runtime": {"context_size":3072,"n_predict":128,"n_threads":$threads,"n_gpu_layers":$gpu_layers,"default_mode":"$default_mode","thinking_mode":"$thinking_mode","max_reflection_rounds":2,"max_plan_revisions":3,"max_research_iterations":4,"memory_learn":"post-turn","agent_plan":"auto","agent_trace":true},
  "stores": {
    "memory":{"backend":"cozo","path":"$cozo_root/memory.cozo"},
    "plan":{"backend":"cozo","path":"$cozo_root/plan.cozo"},
    "data":{"backend":"cozo","path":"$cozo_root/structured.cozo"}
  },
  "resources": {"blob_backend":"fs","blob_root":"$cozo_root/resources","metadata_backend":"cozo","metadata_db":"$cozo_root/resources.cozo"$processor_policies_suffix},
  "tools": {"profile":"$tool_profile","repository_root":"$repository_root","providers":$providers_json},
  "sandbox": {"backend":"$sandbox","docker":{"executable":"$sandbox_executable","default_image":"llama-agent-dev:latest"},"kubernetes":{"namespace":"llama-agent","service_account":"llama-agent-runner","runtime_class":"standard","cleanup":true},"workspace":{"root":"$cozo_root/workspaces","artifact_root":"$cozo_root/artifacts","operation_mode":"ephemeral","project_mode":"persistent"},"defaults":{"timeout_ms":60000,"cpu_count":1,"max_output_bytes":65536,"network":"none","filesystem":"readonly","allow_artifacts":true}},
  "diagnostics": {"semantic_backend":"auto","clang_executable":"clang","clangd_executable":"clangd","compile_commands":"auto"},
  "mcp": {"inbound":{"enabled":$inbound,"listen":"$listen","port":$port,"path":"/mcp","agent_tools":false,"max_delegation_depth":1,$authorization}},
  "jsonl": {"tcp":{"enabled":$jsonl_tcp,"listen":"$listen","port":$port,"max_line_bytes":1048576,"idle_timeout_seconds":300},"unix_socket":{"enabled":$jsonl_unix,"path":"$unix_socket","mode":432}},
  "limits": {"queue_capacity":$queue_capacity,"worker_count":$worker_count,"inference_max_active":$inference_max_active,"turn_timeout_ms":120000,"inference_step_timeout_ms":0,"tool_timeout_ms":30000,"mcp_connect_timeout_ms":5000,"mcp_request_timeout_ms":30000,"mcp_shutdown_timeout_ms":2000,"max_tool_rounds":0}
}
EOF
echo "Wrote $output"
