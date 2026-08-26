#!/usr/bin/env bash
set -euo pipefail

log_file="${LLAMA_AGENT_LOG_FILE:-/var/log/llama-agent/daemon.log}"
mkdir -p "$(dirname "$log_file")"

config_file="${LLAMA_AGENT_CONFIG_FILE:-/etc/llama-agent/config.json}"
bootstrap="${LLAMA_AGENT_CONFIG_BOOTSTRAP:-/opt/llama-agent/bin/llama-agent-config-bootstrap}"
mode="${LLAMA_AGENT_MODE:-all}"
daemon_port="${LLAMA_AGENT_DAEMON_PORT:-8091}"
web_adapter_port="${LLAMA_AGENT_WEB_ADAPTER_PORT:-8090}"
mcp_token_file="${LLAMA_AGENT_MCP_TOKEN_FILE:-/etc/llama-agent/mcp-dev-token}"

# The development image has an inbound MCP config. Keep its token convenient
# for local testing, but never create or print one for release images.
if [[ "${LLAMA_AGENT_IMAGE_CHANNEL:-release}" == "dev" && -z "${LLAMA_AGENT_MCP_TOKEN:-}" ]]; then
    if [[ -s "$mcp_token_file" ]]; then
        LLAMA_AGENT_MCP_TOKEN="$(head -n 1 "$mcp_token_file")"
        export LLAMA_AGENT_MCP_TOKEN
        echo "Using development MCP bearer token from $mcp_token_file" >&2
    else
        LLAMA_AGENT_MCP_TOKEN="${LLAMA_AGENT_DEV_TOKEN:-dev-token}"
        printf '%s\n' "$LLAMA_AGENT_MCP_TOKEN" > "$mcp_token_file"
        chmod 0600 "$mcp_token_file"
        export LLAMA_AGENT_MCP_TOKEN
        echo "Generated development MCP bearer token: $LLAMA_AGENT_MCP_TOKEN" >&2
        echo "Persisted development MCP bearer token in $mcp_token_file" >&2
    fi
fi

case "$mode" in
    daemon|all) ;;
    *)
        echo "LLAMA_AGENT_MODE must be daemon or all" >&2
        exit 2
        ;;
esac

bootstrap_transport=stdio
if [[ "$mode" == all ]]; then
    bootstrap_transport=jsonl-tcp
fi

if [[ ! -s "$config_file" && "${LLAMA_AGENT_IMAGE_CHANNEL:-release}" == "dev" && \
      -s /opt/llama-agent/share/llama-agent/examples/agent-config.docker.example.json ]]; then
    mkdir -p "$(dirname "$config_file")"
    cp /opt/llama-agent/share/llama-agent/examples/agent-config.docker.example.json "$config_file"
    echo "Installed development Docker configuration at $config_file" >&2
fi

if [[ ! -s "$config_file" ]]; then
    mkdir -p "$(dirname "$config_file")"
    bootstrap_args=(
        --output "$config_file"
        --model "${LLAMA_AGENT_MODEL_PATH:-/models/model.gguf}"
        --embedding-model "${LLAMA_AGENT_EMBEDDING_MODEL_PATH:-/models/embedding.gguf}"
        --cozo-root "${LLAMA_AGENT_COZO_ROOT:-/var/lib/llama-agent/data}"
        --repository-root "${LLAMA_AGENT_REPOSITORY_ROOT:-/var/lib/llama-agent/workspace}"
        --tool-profile "${LLAMA_AGENT_TOOL_PROFILE:-all-configured}"
        --threads "${LLAMA_AGENT_THREADS:-4}"
        --gpu-layers "${LLAMA_AGENT_GPU_LAYERS:-0}"
        --worker-count "${LLAMA_AGENT_WORKER_COUNT:-2}"
        --queue-capacity "${LLAMA_AGENT_QUEUE_CAPACITY:-8}"
        --inference-max-active "${LLAMA_AGENT_INFERENCE_MAX_ACTIVE:-1}"
        --default-mode "${LLAMA_AGENT_DEFAULT_MODE:-agent}"
        --thinking-mode "${LLAMA_AGENT_THINKING_MODE:-auto}"
        --sandbox "${LLAMA_AGENT_SANDBOX:-none}"
        --transport "$bootstrap_transport"
        --listen 127.0.0.1
        --port "$daemon_port"
        --pdf-page-image-execution "${LLAMA_AGENT_PDF_PAGE_IMAGE_EXECUTION:-local_preferred}"
        --pdf-page-image-backend "${LLAMA_AGENT_PDF_PAGE_IMAGE_BACKEND:-local}"
        --pdf-page-image-executable "${LLAMA_AGENT_PDF_PAGE_IMAGE_EXECUTABLE:-mutool}"
        --ocr-tesseract-execution "${LLAMA_AGENT_OCR_TESSERACT_EXECUTION:-local_preferred}"
        --ocr-tesseract-backend "${LLAMA_AGENT_OCR_TESSERACT_BACKEND:-local}"
        --ocr-tesseract-executable "${LLAMA_AGENT_OCR_TESSERACT_EXECUTABLE:-tesseract}"
        --pandoc-execution "${LLAMA_AGENT_PANDOC_EXECUTION:-local_preferred}"
        --pandoc-backend "${LLAMA_AGENT_PANDOC_BACKEND:-local}"
        --pandoc-executable "${LLAMA_AGENT_PANDOC_EXECUTABLE:-pandoc}"
    )
    [[ -n "${LLAMA_AGENT_TCP_AUTH_MODE:-}" ]] && bootstrap_args+=(--auth-mode "$LLAMA_AGENT_TCP_AUTH_MODE")
    [[ -n "${LLAMA_AGENT_TCP_TOKEN_ENV:-}" ]] && bootstrap_args+=(--token-env "$LLAMA_AGENT_TCP_TOKEN_ENV")
    [[ -n "${LLAMA_AGENT_TCP_TOKEN_PROFILE:-}" ]] && bootstrap_args+=(--token-profile "$LLAMA_AGENT_TCP_TOKEN_PROFILE")
    "$bootstrap" "${bootstrap_args[@]}"
fi

if [[ "$mode" == daemon ]]; then
    if [[ "${1:-}" == "llama-agent-daemon" ]]; then
        has_config=false
        for argument in "$@"; do
            if [[ "$argument" == "--config" || "$argument" == --config=* ]]; then
                has_config=true
                break
            fi
        done
        if [[ "$has_config" == false ]]; then
            set -- "$@" --config "$config_file"
        fi
    fi
    exec "$@" 2> >(tee -a "$log_file" >&2)
fi

if [[ "${1:-}" != "llama-agent-daemon" ]]; then
    exec "$@"
fi

if [[ "${LLAMA_AGENT_WEB_TLS:-false}" == true ]]; then
    tls_cert="${LLAMA_AGENT_TLS_CERT:-/etc/llama-agent/tls/server.crt}"
    tls_key="${LLAMA_AGENT_TLS_KEY:-/etc/llama-agent/tls/server.key}"
    if [[ -s "$tls_cert" && -s "$tls_key" ]]; then
        :
    elif [[ ! -e "$tls_cert" && ! -e "$tls_key" ]]; then
        mkdir -p "$(dirname "$tls_cert")"
        openssl req -x509 -nodes -newkey rsa:2048 -days 365 \
            -keyout "$tls_key" \
            -out "$tls_cert" \
            -subj "/CN=localhost" \
            -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" \
            >/dev/null 2>&1
        echo "Generated self-signed TLS certificate: $tls_cert" >&2
        openssl x509 -in "$tls_cert" -noout -fingerprint -sha256 >&2
    else
        echo "TLS requires both LLAMA_AGENT_TLS_CERT and LLAMA_AGENT_TLS_KEY" >&2
        exit 2
    fi

    export LLAMA_AGENT_TLS_CERT="$tls_cert"
    export LLAMA_AGENT_TLS_KEY="$tls_key"
    envsubst '${LLAMA_AGENT_TLS_CERT} ${LLAMA_AGENT_TLS_KEY}' \
        < /opt/llama-agent/nginx/llama-agent-web.https.conf.template \
        > /etc/llama-agent/nginx.conf
else
    cp /opt/llama-agent/nginx/llama-agent-web.http.conf /etc/llama-agent/nginx.conf
fi

nginx -t -c /etc/llama-agent/nginx.conf

daemon_args=("$@" --config "$config_file" --tcp-listen 127.0.0.1 --tcp-port "$daemon_port")
web_args=(
    /opt/llama-agent/bin/llama-agent-web
    --daemon-address 127.0.0.1
    --daemon-port "$daemon_port"
    --listen 127.0.0.1
    --port "$web_adapter_port"
)
if [[ -n "${LLAMA_AGENT_WEB_BEARER_TOKEN:-}" ]]; then
    web_args+=(--web-bearer-token "$LLAMA_AGENT_WEB_BEARER_TOKEN")
fi
if [[ -n "${LLAMA_AGENT_DAEMON_AUTHORIZATION:-}" ]]; then
    web_args+=(--daemon-authorization "$LLAMA_AGENT_DAEMON_AUTHORIZATION")
fi

"${daemon_args[@]}" 2> >(tee -a "$log_file" >&2) &
daemon_pid=$!
"${web_args[@]}" 2> >(tee -a /var/log/llama-agent/web.log >&2) &
web_pid=$!
nginx -c /etc/llama-agent/nginx.conf -g 'daemon off;' \
    2> >(tee -a /var/log/llama-agent/nginx.log >&2) &
nginx_pid=$!

cleanup() {
    kill "$nginx_pid" "$web_pid" "$daemon_pid" 2>/dev/null || true
    wait "$nginx_pid" "$web_pid" "$daemon_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait -n "$nginx_pid" "$web_pid" "$daemon_pid"
exit $?
