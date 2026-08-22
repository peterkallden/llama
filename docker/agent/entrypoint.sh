#!/usr/bin/env bash
set -euo pipefail

log_file="${LLAMA_AGENT_LOG_FILE:-/var/log/llama-agent/daemon.log}"
mkdir -p "$(dirname "$log_file")"

config_file="${LLAMA_AGENT_CONFIG_FILE:-/etc/llama-agent/config.json}"
bootstrap="${LLAMA_AGENT_CONFIG_BOOTSTRAP:-/opt/llama-agent/bin/llama-agent-config-bootstrap}"

if [[ ! -s "$config_file" ]]; then
    mkdir -p "$(dirname "$config_file")"
    "$bootstrap" \
        --output "$config_file" \
        --model "${LLAMA_AGENT_MODEL_PATH:-/models/model.gguf}" \
        --embedding-model "${LLAMA_AGENT_EMBEDDING_MODEL_PATH:-/models/embedding.gguf}" \
        --cozo-root "${LLAMA_AGENT_COZO_ROOT:-/var/lib/llama-agent/data}" \
        --repository-root "${LLAMA_AGENT_REPOSITORY_ROOT:-/var/lib/llama-agent/workspace}" \
        --tool-profile "${LLAMA_AGENT_TOOL_PROFILE:-all-configured}" \
        --threads "${LLAMA_AGENT_THREADS:-4}" \
        --gpu-layers "${LLAMA_AGENT_GPU_LAYERS:-0}" \
        --worker-count "${LLAMA_AGENT_WORKER_COUNT:-2}" \
        --queue-capacity "${LLAMA_AGENT_QUEUE_CAPACITY:-8}" \
        --inference-max-active "${LLAMA_AGENT_INFERENCE_MAX_ACTIVE:-1}" \
        --default-mode "${LLAMA_AGENT_DEFAULT_MODE:-agent}" \
        --thinking-mode "${LLAMA_AGENT_THINKING_MODE:-auto}" \
        --sandbox "${LLAMA_AGENT_SANDBOX:-none}" \
        --pdf-page-image-execution "${LLAMA_AGENT_PDF_PAGE_IMAGE_EXECUTION:-local_preferred}" \
        --pdf-page-image-backend "${LLAMA_AGENT_PDF_PAGE_IMAGE_BACKEND:-local}" \
        --pdf-page-image-executable "${LLAMA_AGENT_PDF_PAGE_IMAGE_EXECUTABLE:-mutool}" \
        --ocr-tesseract-execution "${LLAMA_AGENT_OCR_TESSERACT_EXECUTION:-local_preferred}" \
        --ocr-tesseract-backend "${LLAMA_AGENT_OCR_TESSERACT_BACKEND:-local}" \
        --ocr-tesseract-executable "${LLAMA_AGENT_OCR_TESSERACT_EXECUTABLE:-tesseract}" \
        --pandoc-execution "${LLAMA_AGENT_PANDOC_EXECUTION:-local_preferred}" \
        --pandoc-backend "${LLAMA_AGENT_PANDOC_BACKEND:-local}" \
        --pandoc-executable "${LLAMA_AGENT_PANDOC_EXECUTABLE:-pandoc}" \
        --transport stdio
fi

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

# Keep stderr visible to the container runtime and persist the same diagnostics
# in the mounted log volume. JSONL protocol output remains on stdout.
exec "$@" 2> >(tee -a "$log_file" >&2)
