#!/usr/bin/env bash
set -euo pipefail

build_dir="${LLAMA_AGENT_BUILD_DIR:-build}"
model_1="${LLAMA_AGENT_MODEL_1:-}"
model_2="${LLAMA_AGENT_MODEL_2:-}"
backend="${LLAMA_AGENT_MODEL_BACKEND:-server-context}"
extra=()

usage() {
    cat >&2 <<'EOF'
Usage: test-agent-two-model-smoke.sh --model-1 PATH --model-2 PATH
       [--build-dir DIR] [--backend cli|server-context]
       [--threads N] [--n-gpu-layers N]

The model paths may also be supplied through LLAMA_AGENT_MODEL_1 and
LLAMA_AGENT_MODEL_2. This script loads real model files; it is not part of
the default agent smoke suite.
EOF
}

while (($#)); do
    case "$1" in
        --model-1|--model-2|--build-dir|--backend|--threads|--n-gpu-layers)
            if (($# < 2)); then
                echo "missing value for $1" >&2
                exit 2
            fi
            case "$1" in
                --model-1) model_1="$2" ;;
                --model-2) model_2="$2" ;;
                --build-dir) build_dir="$2" ;;
                --backend) backend="$2" ;;
                --threads|--n-gpu-layers) extra+=("$1" "$2") ;;
            esac
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "$model_1" || -z "$model_2" ]]; then
    echo "two-model smoke requires --model-1 and --model-2 (or both environment variables)" >&2
    usage
    exit 2
fi

executable="$build_dir/bin/llama-agent-two-model-smoke"
if [[ ! -x "$executable" ]]; then
    echo "missing $executable; build target llama-agent-two-model-smoke first" >&2
    exit 2
fi

exec "$executable" \
    --model-1 "$model_1" \
    --model-2 "$model_2" \
    --backend "$backend" \
    "${extra[@]}"
