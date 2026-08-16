#!/usr/bin/env bash
set -euo pipefail

build_dir="${LLAMA_AGENT_BUILD_DIR:-build-agent}"
configuration="${LLAMA_AGENT_CONFIGURATION:-RelWithDebInfo}"
parallel="${LLAMA_AGENT_BUILD_PARALLEL_LEVEL:-2}"
verbose=0
targets=()

usage() {
    cat <<'EOF'
Usage: scripts/agent-build.sh [options]

Build an already configured agent build tree through CMake.

Options:
  --build-dir PATH       Build directory (default: $LLAMA_AGENT_BUILD_DIR or build-agent)
  --config NAME          Configuration (default: $LLAMA_AGENT_CONFIGURATION or RelWithDebInfo)
  --target NAME          Target to build; may be repeated
  --parallel N           Parallel build workers (default: $LLAMA_AGENT_BUILD_PARALLEL_LEVEL or 2)
  --verbose              Pass --verbose to CMake
  -h, --help             Show this help
EOF
}

while (($# > 0)); do
    case "$1" in
        --build-dir)
            [[ $# -ge 2 ]] || { echo "Missing value for --build-dir" >&2; exit 2; }
            build_dir="$2"
            shift 2
            ;;
        --config)
            [[ $# -ge 2 ]] || { echo "Missing value for --config" >&2; exit 2; }
            configuration="$2"
            shift 2
            ;;
        --target)
            [[ $# -ge 2 ]] || { echo "Missing value for --target" >&2; exit 2; }
            targets+=("$2")
            shift 2
            ;;
        --parallel)
            [[ $# -ge 2 ]] || { echo "Missing value for --parallel" >&2; exit 2; }
            parallel="$2"
            shift 2
            ;;
        --verbose)
            verbose=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ "$parallel" =~ ^[1-9][0-9]*$ ]] || {
    echo "--parallel must be a positive integer" >&2
    exit 2
}

cmake_bin="${CMAKE_COMMAND:-cmake}"
cmake_args=(--build "$build_dir" --config "$configuration" --parallel "$parallel")
if ((${#targets[@]} > 0)); then
    cmake_args+=(--target "${targets[@]}")
fi
if ((verbose)); then
    cmake_args+=(--verbose)
fi

echo "Building '$build_dir' ($configuration) with $parallel worker(s)."
if ((${#targets[@]} > 0)); then
    printf 'Target(s): %s\n' "${targets[*]}"
fi

exec "$cmake_bin" "${cmake_args[@]}"
