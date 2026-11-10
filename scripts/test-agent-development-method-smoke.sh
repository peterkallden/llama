#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${LLAMA_BUILD_DIR:-${repo_dir}/build}"
exec "${build_dir}/bin/llama-agent-development-method-smoke" "$@"
