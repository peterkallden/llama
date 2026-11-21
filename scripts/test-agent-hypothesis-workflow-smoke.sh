#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=${LLAMA_AGENT_BUILD_DIR:-"$repo_root/build-agent-vulkan-cozo"}
exec "$build_dir/bin/llama-agent-hypothesis-workflow-smoke" \
    "$repo_root/docs/examples/agent-bootstrap-workflows-v1.json"
