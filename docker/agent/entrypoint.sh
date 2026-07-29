#!/usr/bin/env bash
set -euo pipefail

log_file="${LLAMA_AGENT_LOG_FILE:-/var/log/llama-agent/daemon.log}"
mkdir -p "$(dirname "$log_file")"

# Keep stderr visible to the container runtime and persist the same diagnostics
# in the mounted log volume. JSONL protocol output remains on stdout.
exec "$@" 2> >(tee -a "$log_file" >&2)
