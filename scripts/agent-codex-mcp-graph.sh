#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
server="${LLAMA_AGENT_MCP_STDIO_SERVER:-${repo_root}/build-agent-vulkan-cozo/bin/llama-agent-mcp-stdio-server}"
config="${LLAMA_AGENT_GRAPH_CONFIG:-${repo_root}/docs/examples/agent-host-config-codex-mcp-tools.json}"
if [[ ! -x "${server}" || ! -r "${config}" ]]; then
    printf 'The MCP server or graph host configuration is unavailable.\n' >&2
    exit 2
fi
cd "${repo_root}"
exec "${server}" --config "${config}" --repository-root "${repo_root}" --tool-profile graph-project "$@"
