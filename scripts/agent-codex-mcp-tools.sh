#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
server="${LLAMA_AGENT_MCP_STDIO_SERVER:-${repo_root}/build-agent-vulkan-cozo/bin/llama-agent-mcp-stdio-server}"
config="${LLAMA_AGENT_CODEX_MCP_CONFIG:-${repo_root}/docs/examples/agent-host-config-codex-mcp-tools.json}"

if [[ ! -x "${server}" ]]; then
    printf 'llama-agent MCP server binary is missing or not executable: %s\n' "${server}" >&2
    exit 127
fi
if [[ ! -r "${config}" ]]; then
    printf 'llama-agent Codex MCP config is not readable: %s\n' "${config}" >&2
    exit 2
fi

mkdir -p "${repo_root}/var/codex-mcp-tool-audit/resources"
cd "${repo_root}"

# The dedicated Codex audit launcher writes raw host-side tool diagnostics to
# its stderr trace when an operation fails. MCP clients still receive only the
# bounded safe summary.
export LLAMA_AGENT_MCP_TRACE_TOOL_ERRORS=1

exec "${server}" --config "${config}" --repository-root "${repo_root}" "$@"
