#!/usr/bin/env bash
set -euo pipefail

SOCKET_PATH="${1:-/tmp/llama-agent-smoke.sock}"
TOKEN="${LLAMA_AGENT_UNIX_TOKEN:?set LLAMA_AGENT_UNIX_TOKEN}"

printf '{"authorization":"Bearer %s"}\n{"command":"status"}\n{"command":"shutdown"}\n' "$TOKEN" \
  | socat - UNIX-CONNECT:"$SOCKET_PATH"
