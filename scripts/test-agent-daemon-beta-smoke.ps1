[CmdletBinding()]
param(
    [string]$BuildDir = "build-plan-resident-cozo-debug-3",
    [string]$Configuration = "Release",
    [string]$ChatModel = "$HOME\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    [int]$Port = 18090
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$bin = Join-Path $repoRoot "$BuildDir\bin\$Configuration"

$deterministic = @(
    "llama-agent-runtime-operation-manager-smoke",
    "llama-agent-runtime-session-manager-smoke",
    "llama-agent-daemon-dispatcher-smoke",
    "llama-agent-daemon-protocol-smoke",
    "llama-agent-daemon-jsonl-protocol-smoke",
    "llama-agent-daemon-mcp-config-smoke",
    "llama-agent-resource-store-smoke",
    "llama-agent-mcp-tool-provider-smoke",
    "llama-agent-mcp-stdio-client-smoke",
    "llama-agent-mcp-http-client-smoke",
    "llama-agent-mcp-http-inbound-dispatcher-smoke",
    "llama-agent-daemon-wait-events-smoke"
)
foreach ($target in $deterministic) {
    $path = Join-Path $bin "$target.exe"
    if (-not (Test-Path -LiteralPath $path)) { throw "Smoke binary not found: $path" }
    & $path
    if ($LASTEXITCODE -ne 0) { throw "$target failed with exit code $LASTEXITCODE" }
}

powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repoRoot "scripts\test-agent-daemon-tcp-smoke.ps1") `
    -BuildDir $BuildDir -Configuration $Configuration -ChatModel $ChatModel -Port $Port
if ($LASTEXITCODE -ne 0) { throw "TCP policy smoke failed" }

powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repoRoot "scripts\test-agent-daemon-smoke.ps1") `
    -BuildDir $BuildDir -Configuration $Configuration -ChatModel $ChatModel
if ($LASTEXITCODE -ne 0) { throw "foreground daemon smoke failed" }

Write-Output "daemon_beta_smoke=passed"
