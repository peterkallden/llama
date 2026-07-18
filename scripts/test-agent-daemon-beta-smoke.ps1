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

& (Join-Path $bin "llama-agent-daemon-mcp-config-smoke.exe")
if ($LASTEXITCODE -ne 0) { throw "MCP/config policy smoke failed" }

powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repoRoot "scripts\test-agent-daemon-tcp-smoke.ps1") `
    -BuildDir $BuildDir -Configuration $Configuration -ChatModel $ChatModel -Port $Port
if ($LASTEXITCODE -ne 0) { throw "TCP policy smoke failed" }

Write-Output "daemon_beta_smoke=passed"
