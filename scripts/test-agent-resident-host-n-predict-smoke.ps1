[CmdletBinding()]
param(
    [string]$BuildDir = "build-plan",
    [string]$ChatModel = "$HOME\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    [switch]$Build
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$smokeScript = Join-Path $PSScriptRoot "test-agent-resident-host-smoke.ps1"

if (-not (Test-Path -LiteralPath $smokeScript)) {
    throw "Base resident host smoke script not found: $smokeScript"
}

& $smokeScript `
    -BuildDir $BuildDir `
    -ChatModel $ChatModel `
    -FirstPrompt "Reply with SHORT only." `
    -SecondPrompt "Reply with LONGER only." `
    -FirstNPredict 8 `
    -SecondNPredict 32 `
    -WorkSubdir "work\agent-resident-host-n-predict-smoke" `
    -Build:$Build

Write-Host ""
Write-Host "Resident n_predict keepalive smoke test complete."
