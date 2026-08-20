[CmdletBinding()]
param(
    [string]$BuildDir = "build-plan-resident-cozo-debug",
    [string]$Configuration = "Release",
    [string]$ChatModel = "$HOME\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    [string]$CozoBin = "",
    [switch]$Build
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-PathExists {
    param(
        [string]$Path,
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label not found: $Path"
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$smokeScript = Join-Path $PSScriptRoot "test-agent-daemon-smoke.ps1"
$runId = [guid]::NewGuid().ToString("N")
$memoryDb = Join-Path $env:TEMP "llama-agent-daemon-cozo-smoke-$runId-memory.cozo"
$planDb = Join-Path $env:TEMP "llama-agent-daemon-cozo-smoke-$runId-plan.cozo"

if ([string]::IsNullOrWhiteSpace($CozoBin)) {
    $CozoBin = Join-Path $repoRoot "work\cozo-release\win"
}

Assert-PathExists -Path $smokeScript -Label "Base daemon smoke script"
Assert-PathExists -Path $CozoBin -Label "Cozo runtime directory"

try {
    & $smokeScript `
        -BuildDir $BuildDir `
        -Configuration $Configuration `
        -ChatModel $ChatModel `
        -PathPrefix $CozoBin `
        -ExtraDaemonArgs @(
            "--backend", "cozo",
            "--memory-db", $memoryDb,
            "--plan-backend", "cozo",
            "--plan-db", $planDb
        ) `
        -Build:$Build

    Assert-PathExists -Path $memoryDb -Label "Cozo memory db"
    Assert-PathExists -Path $planDb -Label "Cozo plan db"

    $memoryInfo = Get-Item -LiteralPath $memoryDb
    $planInfo = Get-Item -LiteralPath $planDb

    Write-Host ""
    Write-Host "Agent daemon Cozo smoke test complete."
    Write-Host "Memory DB: $($memoryInfo.FullName) ($($memoryInfo.Length) bytes)"
    Write-Host "Plan DB: $($planInfo.FullName) ($($planInfo.Length) bytes)"
}
finally {
    Remove-Item -LiteralPath $memoryDb -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $planDb -ErrorAction SilentlyContinue
}
