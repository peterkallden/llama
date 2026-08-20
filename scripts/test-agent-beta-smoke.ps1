[CmdletBinding()]
param(
    [string]$BuildDir = "build-plan-resident-cozo-debug-3",
    [string]$Configuration = "Release",
    [switch]$Build,
    [switch]$IncludeDaemon
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$binRoot = Join-Path $repoRoot "$BuildDir\bin\$Configuration"

function Assert-Executable([string]$Name) {
    $path = Join-Path $binRoot $Name
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Smoke executable not found: $path"
    }
    return $path
}

if ($Build) {
    $cmake = Get-Command cmake -ErrorAction Stop
    & $cmake.Source --build $BuildDir --config $Configuration --target llama-agent-smoke-all -j 1
    if ($LASTEXITCODE -ne 0) {
        throw "Beta smoke build failed with exit code $LASTEXITCODE"
    }
}

$smokes = @(
    "llama-agent-runtime-operation-manager-smoke.exe",
    "llama-agent-runtime-session-manager-smoke.exe",
    "llama-agent-mcp-tool-provider-smoke.exe",
    "llama-agent-mcp-stdio-client-smoke.exe",
    "llama-agent-mcp-http-client-smoke.exe",
    "llama-agent-mcp-http-inbound-dispatcher-smoke.exe",
    "llama-agent-mcp-http-vertical-smoke.exe",
    "llama-agent-resource-store-smoke.exe"
)

foreach ($name in $smokes) {
    $path = Assert-Executable $name
    Write-Host "Running $name"
    & $path
    if ($LASTEXITCODE -ne 0) {
        throw "$name failed with exit code $LASTEXITCODE"
    }
}

if ($IncludeDaemon) {
    $model = Join-Path $HOME "models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
    if (-not (Test-Path -LiteralPath $model)) {
        throw "-IncludeDaemon requires the default chat model: $model"
    }
    & (Join-Path $PSScriptRoot "test-agent-daemon-cozo-smoke.ps1") `
        -BuildDir $BuildDir -Configuration $Configuration -ChatModel $model
    if ($LASTEXITCODE -ne 0) {
        throw "Cozo daemon smoke failed with exit code $LASTEXITCODE"
    }
    & (Join-Path $PSScriptRoot "test-agent-daemon-dispatcher-cancel-smoke.ps1") `
        -BuildDir $BuildDir -Configuration $Configuration -ChatModel $model
    if ($LASTEXITCODE -ne 0) {
        throw "Daemon cancellation smoke failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Agent beta smoke pack complete."
