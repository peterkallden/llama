[CmdletBinding()]
param(
    [string]$BuildDir = "build-plan",
    [string]$ChatModel = "$HOME\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
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

function Resolve-CMake {
    $candidates = @(
        "C:\Users\kalld\AppData\Roaming\Python\Python312\Scripts\cmake.exe",
        "cmake"
    )

    foreach ($candidate in $candidates) {
        if ($candidate -eq "cmake") {
            $found = Get-Command cmake -ErrorAction SilentlyContinue
            if ($found) {
                return $found.Source
            }
            continue
        }

        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    throw "Unable to locate cmake.exe"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

$cmake = Resolve-CMake
$cliPath = Join-Path $repoRoot "$BuildDir\bin\Release\llama-agent.exe"
$daemonPath = Join-Path $repoRoot "$BuildDir\bin\Release\llama-agent-daemon.exe"

Write-Host "Repo root: $repoRoot"
Write-Host "Build dir: $BuildDir"
Write-Host "Chat model: $ChatModel"

Assert-PathExists -Path $ChatModel -Label "Chat model"

if ($Build) {
    & $cmake --build $BuildDir --config Release --target llama-agent-cli llama-agent-daemon -j 1
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }
}

Assert-PathExists -Path $cliPath -Label "Agent CLI executable"
Assert-PathExists -Path $daemonPath -Label "Agent daemon executable"

$stdinLines = @(
    "Reply with DONE only."
)

$argumentList = @(
    "daemon-session",
    "--model", $ChatModel,
    "--prompt", "Reply with OK only.",
    "--agent-inference-backend", "server-context",
    "--memory-scope", "project",
    "--memory-namespace", "cli-session",
    "--memory-session", "daemon-session-smoke",
    "--memory-project", "repo-smoke",
    "--plan-scope", "project",
    "--n-predict", "16",
    "-ngl", "0"
)

$output = $stdinLines | & $cliPath @argumentList
if ($LASTEXITCODE -ne 0) {
    throw "daemon-session smoke failed with exit code $LASTEXITCODE"
}

$stdoutLines = @($output | Where-Object { $_ -and $_.Trim().Length -gt 0 })
if ($stdoutLines.Count -lt 2) {
    throw "Expected at least 2 stdout lines, got $($stdoutLines.Count)"
}
if ($stdoutLines[0].Trim() -ne "OK") {
    throw "First daemon-session response mismatch: $($stdoutLines[0])"
}
if ($stdoutLines[1].Trim() -ne "DONE") {
    throw "Second daemon-session response mismatch: $($stdoutLines[1])"
}

Write-Host ""
Write-Host "Agent daemon-session smoke test complete."
Write-Host $stdoutLines[0]
Write-Host $stdoutLines[1]
