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
$exePath = Join-Path $repoRoot "$BuildDir\bin\Release\llama-agent-daemon.exe"

Write-Host "Repo root: $repoRoot"
Write-Host "Build dir: $BuildDir"
Write-Host "Chat model: $ChatModel"

Assert-PathExists -Path $ChatModel -Label "Chat model"

if ($Build) {
    & $cmake --build $BuildDir --config Release --target llama-agent-daemon -j 1
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }
}

Assert-PathExists -Path $exePath -Label "Daemon executable"

$requests = @(
    '{"mode":"chat","prompt":"Reply with OK only.","session_id":"smoke-session","namespace_id":"smoke","project_id":"repo-smoke","memory_scope":"project","plan_scope":"project"}',
    '{"mode":"chat","prompt":"Reply with DONE only.","session_id":"smoke-session","namespace_id":"smoke","project_id":"repo-smoke","memory_scope":"project","plan_scope":"project"}',
    '{"command":"shutdown"}'
)

$output = $requests | & $exePath --model $ChatModel --default-mode chat -n 32 -ngl 0
if ($LASTEXITCODE -ne 0) {
    throw "Daemon exited with code $LASTEXITCODE"
}

$lines = @($output | Where-Object { $_ -and $_.Trim().StartsWith("{") })
if ($lines.Count -lt 4) {
    throw "Expected at least 4 output lines, got $($lines.Count)"
}

$ready = $lines[0] | ConvertFrom-Json
$first = $lines[1] | ConvertFrom-Json
$second = $lines[2] | ConvertFrom-Json
$shutdown = $lines[3] | ConvertFrom-Json

if (-not $ready.ok -or $ready.event -ne "ready") {
    throw "Daemon did not report ready"
}
if (-not $first.ok -or $first.response -ne "OK") {
    throw "First daemon response mismatch: $($lines[1])"
}
if ($first.runtime_reused) {
    throw "First daemon response unexpectedly reported runtime reuse: $($lines[1])"
}
if (-not $second.ok -or $second.response -ne "DONE") {
    throw "Second daemon response mismatch: $($lines[2])"
}
if (-not $second.runtime_reused) {
    throw "Second daemon response did not report runtime reuse: $($lines[2])"
}
if (-not $shutdown.ok -or $shutdown.event -ne "shutdown") {
    throw "Shutdown response mismatch: $($lines[3])"
}

Write-Host ""
Write-Host "Agent daemon smoke test complete."
Write-Host $lines[0]
Write-Host $lines[1]
Write-Host $lines[2]
Write-Host $lines[3]
