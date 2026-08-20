[CmdletBinding()]
param(
    [string]$BuildDir = "build-plan-resident-cozo-debug-3",
    [string]$Configuration = "Release",
    [string]$ChatModel = "$HOME\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    [string]$EmbeddingModel = "$HOME\models\nomic-embed-text-v1.5.Q4_K_M.gguf",
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
$exePath = Join-Path $repoRoot "$BuildDir\bin\$Configuration\llama-agent.exe"

Assert-PathExists -Path $ChatModel -Label "Chat model"
Assert-PathExists -Path $EmbeddingModel -Label "Embedding model"

if ($Build) {
    & $cmake --build $BuildDir --config $Configuration --target llama-agent-cli -- /m:1
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }
}

Assert-PathExists -Path $exePath -Label "CLI executable"

$stdoutPath = Join-Path $env:TEMP "llama-agent-daemon-client-stdout.txt"
$stderrPath = Join-Path $env:TEMP "llama-agent-daemon-client-stderr.txt"
Remove-Item -LiteralPath $stdoutPath -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $stderrPath -ErrorAction SilentlyContinue

try {
    $lines = @("/status", "/help", "/quit")
    (($lines -join "`n") + "`n") |
        & $exePath daemon-session `
            --model $ChatModel `
            --embedding-model $EmbeddingModel `
            --agent-inference-backend server-context `
            --backend in-memory `
            --plan-backend in-memory `
            --resource-blob-backend in-memory `
            --resource-metadata-backend in-memory `
            --prompt "Reply with exactly READY" `
            1> $stdoutPath 2> $stderrPath

    if ($LASTEXITCODE -ne 0) {
        throw "daemon-session exited with code $LASTEXITCODE"
    }

    $stdoutLines = @(Get-Content -LiteralPath $stdoutPath)
    $stderrLines = @(Get-Content -LiteralPath $stderrPath)

    if ($stdoutLines.Count -lt 3) {
        throw "Expected at least 3 stdout lines, got $($stdoutLines.Count)"
    }
    if ($stdoutLines[0] -ne "READY") {
        throw "Expected first stdout line to be READY, got: $($stdoutLines[0])"
    }
    if (-not $stdoutLines[1].StartsWith("[daemon-status] state=ready ")) {
        throw "Expected typed status line on stdout, got: $($stdoutLines[1])"
    }
    if ($stdoutLines[1] -notmatch "sessions=1") {
        throw "Expected status line to include sessions=1, got: $($stdoutLines[1])"
    }
    if ($stdoutLines[2] -ne "[daemon-help] /status /sessions /session /resources /memories /plans /resource-put <path> /resource <uri> /reset /close /drain /quit") {
        throw "Expected help line on stdout, got: $($stdoutLines[2])"
    }
    if ($stderrLines.Count -ne 0) {
        throw "Expected empty stderr, got: $($stderrLines -join ' | ')"
    }

    Write-Host ""
    Write-Host "Daemon client clean IO smoke passed."
    $stdoutLines | ForEach-Object { Write-Host $_ }
}
finally {
    Remove-Item -LiteralPath $stdoutPath -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stderrPath -ErrorAction SilentlyContinue
}
