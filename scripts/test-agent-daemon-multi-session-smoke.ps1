[CmdletBinding()]
param(
    [string]$BuildDir = "build-plan",
    [string]$Configuration = "Release",
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

function Show-Diagnostics {
    param([string]$Path)

    if (Test-Path -LiteralPath $Path) {
        $diag = @(Get-Content -LiteralPath $Path)
        if ($diag.Count -gt 0) {
            Write-Host ""
            Write-Host "Daemon diagnostics:"
            $diag | Select-Object -First 60 | ForEach-Object { Write-Host $_ }
        }
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

$cmake = Resolve-CMake
$exePath = Join-Path $repoRoot "$BuildDir\bin\$Configuration\llama-agent-daemon.exe"

Write-Host "Repo root: $repoRoot"
Write-Host "Build dir: $BuildDir"
Write-Host "Configuration: $Configuration"
Write-Host "Chat model: $ChatModel"

Assert-PathExists -Path $ChatModel -Label "Chat model"

if ($Build) {
    & $cmake --build $BuildDir --config $Configuration --target llama-agent-daemon -j 1
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }
}

Assert-PathExists -Path $exePath -Label "Daemon executable"

$requests = @(
    '{"mode":"chat","prompt":"Reply with A1 only.","session_id":"session-a","namespace_id":"multi-smoke","project_id":"repo-smoke","memory_scope":"project","plan_scope":"project"}',
    '{"mode":"chat","prompt":"Reply with B1 only.","session_id":"session-b","namespace_id":"multi-smoke","project_id":"repo-smoke","memory_scope":"project","plan_scope":"project"}',
    '{"mode":"chat","prompt":"Reply with A2 only.","session_id":"session-a","namespace_id":"multi-smoke","project_id":"repo-smoke","memory_scope":"project","plan_scope":"project"}',
    '{"command":"shutdown"}'
)

$runId = [guid]::NewGuid().ToString("N")
$requestsPath = Join-Path $env:TEMP "llama-agent-daemon-multi-session-smoke-$runId-requests.txt"
$stdoutPath = Join-Path $env:TEMP "llama-agent-daemon-multi-session-smoke-$runId-stdout.log"
$stderrPath = Join-Path $env:TEMP "llama-agent-daemon-multi-session-smoke-$runId-stderr.log"
Set-Content -LiteralPath $requestsPath -Value $requests -Encoding Ascii
Remove-Item -LiteralPath $stdoutPath -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $stderrPath -ErrorAction SilentlyContinue

try {
    $cmd = "type `"$requestsPath`" | `"$exePath`" --model `"$ChatModel`" --default-mode chat -n 32 -ngl 0 1> `"$stdoutPath`" 2> `"$stderrPath`""
    cmd /d /c $cmd | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Show-Diagnostics -Path $stderrPath
        throw "Daemon exited with code $LASTEXITCODE"
    }

    $output = Get-Content -LiteralPath $stdoutPath
    $lines = @($output | Where-Object { $_ -and $_.Trim().StartsWith("{") } | ForEach-Object { $message = $_ | ConvertFrom-Json; if ($message.message_type -ne "event") { $_ } })
    if ($lines.Count -ne 5) {
        Show-Diagnostics -Path $stderrPath
        throw "Expected exactly 5 protocol lines, got $($lines.Count)"
    }

    $ready = $lines[0] | ConvertFrom-Json
    $firstTurn = $lines[1] | ConvertFrom-Json
    $secondTurn = $lines[2] | ConvertFrom-Json
    $thirdTurn = $lines[3] | ConvertFrom-Json
    $shutdown = $lines[4] | ConvertFrom-Json

    if (-not $ready.ok -or $ready.event -ne "ready") {
        Show-Diagnostics -Path $stderrPath
        throw "Daemon did not report ready"
    }
    if (-not $firstTurn.ok -or $firstTurn.response -notlike "*A1*") {
        Show-Diagnostics -Path $stderrPath
        throw "First multi-session response mismatch: $($lines[1])"
    }
    if ($firstTurn.runtime_reused) {
        Show-Diagnostics -Path $stderrPath
        throw "First multi-session turn unexpectedly reused runtime: $($lines[1])"
    }
    if (-not $secondTurn.ok -or $secondTurn.response -notlike "*B1*") {
        Show-Diagnostics -Path $stderrPath
        throw "Second multi-session response mismatch: $($lines[2])"
    }
    if ($secondTurn.runtime_reused) {
        Show-Diagnostics -Path $stderrPath
        throw "Second multi-session turn unexpectedly reused runtime: $($lines[2])"
    }
    if (-not $thirdTurn.ok -or [string]::IsNullOrWhiteSpace($thirdTurn.response)) {
        Show-Diagnostics -Path $stderrPath
        throw "Third multi-session response mismatch: $($lines[3])"
    }
    if (-not $thirdTurn.runtime_reused) {
        Show-Diagnostics -Path $stderrPath
        throw "Third multi-session turn did not reuse the prior session-a runtime: $($lines[3])"
    }
    if (-not $shutdown.ok -or $shutdown.event -ne "shutdown") {
        Show-Diagnostics -Path $stderrPath
        throw "Shutdown response mismatch: $($lines[4])"
    }

    Write-Host ""
    Write-Host "Agent daemon multi-session smoke test complete."
    Write-Host $lines[0]
    Write-Host $lines[1]
    Write-Host $lines[2]
    Write-Host $lines[3]
    Write-Host $lines[4]
}
finally {
    Remove-Item -LiteralPath $requestsPath -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stdoutPath -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stderrPath -ErrorAction SilentlyContinue
}
