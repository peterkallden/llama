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
    '{"mode":"agent","prompt":"Say OK after making a tiny plan.","session_id":"agent-session-smoke","namespace_id":"agent-session","project_id":"repo-smoke","memory_scope":"project","plan_scope":"project"}',
    '{"mode":"agent","prompt":"Say DONE while continuing the same tiny plan.","session_id":"agent-session-smoke","namespace_id":"agent-session","project_id":"repo-smoke","memory_scope":"project","plan_scope":"project"}',
    '{"command":"shutdown"}'
)

$runId = [guid]::NewGuid().ToString("N")
$requestsPath = Join-Path $env:TEMP "llama-agent-daemon-agent-session-smoke-$runId-requests.txt"
$stdoutPath = Join-Path $env:TEMP "llama-agent-daemon-agent-session-smoke-$runId-stdout.log"
$stderrPath = Join-Path $env:TEMP "llama-agent-daemon-agent-session-smoke-$runId-stderr.log"
Set-Content -LiteralPath $requestsPath -Value $requests -Encoding Ascii
Remove-Item -LiteralPath $stdoutPath -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $stderrPath -ErrorAction SilentlyContinue

try {
    $cmd = "type `"$requestsPath`" | `"$exePath`" --model `"$ChatModel`" --default-mode agent --thinking-mode reflective --memory-learn off --agent-plan auto -n 64 -ngl 0 1> `"$stdoutPath`" 2> `"$stderrPath`""
    cmd /d /c $cmd | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Show-Diagnostics -Path $stderrPath
        throw "Daemon exited with code $LASTEXITCODE"
    }

    $output = Get-Content -LiteralPath $stdoutPath
    $lines = @($output | Where-Object { $_ -and $_.Trim().StartsWith("{") } | ForEach-Object { $message = $_ | ConvertFrom-Json; if ($message.message_type -ne "event") { $_ } })
    if ($lines.Count -ne 4) {
        Show-Diagnostics -Path $stderrPath
        throw "Expected exactly 4 protocol lines, got $($lines.Count)"
    }

    $ready = $lines[0] | ConvertFrom-Json
    $firstTurn = $lines[1] | ConvertFrom-Json
    $secondTurn = $lines[2] | ConvertFrom-Json
    $shutdown = $lines[3] | ConvertFrom-Json

    if (-not $ready.ok -or $ready.event -ne "ready") {
        Show-Diagnostics -Path $stderrPath
        throw "Daemon did not report ready"
    }
    if (-not $firstTurn.ok) {
        Show-Diagnostics -Path $stderrPath
        throw "First agent daemon turn failed: $($lines[1])"
    }
    if (-not $secondTurn.ok) {
        Show-Diagnostics -Path $stderrPath
        throw "Second agent daemon turn failed: $($lines[2])"
    }
    if ($firstTurn.runtime_reused) {
        Show-Diagnostics -Path $stderrPath
        throw "First agent daemon turn unexpectedly reused runtime: $($lines[1])"
    }
    if (-not $secondTurn.runtime_reused) {
        Show-Diagnostics -Path $stderrPath
        throw "Second agent daemon turn did not reuse runtime: $($lines[2])"
    }
    if ($firstTurn.response -ne "OK") {
        Show-Diagnostics -Path $stderrPath
        throw "First agent daemon response mismatch: $($lines[1])"
    }
    if ($secondTurn.response -ne "DONE") {
        Show-Diagnostics -Path $stderrPath
        throw "Second agent daemon response mismatch: $($lines[2])"
    }
    if (-not $firstTurn.plan_id) {
        Show-Diagnostics -Path $stderrPath
        throw "First agent daemon response did not include plan_id: $($lines[1])"
    }
    if ($firstTurn.plan_id -ne $secondTurn.plan_id) {
        Show-Diagnostics -Path $stderrPath
        throw "Agent daemon session did not preserve plan_id across turns: $($lines[1]) / $($lines[2])"
    }
    if (-not $shutdown.ok -or $shutdown.event -ne "shutdown") {
        Show-Diagnostics -Path $stderrPath
        throw "Shutdown response mismatch: $($lines[3])"
    }

    Write-Host ""
    Write-Host "Agent daemon agent session smoke test complete."
    Write-Host $lines[0]
    Write-Host $lines[1]
    Write-Host $lines[2]
    Write-Host $lines[3]
}
finally {
    Remove-Item -LiteralPath $requestsPath -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stdoutPath -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stderrPath -ErrorAction SilentlyContinue
}
