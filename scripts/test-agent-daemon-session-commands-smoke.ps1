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
    '{"request_id":"1","command":"status"}',
    '{"request_id":"2","mode":"chat","prompt":"Reply with OK only.","session_id":"session-cmds","namespace_id":"smoke","project_id":"repo-smoke","memory_scope":"project","plan_scope":"project"}',
    '{"request_id":"3","command":"status"}',
    '{"request_id":"4","command":"reset_session","session_id":"session-cmds","namespace_id":"smoke","project_id":"repo-smoke"}',
    '{"request_id":"5","command":"close_session","session_id":"session-cmds","namespace_id":"smoke","project_id":"repo-smoke"}',
    '{"request_id":"6","command":"status"}',
    '{"request_id":"7","command":"shutdown"}'
)

$runId = [guid]::NewGuid().ToString("N")
$requestsPath = Join-Path $env:TEMP "llama-agent-daemon-session-cmds-$runId-requests.txt"
$stdoutPath = Join-Path $env:TEMP "llama-agent-daemon-session-cmds-$runId-stdout.log"
$stderrPath = Join-Path $env:TEMP "llama-agent-daemon-session-cmds-$runId-stderr.log"
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
    if ($lines.Count -ne 8) {
        Show-Diagnostics -Path $stderrPath
        throw "Expected exactly 8 protocol lines, got $($lines.Count)"
    }

    $ready = $lines[0] | ConvertFrom-Json
    $initialStatus = $lines[1] | ConvertFrom-Json
    $turn = $lines[2] | ConvertFrom-Json
    $postTurnStatus = $lines[3] | ConvertFrom-Json
    $reset = $lines[4] | ConvertFrom-Json
    $close = $lines[5] | ConvertFrom-Json
    $finalStatus = $lines[6] | ConvertFrom-Json
    $shutdown = $lines[7] | ConvertFrom-Json

    if (-not $ready.ok -or $ready.event -ne "ready") {
        Show-Diagnostics -Path $stderrPath
        throw "Daemon did not report ready"
    }
    if (-not $initialStatus.ok -or $initialStatus.event -ne "status" -or $initialStatus.sessions -ne 0) {
        Show-Diagnostics -Path $stderrPath
        throw "Initial status mismatch: $($lines[1])"
    }
    if (-not $initialStatus.worker_running -or -not $initialStatus.accepting_commands -or
            $initialStatus.shutdown_requested -or $initialStatus.max_queue_size -lt 1) {
        Show-Diagnostics -Path $stderrPath
        throw "Initial lifecycle status mismatch: $($lines[1])"
    }
    if (-not $turn.ok -or $turn.response -ne "OK") {
        Show-Diagnostics -Path $stderrPath
        throw "Turn response mismatch: $($lines[2])"
    }
    if (-not $postTurnStatus.ok -or $postTurnStatus.event -ne "status" -or $postTurnStatus.sessions -ne 1) {
        Show-Diagnostics -Path $stderrPath
        throw "Post-turn status mismatch: $($lines[3])"
    }
    if ($postTurnStatus.session_keys.Count -ne 1 -or
            $postTurnStatus.session_keys[0].namespace_id -ne "smoke" -or
            $postTurnStatus.session_keys[0].session_id -ne "session-cmds") {
        Show-Diagnostics -Path $stderrPath
        throw "Post-turn session key mismatch: $($lines[3])"
    }
    if (-not $reset.ok -or $reset.event -ne "session_reset") {
        Show-Diagnostics -Path $stderrPath
        throw "reset_session response mismatch: $($lines[4])"
    }
    if ($reset.state -ne "ready" -or -not $reset.accepting_commands) {
        Show-Diagnostics -Path $stderrPath
        throw "reset_session lifecycle snapshot mismatch: $($lines[4])"
    }
    if (-not $close.ok -or $close.event -ne "session_closed") {
        Show-Diagnostics -Path $stderrPath
        throw "close_session response mismatch: $($lines[5])"
    }
    if ($close.state -ne "ready" -or -not $close.accepting_commands) {
        Show-Diagnostics -Path $stderrPath
        throw "close_session lifecycle snapshot mismatch: $($lines[5])"
    }
    if (-not $finalStatus.ok -or $finalStatus.event -ne "status" -or $finalStatus.sessions -ne 0) {
        Show-Diagnostics -Path $stderrPath
        throw "Final status mismatch: $($lines[6])"
    }
    if (-not $finalStatus.worker_running -or -not $finalStatus.accepting_commands -or
            $finalStatus.shutdown_requested) {
        Show-Diagnostics -Path $stderrPath
        throw "Final lifecycle status mismatch: $($lines[6])"
    }
    if (-not $shutdown.ok -or $shutdown.event -ne "shutdown") {
        Show-Diagnostics -Path $stderrPath
        throw "Shutdown response mismatch: $($lines[7])"
    }
    if ($shutdown.state -ne "draining" -or -not $shutdown.shutdown_requested -or $shutdown.accepting_commands) {
        Show-Diagnostics -Path $stderrPath
        throw "Shutdown lifecycle snapshot mismatch: $($lines[7])"
    }

    Write-Host ""
    Write-Host "Agent daemon session command smoke test complete."
    Write-Host $lines[0]
    Write-Host $lines[1]
    Write-Host $lines[2]
    Write-Host $lines[3]
    Write-Host $lines[4]
    Write-Host $lines[5]
    Write-Host $lines[6]
    Write-Host $lines[7]
}
finally {
    Remove-Item -LiteralPath $requestsPath -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stdoutPath -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stderrPath -ErrorAction SilentlyContinue
}
