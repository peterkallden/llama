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
    '{"request_id":"1","mode":"chat","prompt":"Reply with A only.","session_id":"session-shared","namespace_id":"binding-smoke","project_id":"project-a","memory_scope":"project","plan_scope":"project"}',
    '{"request_id":"2","command":"status"}',
    '{"request_id":"3","mode":"chat","prompt":"Reply with B only.","session_id":"session-shared","namespace_id":"binding-smoke","project_id":"project-b","memory_scope":"project","plan_scope":"project"}',
    '{"request_id":"4","command":"status"}',
    '{"request_id":"5","command":"shutdown"}'
)

$runId = [guid]::NewGuid().ToString("N")
$requestsPath = Join-Path $env:TEMP "llama-agent-daemon-session-project-binding-$runId-requests.txt"
$stdoutPath = Join-Path $env:TEMP "llama-agent-daemon-session-project-binding-$runId-stdout.log"
$stderrPath = Join-Path $env:TEMP "llama-agent-daemon-session-project-binding-$runId-stderr.log"
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
    if ($lines.Count -ne 6) {
        Show-Diagnostics -Path $stderrPath
        throw "Expected exactly 6 protocol lines, got $($lines.Count)"
    }

    $ready = $lines[0] | ConvertFrom-Json
    $firstTurn = $lines[1] | ConvertFrom-Json
    $firstStatus = $lines[2] | ConvertFrom-Json
    $secondTurn = $lines[3] | ConvertFrom-Json
    $secondStatus = $lines[4] | ConvertFrom-Json
    $shutdown = $lines[5] | ConvertFrom-Json

    if (-not $ready.ok -or $ready.event -ne "ready") {
        Show-Diagnostics -Path $stderrPath
        throw "Daemon did not report ready"
    }
    if (-not $firstTurn.ok -or $firstTurn.response -ne "A") {
        Show-Diagnostics -Path $stderrPath
        throw "First project-bound turn mismatch: $($lines[1])"
    }
    if (-not $firstStatus.ok -or $firstStatus.sessions -ne 1 -or
            $firstStatus.session_keys.Count -ne 1 -or
            $firstStatus.session_keys[0].project_id -ne "project-a") {
        Show-Diagnostics -Path $stderrPath
        throw "First project-bound status mismatch: $($lines[2])"
    }
    if (-not $secondTurn.ok -or $secondTurn.response -ne "B") {
        Show-Diagnostics -Path $stderrPath
        throw "Second project-bound turn mismatch: $($lines[3])"
    }
    if ($secondTurn.runtime_reused) {
        Show-Diagnostics -Path $stderrPath
        throw "Project switch unexpectedly reused the resident runtime: $($lines[3])"
    }
    if (-not $secondStatus.ok -or $secondStatus.sessions -ne 1 -or
            $secondStatus.session_keys.Count -ne 1 -or
            $secondStatus.session_keys[0].project_id -ne "project-b" -or
            $secondStatus.session_keys[0].session_id -ne "session-shared") {
        Show-Diagnostics -Path $stderrPath
        throw "Second project-bound status mismatch: $($lines[4])"
    }
    if (-not $shutdown.ok -or $shutdown.event -ne "shutdown") {
        Show-Diagnostics -Path $stderrPath
        throw "Shutdown response mismatch: $($lines[5])"
    }

    Write-Host ""
    Write-Host "Agent daemon session/project binding smoke test complete."
    Write-Host $lines[0]
    Write-Host $lines[1]
    Write-Host $lines[2]
    Write-Host $lines[3]
    Write-Host $lines[4]
    Write-Host $lines[5]
}
finally {
    Remove-Item -LiteralPath $requestsPath -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stdoutPath -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stderrPath -ErrorAction SilentlyContinue
}
