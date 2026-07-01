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

function Show-Diagnostics {
    param([string]$Path)

    if (Test-Path -LiteralPath $Path) {
        $diag = Get-Content -LiteralPath $Path
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
    '{"mode":"mini","prompt":"Say OK after making a tiny plan.","session_id":"mini-smoke-session","namespace_id":"mini-smoke","project_id":"repo-smoke","memory_scope":"project","plan_scope":"project"}',
    '{"command":"shutdown"}'
)

$requestsPath = Join-Path $env:TEMP "llama-agent-daemon-mini-smoke-requests.txt"
$stdoutPath = Join-Path $env:TEMP "llama-agent-daemon-mini-smoke-stdout.log"
$stderrPath = Join-Path $env:TEMP "llama-agent-daemon-mini-smoke-stderr.log"
Set-Content -LiteralPath $requestsPath -Value $requests -Encoding Ascii
Remove-Item -LiteralPath $stdoutPath -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $stderrPath -ErrorAction SilentlyContinue

try {
    $cmd = "type `"$requestsPath`" | `"$exePath`" --model `"$ChatModel`" --default-mode mini --planning-mode mini --reflection-mode off --memory-learn off --agent-plan auto -n 64 -ngl 0 1> `"$stdoutPath`" 2> `"$stderrPath`""
    cmd /d /c $cmd | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Show-Diagnostics -Path $stderrPath
        throw "Daemon exited with code $LASTEXITCODE"
    }

    $output = Get-Content -LiteralPath $stdoutPath
    $lines = @($output | Where-Object { $_ -and $_.Trim().StartsWith("{") })
    if ($lines.Count -ne 3) {
        Show-Diagnostics -Path $stderrPath
        throw "Expected exactly 3 protocol lines, got $($lines.Count)"
    }

    $ready = $lines[0] | ConvertFrom-Json
    $turn = $lines[1] | ConvertFrom-Json
    $shutdown = $lines[2] | ConvertFrom-Json

    if (-not $ready.ok -or $ready.event -ne "ready") {
        Show-Diagnostics -Path $stderrPath
        throw "Daemon did not report ready"
    }
    if (-not $turn.ok) {
        Show-Diagnostics -Path $stderrPath
        throw "Mini daemon turn failed: $($lines[1])"
    }
    if ($turn.response -ne "OK") {
        Show-Diagnostics -Path $stderrPath
        throw "Mini daemon response mismatch: $($lines[1])"
    }
    if (-not $turn.plan_id) {
        Show-Diagnostics -Path $stderrPath
        throw "Mini daemon response did not include plan_id: $($lines[1])"
    }
    if ($turn.event_count -lt 1) {
        Show-Diagnostics -Path $stderrPath
        throw "Mini daemon response did not report any runtime events: $($lines[1])"
    }
    if (-not $shutdown.ok -or $shutdown.event -ne "shutdown") {
        Show-Diagnostics -Path $stderrPath
        throw "Shutdown response mismatch: $($lines[2])"
    }

    Write-Host ""
    Write-Host "Agent daemon mini smoke test complete."
    Write-Host $lines[0]
    Write-Host $lines[1]
    Write-Host $lines[2]
}
finally {
    Remove-Item -LiteralPath $requestsPath -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stdoutPath -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stderrPath -ErrorAction SilentlyContinue
}
