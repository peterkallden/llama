[CmdletBinding()]
param(
    [string]$BuildDir = "build-plan",
    [string]$Configuration = "Release",
    [string]$ChatModel = "$HOME\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    [string[]]$ExtraDaemonArgs = @(),
    [string]$PathPrefix = "",
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

function Quote-CmdArg {
    param([string]$Value)

    if ($Value -notmatch '[\s"]') {
        return $Value
    }

    return '"' + $Value.Replace('"', '\"') + '"'
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

function Show-Diagnostics {
    param([string]$Path)

    if (Test-Path -LiteralPath $Path) {
        $diag = @(Get-Content -LiteralPath $Path)
        if ($diag.Count -gt 0) {
            Write-Host ""
            Write-Host "Daemon diagnostics:"
            $diag | Select-Object -First 40 | ForEach-Object { Write-Host $_ }
        }
    }
}

$requests = @(
    '{"mode":"chat","prompt":"Reply with OK only.","session_id":"smoke-session","namespace_id":"smoke","project_id":"repo-smoke","memory_scope":"project","plan_scope":"project"}',
    '{"mode":"chat","prompt":"Reply with DONE only.","session_id":"smoke-session","namespace_id":"smoke","project_id":"repo-smoke","memory_scope":"project","plan_scope":"project"}',
    '{"command":"shutdown"}'
)

$runId = [guid]::NewGuid().ToString("N")
$requestsPath = Join-Path $env:TEMP "llama-agent-daemon-smoke-$runId-requests.txt"
$stdoutPath = Join-Path $env:TEMP "llama-agent-daemon-smoke-$runId-stdout.log"
$stderrPath = Join-Path $env:TEMP "llama-agent-daemon-smoke-$runId-stderr.log"
Set-Content -LiteralPath $requestsPath -Value $requests -Encoding Ascii
Remove-Item -LiteralPath $stdoutPath -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $stderrPath -ErrorAction SilentlyContinue

try {
    if (-not [string]::IsNullOrWhiteSpace($PathPrefix)) {
        $env:PATH = "$PathPrefix;$env:PATH"
    }

    $daemonArgs = @(
        "--model", $ChatModel,
        "--default-mode", "chat",
        "-n", "32",
        "-ngl", "0"
    ) + $ExtraDaemonArgs
    $daemonArgString = ($daemonArgs | ForEach-Object { Quote-CmdArg $_ }) -join " "
    $cmd = "type `"$requestsPath`" | `"$exePath`" $daemonArgString 1> `"$stdoutPath`" 2> `"$stderrPath`""
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
    $first = $lines[1] | ConvertFrom-Json
    $second = $lines[2] | ConvertFrom-Json
    $shutdown = $lines[3] | ConvertFrom-Json

    if (-not $ready.ok -or $ready.event -ne "ready") {
        Show-Diagnostics -Path $stderrPath
        throw "Daemon did not report ready"
    }
    if ($ready.protocol_version -ne 1) {
        Show-Diagnostics -Path $stderrPath
        throw "Daemon ready response missing protocol_version=1"
    }
    if (-not ($ready.capabilities -contains "chat") -or -not ($ready.capabilities -contains "agent")) {
        Show-Diagnostics -Path $stderrPath
        throw "Daemon ready response missing expected capabilities"
    }
    if (-not $first.ok -or $first.response -ne "OK") {
        Show-Diagnostics -Path $stderrPath
        throw "First daemon response mismatch: $($lines[1])"
    }
    if ($first.runtime_reused) {
        Show-Diagnostics -Path $stderrPath
        throw "First daemon response unexpectedly reported runtime reuse: $($lines[1])"
    }
    if (-not $second.ok -or $second.response -ne "DONE") {
        Show-Diagnostics -Path $stderrPath
        throw "Second daemon response mismatch: $($lines[2])"
    }
    if (-not $second.runtime_reused) {
        Show-Diagnostics -Path $stderrPath
        throw "Second daemon response did not report runtime reuse: $($lines[2])"
    }
    if (-not $shutdown.ok -or $shutdown.event -ne "shutdown") {
        Show-Diagnostics -Path $stderrPath
        throw "Shutdown response mismatch: $($lines[3])"
    }

    Write-Host ""
    Write-Host "Agent daemon smoke test complete."
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
