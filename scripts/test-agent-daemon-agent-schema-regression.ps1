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
            $diag | Select-Object -First 120 | ForEach-Object { Write-Host $_ }
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
    '{"mode":"agent","prompt":"Say OK after making a tiny plan.","session_id":"agent-schema-regression","namespace_id":"agent-schema","project_id":"repo-smoke","memory_scope":"project","plan_scope":"project"}',
    '{"command":"shutdown"}'
)

$runId = [guid]::NewGuid().ToString("N")
$requestsPath = Join-Path $env:TEMP "llama-agent-daemon-agent-schema-regression-$runId-requests.txt"
$stdoutPath = Join-Path $env:TEMP "llama-agent-daemon-agent-schema-regression-$runId-stdout.log"
$stderrPath = Join-Path $env:TEMP "llama-agent-daemon-agent-schema-regression-$runId-stderr.log"
Set-Content -LiteralPath $requestsPath -Value $requests -Encoding Ascii
Remove-Item -LiteralPath $stdoutPath -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $stderrPath -ErrorAction SilentlyContinue

$previousTrace = $env:LLAMA_AGENT_RESIDENT_TRACE
$env:LLAMA_AGENT_RESIDENT_TRACE = "1"

try {
    $cmd = "type `"$requestsPath`" | `"$exePath`" --model `"$ChatModel`" --default-mode agent --thinking-mode reflective --memory-learn off --agent-plan auto -n 64 -ngl 0 1> `"$stdoutPath`" 2> `"$stderrPath`""
    cmd /d /c $cmd | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Show-Diagnostics -Path $stderrPath
        throw "Daemon exited with code $LASTEXITCODE"
    }

    $output = Get-Content -LiteralPath $stdoutPath
    $lines = @($output | Where-Object { $_ -and $_.Trim().StartsWith("{") } | ForEach-Object { $message = $_ | ConvertFrom-Json; if ($message.message_type -ne "event") { $_ } })
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
        throw "Agent schema regression turn failed: $($lines[1])"
    }
    if ($turn.response -ne "OK") {
        Show-Diagnostics -Path $stderrPath
        throw "Agent schema regression response mismatch: $($lines[1])"
    }
    if (-not $turn.plan_id) {
        Show-Diagnostics -Path $stderrPath
        throw "Agent schema regression response did not include plan_id: $($lines[1])"
    }
    if (-not $shutdown.ok -or $shutdown.event -ne "shutdown") {
        Show-Diagnostics -Path $stderrPath
        throw "Shutdown response mismatch: $($lines[2])"
    }

    $stderr = @(Get-Content -LiteralPath $stderrPath)
    $plannerTrace = @($stderr | Where-Object { $_ -match "agent resident trace: event=start purpose=planner schema=yes" })
    $reasoningTrace = @($stderr | Where-Object { $_ -match "agent resident trace: event=start purpose=reasoning schema=yes" })
    if ($plannerTrace.Count -lt 1) {
        Show-Diagnostics -Path $stderrPath
        throw "Resident schema regression did not exercise planner schema path"
    }
    if ($reasoningTrace.Count -lt 1) {
        Show-Diagnostics -Path $stderrPath
        throw "Resident schema regression did not exercise reasoning schema path"
    }

    $forbiddenPatterns = @(
        "Access violation",
        "no RTTI data",
        "server returned a non-completion final result",
        "server returned a non-completion streaming result"
    )
    foreach ($pattern in $forbiddenPatterns) {
        if ($stderr | Where-Object { $_ -match [regex]::Escape($pattern) }) {
            Show-Diagnostics -Path $stderrPath
            throw "Resident schema regression detected forbidden diagnostic: $pattern"
        }
    }

    Write-Host ""
    Write-Host "Agent daemon agent schema regression test complete."
    Write-Host $lines[0]
    Write-Host $lines[1]
    Write-Host $lines[2]
}
finally {
    if ($null -eq $previousTrace) {
        Remove-Item Env:LLAMA_AGENT_RESIDENT_TRACE -ErrorAction SilentlyContinue
    } else {
        $env:LLAMA_AGENT_RESIDENT_TRACE = $previousTrace
    }
    Remove-Item -LiteralPath $requestsPath -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stdoutPath -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stderrPath -ErrorAction SilentlyContinue
}
