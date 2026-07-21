[CmdletBinding()]
param(
    [string]$BuildDir = "build-plan",
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

function Show-Diagnostics {
    param([string]$Path)

    if (Test-Path -LiteralPath $Path) {
        $diag = @(Get-Content -LiteralPath $Path)
        if ($diag.Count -gt 0) {
            Write-Host ""
            Write-Host "Daemon diagnostics:"
            $diag | Select-Object -First 80 | ForEach-Object { Write-Host $_ }
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
Write-Host "Embedding model: $EmbeddingModel"

Assert-PathExists -Path $ChatModel -Label "Chat model"
Assert-PathExists -Path $EmbeddingModel -Label "Embedding model"

if ($Build) {
    & $cmake --build $BuildDir --config $Configuration --target llama-agent-daemon -j 1
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }
}

Assert-PathExists -Path $exePath -Label "Daemon executable"

$requests = @(
    '{"mode":"agent","prompt":"Say OK after making a tiny plan about remembering that the project codename is Maple.","session_id":"agent-learning-smoke","namespace_id":"agent-learning","project_id":"repo-smoke","memory_scope":"project","plan_scope":"project"}',
    '{"mode":"agent","prompt":"Say DONE after making a tiny plan about remembering that the daemon target is llama-agent-daemon.","session_id":"agent-learning-smoke","namespace_id":"agent-learning","project_id":"repo-smoke","memory_scope":"project","plan_scope":"project"}',
    '{"command":"shutdown"}'
)

$runId = [guid]::NewGuid().ToString("N")
$requestsPath = Join-Path $env:TEMP "llama-agent-daemon-agent-learning-smoke-$runId-requests.txt"
$stdoutPath = Join-Path $env:TEMP "llama-agent-daemon-agent-learning-smoke-$runId-stdout.log"
$stderrPath = Join-Path $env:TEMP "llama-agent-daemon-agent-learning-smoke-$runId-stderr.log"
Set-Content -LiteralPath $requestsPath -Value $requests -Encoding Ascii
Remove-Item -LiteralPath $stdoutPath -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $stderrPath -ErrorAction SilentlyContinue

try {
    $cmd = "type `"$requestsPath`" | `"$exePath`" --embedding-model `"$EmbeddingModel`" --model `"$ChatModel`" --default-mode agent --thinking-mode reflective --memory-learn post-turn --agent-plan auto -n 64 -ngl 0 1> `"$stdoutPath`" 2> `"$stderrPath`""
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
        throw "First agent learning turn failed: $($lines[1])"
    }
    if (-not $secondTurn.ok) {
        Show-Diagnostics -Path $stderrPath
        throw "Second agent learning turn failed: $($lines[2])"
    }
    if ($firstTurn.response -ne "OK") {
        Show-Diagnostics -Path $stderrPath
        throw "First agent learning response mismatch: $($lines[1])"
    }
    if ($secondTurn.response -ne "DONE") {
        Show-Diagnostics -Path $stderrPath
        throw "Second agent learning response mismatch: $($lines[2])"
    }
    if ($firstTurn.runtime_reused) {
        Show-Diagnostics -Path $stderrPath
        throw "First agent learning turn unexpectedly reused runtime: $($lines[1])"
    }
    if (-not $secondTurn.runtime_reused) {
        Show-Diagnostics -Path $stderrPath
        throw "Second agent learning turn did not reuse runtime: $($lines[2])"
    }
    if (-not $firstTurn.plan_id -or $firstTurn.plan_id -ne $secondTurn.plan_id) {
        Show-Diagnostics -Path $stderrPath
        throw "Agent learning session did not preserve plan_id across turns: $($lines[1]) / $($lines[2])"
    }
    if ([string]::IsNullOrWhiteSpace($firstTurn.memory_learning_summary) -or [string]::IsNullOrWhiteSpace($secondTurn.memory_learning_summary)) {
        Show-Diagnostics -Path $stderrPath
        throw "Agent learning turn did not report memory learning summary"
    }
    if ($firstTurn.memory_learning_summary -like "*candidate embedding failed safely*" -or
            $secondTurn.memory_learning_summary -like "*candidate embedding failed safely*") {
        Show-Diagnostics -Path $stderrPath
        throw "Agent learning daemon path still appears to be missing embedding support: $($lines[1]) / $($lines[2])"
    }
    if (-not $shutdown.ok -or $shutdown.event -ne "shutdown") {
        Show-Diagnostics -Path $stderrPath
        throw "Shutdown response mismatch: $($lines[3])"
    }

    Write-Host ""
    Write-Host "Agent daemon agent learning smoke test complete."
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
