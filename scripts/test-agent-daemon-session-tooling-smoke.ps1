[CmdletBinding()]
param(
    [string]$BuildDir = "build-plan-resident-debug",
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
$cliPath = Join-Path $repoRoot "$BuildDir\bin\$Configuration\llama-agent.exe"
$daemonPath = Join-Path $repoRoot "$BuildDir\bin\$Configuration\llama-agent-daemon.exe"
$fakeServerPath = Join-Path $repoRoot "$BuildDir\bin\$Configuration\llama-agent-mcp-stdio-fake-server.exe"

Write-Host "Repo root: $repoRoot"
Write-Host "Build dir: $BuildDir"
Write-Host "Configuration: $Configuration"
Write-Host "Chat model: $ChatModel"

Assert-PathExists -Path $ChatModel -Label "Chat model"

if ($Build) {
    & $cmake --build $BuildDir --config $Configuration --target llama-agent-cli llama-agent-daemon llama-agent-mcp-stdio-fake-server -j 1
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }
}

Assert-PathExists -Path $cliPath -Label "Agent CLI executable"
Assert-PathExists -Path $daemonPath -Label "Agent daemon executable"
Assert-PathExists -Path $fakeServerPath -Label "Fake MCP server executable"

$stdinLines = @(
    "Reply with DONE only."
)

$argumentList = @(
    "daemon-session",
    "--model", $ChatModel,
    "--prompt", "Reply with OK only.",
    "--agent-inference-backend", "server-context",
    "--memory-scope", "project",
    "--memory-namespace", "cli-session",
    "--memory-session", "daemon-session-tooling-smoke",
    "--memory-project", "repo-smoke",
    "--plan-scope", "project",
    "--tool-profile", "minimal",
    "--repository-root", $repoRoot,
    "--mcp-tool-command", $fakeServerPath,
    "--mcp-tool-server-name", "github",
    "--mcp-tool-prefix", "github",
    "--max-tool-rounds", "2",
    "--n-predict", "16",
    "-ngl", "0"
)

$runId = [guid]::NewGuid().ToString("N")
$stdinPath = Join-Path $env:TEMP "llama-agent-daemon-session-tooling-$runId-stdin.txt"
$stdoutPath = Join-Path $env:TEMP "llama-agent-daemon-session-tooling-$runId-stdout.log"
$stderrPath = Join-Path $env:TEMP "llama-agent-daemon-session-tooling-$runId-stderr.log"
Set-Content -LiteralPath $stdinPath -Value $stdinLines -Encoding Ascii
Remove-Item -LiteralPath $stdoutPath -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $stderrPath -ErrorAction SilentlyContinue

try {
    $quotedArgs = ($argumentList | ForEach-Object { "`"$_`"" }) -join " "
    $cmd = "type `"$stdinPath`" | `"$cliPath`" $quotedArgs 1> `"$stdoutPath`" 2> `"$stderrPath`""
    cmd /d /c $cmd | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Show-Diagnostics -Path $stderrPath
        throw "daemon-session tooling smoke failed with exit code $LASTEXITCODE"
    }

    $output = Get-Content -LiteralPath $stdoutPath
    $stdoutLines = @($output | Where-Object { $_ -and $_.Trim().Length -gt 0 })
    if ($stdoutLines.Count -ne 2) {
        Show-Diagnostics -Path $stderrPath
        throw "Expected exactly 2 stdout lines, got $($stdoutLines.Count)"
    }
    if ($stdoutLines[0].Trim() -ne "OK") {
        Show-Diagnostics -Path $stderrPath
        throw "First daemon-session tooling response mismatch: $($stdoutLines[0])"
    }
    if ($stdoutLines[1].Trim() -ne "DONE") {
        Show-Diagnostics -Path $stderrPath
        throw "Second daemon-session tooling response mismatch: $($stdoutLines[1])"
    }

    Write-Host ""
    Write-Host "Agent daemon-session tooling smoke test complete."
    Write-Host $stdoutLines[0]
    Write-Host $stdoutLines[1]
}
finally {
    Remove-Item -LiteralPath $stdinPath -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stdoutPath -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stderrPath -ErrorAction SilentlyContinue
}
