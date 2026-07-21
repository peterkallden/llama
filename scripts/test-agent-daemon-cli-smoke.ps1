[CmdletBinding()]
param(
    [string]$BuildDir = "build-plan",
    [string]$Configuration = "Release",
    [string]$ChatModel = "$HOME\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    [string]$Prompt = "Reply with OK only.",
    [string]$ExpectedResponse = "OK",
    [string]$WorkSubdir = "work\agent-daemon-cli-smoke",
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

function Quote-Argument {
    param([string]$Value)

    if ($null -eq $Value) {
        return '""'
    }

    if ($Value -notmatch '[\s"]') {
        return $Value
    }

    $escaped = $Value -replace '(\\*)"', '$1$1\"'
    $escaped = $escaped -replace '(\\+)$', '$1$1'
    return '"' + $escaped + '"'
}

function Invoke-LoggedCommand {
    param(
        [string]$Name,
        [string]$LogPath,
        [string]$FilePath,
        [string[]]$ArgumentList
    )

    $stdoutPath = "$LogPath.stdout"
    $stderrPath = "$LogPath.stderr"

    try {
        if (Test-Path -LiteralPath $stdoutPath) { Remove-Item -LiteralPath $stdoutPath -Force }
        if (Test-Path -LiteralPath $stderrPath) { Remove-Item -LiteralPath $stderrPath -Force }
        $argumentString = ($ArgumentList | ForEach-Object { Quote-Argument $_ }) -join ' '

        $cmd = "`"$FilePath`" $argumentString 1> `"$stdoutPath`" 2> `"$stderrPath`""
        cmd /d /c $cmd | Out-Null
        $exitCode = $LASTEXITCODE
    } finally {
        $lines = @()
        if (Test-Path -LiteralPath $stdoutPath) { $lines += Get-Content -LiteralPath $stdoutPath }
        if (Test-Path -LiteralPath $stderrPath) { $lines += Get-Content -LiteralPath $stderrPath }
        $lines | Set-Content -LiteralPath $LogPath
        $lines | ForEach-Object { Write-Host $_ }
    }

    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode. See $LogPath"
    }

    return @{
        StdoutPath = $stdoutPath
        StderrPath = $stderrPath
        ExitCode = $exitCode
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

$cmake = Resolve-CMake
$cliPath = Join-Path $repoRoot "$BuildDir\bin\$Configuration\llama-agent.exe"
$daemonPath = Join-Path $repoRoot "$BuildDir\bin\$Configuration\llama-agent-daemon.exe"
$workDir = Join-Path $repoRoot $WorkSubdir
$logPath = Join-Path $workDir "agent-daemon-cli-smoke.log"

Write-Host "Repo root: $repoRoot"
Write-Host "Build dir: $BuildDir"
Write-Host "Configuration: $Configuration"
Write-Host "Chat model: $ChatModel"
Write-Host "Work dir: $workDir"

Assert-PathExists -Path $ChatModel -Label "Chat model"

if ($Build) {
    & $cmake --build $BuildDir --config $Configuration --target llama-agent-cli llama-agent-daemon -j 1
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }
}

Assert-PathExists -Path $cliPath -Label "Agent CLI executable"
Assert-PathExists -Path $daemonPath -Label "Agent daemon executable"
New-Item -ItemType Directory -Force -Path $workDir | Out-Null

$run = Invoke-LoggedCommand -Name "agent daemon cli smoke" -LogPath $logPath -FilePath $cliPath -ArgumentList @(
    "daemon-chat",
    "--model", $ChatModel,
    "--prompt", $Prompt,
    "--agent-inference-backend", "server-context",
    "--memory-scope", "session",
    "--memory-namespace", "cli-smoke",
    "--memory-session", "daemon-cli-smoke",
    "--n-predict", "16",
    "-ngl", "0"
)

$stdoutLines = @()
if (Test-Path -LiteralPath $run.StdoutPath) {
    $stdoutLines = @(Get-Content -LiteralPath $run.StdoutPath | Where-Object { $_ -and $_.Trim().Length -gt 0 })
}

if ($stdoutLines.Count -lt 1) {
    throw "No stdout response captured. See $logPath"
}

$firstLine = $stdoutLines[0].Trim()
if ($firstLine -ne $ExpectedResponse) {
    throw "Unexpected daemon-chat response '$firstLine' (expected '$ExpectedResponse'). See $logPath"
}

Write-Host ""
Write-Host "Daemon CLI smoke test complete."
Write-Host "First response: $firstLine"
Write-Host "Log written to: $logPath"
