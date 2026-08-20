[CmdletBinding()]
param(
    [string]$BuildDir = "build-plan-cozo-ssl",
    [string]$Configuration = "Release",
    [string]$ChatModel = "$HOME\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    [string]$EmbeddingModel = "$HOME\models\nomic-embed-text-v1.5.Q4_K_M.gguf",
    [string]$WorkSubdir = "work\agent-runtime-driver-server-context-smoke",
    [string]$Prompt = "Say OK after making a tiny plan.",
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

        $process = Start-Process `
            -FilePath $FilePath `
            -ArgumentList $argumentString `
            -WorkingDirectory $repoRoot `
            -NoNewWindow `
            -PassThru `
            -Wait `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath

        $exitCode = $process.ExitCode
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
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

$cmake = Resolve-CMake
$exePath = Join-Path $repoRoot "$BuildDir\bin\$Configuration\llama-memory.exe"
$workDir = Join-Path $repoRoot $WorkSubdir
$memoryDb = Join-Path $workDir "memory.cozo"
$planDb = Join-Path $workDir "plan.cozo"
$logPath = Join-Path $workDir "runtime-driver-server-context-smoke.log"

Write-Host "Repo root: $repoRoot"
Write-Host "Build dir: $BuildDir"
Write-Host "Configuration: $Configuration"
Write-Host "Chat model: $ChatModel"
Write-Host "Embedding model: $EmbeddingModel"
Write-Host "Work dir: $workDir"

Assert-PathExists -Path $ChatModel -Label "Chat model"
Assert-PathExists -Path $EmbeddingModel -Label "Embedding model"

if ($Build) {
    & $cmake --build $BuildDir --config $Configuration --target llama-memory-poc -j 1
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }
}

Assert-PathExists -Path $exePath -Label "llama-memory executable"
New-Item -ItemType Directory -Force -Path $workDir | Out-Null

Invoke-LoggedCommand -Name "agent runtime driver server-context smoke" -LogPath $logPath -FilePath $exePath -ArgumentList @(
    "chat",
    "--memory-db", $memoryDb,
    "--plan-db", $planDb,
    "--model", $ChatModel,
    "--embedding-model", $EmbeddingModel,
    "--agent-profile", "safe",
    "--thinking-mode", "reflective",
    "--agent-bootstrap", "none",
    "--agent-inference-backend", "server-context",
    "--agent-trace",
    "--plan-show-summary",
    "--prompt", $Prompt,
    "-ngl", "0"
)

Write-Host ""
Write-Host "Smoke test complete."
Write-Host "Log written to: $logPath"
