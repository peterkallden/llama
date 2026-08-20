[CmdletBinding()]
param(
    [string]$BuildDir = "build-plan-cozo-ssl",
    [string]$ChatModel = "C:\Users\kalld\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    [string]$EmbeddingModel = "C:\Users\kalld\models\nomic-embed-text-v1.5.Q4_K_M.gguf",
    [string]$WorkSubdir = "work\qwen-nomic-manual-test",
    [string]$LearningPrompt = "When debugging agent regressions, explain the reusable procedure you followed and keep the final answer brief.",
    [ValidateSet("reflective", "deliberate")]
    [string]$ThinkingMode = "reflective",
    [string]$AgentPrompt = "Say OK after making a tiny plan.",
    [ValidateRange(1, 64)]
    [int]$Threads = 2,
    [switch]$Build,
    [switch]$SkipAddSearch,
    [switch]$SkipStaticChat,
    [switch]$SkipAgentChat,
    [switch]$SkipLearningChat
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "== $Title =="
}

function Assert-PathExists {
    param(
        [string]$Path,
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label not found: $Path"
    }
}

function Invoke-LoggedCommand {
    param(
        [string]$Name,
        [string]$LogPath,
        [string]$FilePath,
        [string[]]$ArgumentList
    )

    Write-Host "Running $Name"
    $stdoutPath = "$LogPath.stdout"
    $stderrPath = "$LogPath.stderr"

    try {
        if (Test-Path -LiteralPath $stdoutPath) { Remove-Item -LiteralPath $stdoutPath -Force }
        if (Test-Path -LiteralPath $stderrPath) { Remove-Item -LiteralPath $stderrPath -Force }
        $argumentString = ($ArgumentList | ForEach-Object { Quote-Argument $_ }) -join ' '
        $command = ('"{0}" {1} 1> "{2}" 2> "{3}" < NUL' -f $FilePath, $argumentString, $stdoutPath, $stderrPath)

        Push-Location -LiteralPath $repoRoot
        try {
            & cmd.exe /d /c $command
            $exitCode = $LASTEXITCODE
        } finally {
            Pop-Location
        }
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

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

$cmake = Resolve-CMake
$resolvedBuildDir = if ([System.IO.Path]::IsPathRooted($BuildDir)) {
    [System.IO.Path]::GetFullPath($BuildDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $BuildDir))
}
$memoryExePath = Join-Path $resolvedBuildDir "bin\Release\llama-memory.exe"
$agentExePath = Join-Path $resolvedBuildDir "bin\Release\llama-agent.exe"
if (-not (Test-Path -LiteralPath $memoryExePath)) {
    $memoryExePath = Join-Path $resolvedBuildDir "bin\llama-memory.exe"
}
if (-not (Test-Path -LiteralPath $agentExePath)) {
    $agentExePath = Join-Path $resolvedBuildDir "bin\llama-agent.exe"
}
$workDir = if ([System.IO.Path]::IsPathRooted($WorkSubdir)) {
    [System.IO.Path]::GetFullPath($WorkSubdir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $WorkSubdir))
}
$memoryDb = Join-Path $workDir "memory.cozo"
$planDb = Join-Path $workDir "plan.cozo"

Write-Section "Configuration"
Write-Host "Repo root: $repoRoot"
Write-Host "Build dir: $resolvedBuildDir"
Write-Host "Chat model: $ChatModel"
Write-Host "Embedding model: $EmbeddingModel"
Write-Host "Thinking mode: $ThinkingMode"
Write-Host "Inference threads: $Threads"
Write-Host "Work dir: $workDir"

Assert-PathExists -Path $ChatModel -Label "Chat model"
Assert-PathExists -Path $EmbeddingModel -Label "Embedding model"

if ($Build) {
    Write-Section "Build"
    & $cmake --build $resolvedBuildDir --config Release --target llama-memory-poc llama-agent-cli -j 1
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }
}

Assert-PathExists -Path $memoryExePath -Label "llama-memory executable"
Assert-PathExists -Path $agentExePath -Label "llama-agent executable"

New-Item -ItemType Directory -Force -Path $workDir | Out-Null

if (-not $SkipAddSearch) {
    Write-Section "Embedding Add/Search"
    $addLog = Join-Path $workDir "01-add.log"
    $searchLog = Join-Path $workDir "02-search.log"

    Invoke-LoggedCommand -Name "memory add" -LogPath $addLog -FilePath $memoryExePath -ArgumentList @(
        "add",
        "--memory-db", $memoryDb,
        "--embedding-model", $EmbeddingModel,
        "--id", "note-1",
        "--kind", "fact",
        "--content", "Agent regressions should use the regression diagnosis procedure."
    )

    Invoke-LoggedCommand -Name "memory search" -LogPath $searchLog -FilePath $memoryExePath -ArgumentList @(
        "search",
        "--memory-db", $memoryDb,
        "--embedding-model", $EmbeddingModel,
        "--query", "How should we handle agent regressions?"
    )
}

if (-not $SkipStaticChat) {
    Write-Section "Static Chat"
    $staticLog = Join-Path $workDir "03-static-chat.log"

    Invoke-LoggedCommand -Name "static chat" -LogPath $staticLog -FilePath $agentExePath -ArgumentList @(
        "chat",
        "--memory-db", $memoryDb,
        "--model", $ChatModel,
        "--agent-profile", "static",
        "--agent-bootstrap", "none",
        "--prompt", "Say OK.",
        "-t", $Threads.ToString(),
        "-ngl", "0"
    )
}

if (-not $SkipAgentChat) {
    Write-Section "Agent Chat"
    $agentLog = Join-Path $workDir "04-agent-chat.log"

    Invoke-LoggedCommand -Name "agent chat" -LogPath $agentLog -FilePath $agentExePath -ArgumentList @(
        "chat",
        "--memory-db", $memoryDb,
        "--plan-db", $planDb,
        "--model", $ChatModel,
        "--embedding-model", $EmbeddingModel,
        "--agent-profile", "safe",
        "--agent-bootstrap", "none",
        "--thinking-mode", $ThinkingMode,
        "--agent-trace",
        "--plan-show-summary",
        "--prompt", $AgentPrompt,
        "-t", $Threads.ToString(),
        "-ngl", "0"
    )
}

if (-not $SkipLearningChat) {
    Write-Section "Reflection And Learning Chat"
    $learningLog = Join-Path $workDir "05-learning-chat.log"

    Invoke-LoggedCommand -Name "learning chat" -LogPath $learningLog -FilePath $agentExePath -ArgumentList @(
        "chat",
        "--memory-db", $memoryDb,
        "--plan-db", $planDb,
        "--model", $ChatModel,
        "--embedding-model", $EmbeddingModel,
        "--agent-profile", "learning",
        "--agent-bootstrap", "none",
        "--thinking-mode", "reflective",
        "--memory-learn", "post-turn",
        "--memory-learn-show-candidate",
        "--memory-project", "qwen-nomic-script",
        "--plan-scope", "project",
        "--agent-trace",
        "--plan-show-summary",
        "--prompt", $LearningPrompt,
        "-t", $Threads.ToString(),
        "-ngl", "0"
    )
}

Write-Section "Done"
Write-Host "Logs written under: $workDir"
Write-Host "Suggested files to inspect:"
Write-Host "  $(Join-Path $workDir '01-add.log')"
Write-Host "  $(Join-Path $workDir '02-search.log')"
Write-Host "  $(Join-Path $workDir '03-static-chat.log')"
Write-Host "  $(Join-Path $workDir '04-agent-chat.log')"
Write-Host "  $(Join-Path $workDir '05-learning-chat.log')"
Write-Host "Agent chat thinking mode: $ThinkingMode"
