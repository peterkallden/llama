[CmdletBinding()]
param(
    [string]$BuildDir = "build-plan-cozo-ssl",
    [string]$ChatModel = "C:\Users\kalld\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    [string]$EmbeddingModel = "C:\Users\kalld\models\nomic-embed-text-v1.5.Q4_K_M.gguf",
    [string]$WorkSubdir = "work\qwen-nomic-multistep-test",
    [string]$ReuseWorkSubdir = "work\qwen-nomic-multistep-reuse-test",
    [string]$ExportPath = "work\qwen-nomic-multistep-export.json",
    [string]$LearningPromptOne = "Write a short reusable procedure for diagnosing agent regressions. Mention the observed failure, the repair path, and the reuse criterion.",
    [string]$LearningPromptTwo = "Refine the earlier debugging procedure for a neighboring issue. Keep it brief, but preserve the reusable steps.",
    [string]$ReusePrompt = "Use the imported procedure to solve a nearby agent regression and keep the answer short.",
    [switch]$Build
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

    Write-Host "Running $Name"
    $stdoutPath = "$LogPath.stdout"
    $stderrPath = "$LogPath.stderr"

    try {
        if (Test-Path -LiteralPath $stdoutPath) { Remove-Item -LiteralPath $stdoutPath -Force }
        if (Test-Path -LiteralPath $stderrPath) { Remove-Item -LiteralPath $stderrPath -Force }
        $argumentString = ($ArgumentList | ForEach-Object { Quote-Argument $_ }) -join ' '
        $command = ('"{0}" {1} 1> "{2}" 2> "{3}"' -f $FilePath, $argumentString, $stdoutPath, $stderrPath)

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

function Assert-LogContains {
    param(
        [string]$LogPath,
        [string[]]$Patterns,
        [string]$Label
    )

    $text = Get-Content -LiteralPath $LogPath -Raw
    foreach ($pattern in $Patterns) {
        if ($text -notmatch [regex]::Escape($pattern)) {
            throw "$Label missing expected text: $pattern"
        }
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

$runStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$cmake = Resolve-CMake
$exePath = Join-Path $repoRoot "$BuildDir\bin\Release\llama-memory.exe"
$workDir = Join-Path $repoRoot "$WorkSubdir-$runStamp"
$reuseWorkDir = Join-Path $repoRoot "$ReuseWorkSubdir-$runStamp"
$memoryDb = Join-Path $workDir "memory.cozo"
$planDb = Join-Path $workDir "plan.cozo"
$reuseMemoryDb = Join-Path $reuseWorkDir "memory.cozo"
$reusePlanDb = Join-Path $reuseWorkDir "plan.cozo"
$exportFile = Join-Path $repoRoot ("{0}-{1}{2}" -f [System.IO.Path]::GetFileNameWithoutExtension($ExportPath), $runStamp, [System.IO.Path]::GetExtension($ExportPath))

Write-Section "Configuration"
Write-Host "Repo root: $repoRoot"
Write-Host "Build dir: $BuildDir"
Write-Host "Chat model: $ChatModel"
Write-Host "Embedding model: $EmbeddingModel"
Write-Host "Work dir: $workDir"
Write-Host "Reuse dir: $reuseWorkDir"
Write-Host "Export: $exportFile"

Assert-PathExists -Path $ChatModel -Label "Chat model"
Assert-PathExists -Path $EmbeddingModel -Label "Embedding model"

if ($Build) {
    Write-Section "Build"
    & $cmake --build $BuildDir --config Release --target llama-memory-poc -j 1
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }
}

Assert-PathExists -Path $exePath -Label "llama-memory executable"

New-Item -ItemType Directory -Force -Path $workDir | Out-Null
New-Item -ItemType Directory -Force -Path $reuseWorkDir | Out-Null
if (Test-Path -LiteralPath $exportFile) {
    Remove-Item -LiteralPath $exportFile -Force
}

Write-Section "Memory Warmup"
Invoke-LoggedCommand -Name "memory add" -LogPath (Join-Path $workDir "01-add.log") -FilePath $exePath -ArgumentList @(
    "add",
    "--memory-db", $memoryDb,
    "--embedding-model", $EmbeddingModel,
    "--id", "note-1",
    "--kind", "fact",
    "--content", "Agent regressions should use a repeatable diagnosis procedure."
)

Invoke-LoggedCommand -Name "memory search" -LogPath (Join-Path $workDir "02-search.log") -FilePath $exePath -ArgumentList @(
    "search",
    "--memory-db", $memoryDb,
    "--embedding-model", $EmbeddingModel,
    "--query", "How should we debug agent regressions?"
)

Write-Section "Bootstrap Install"
$bootstrapLog = Join-Path $workDir "03-bootstrap.log"
Invoke-LoggedCommand -Name "bootstrap install" -LogPath $bootstrapLog -FilePath $exePath -ArgumentList @(
    "chat",
    "--memory-db", $memoryDb,
    "--plan-db", $planDb,
    "--model", $ChatModel,
    "--embedding-model", $EmbeddingModel,
    "--agent-profile", "learning",
    "--agent-bootstrap", "default",
    "--thinking-mode", "reflective",
    "--memory-project", "qwen-nomic-multistep",
    "--plan-scope", "project",
    "--agent-trace",
    "--plan-show-summary",
    "--prompt", "Install the default reusable procedures and blueprints for later export.",
    "-ngl", "0"
)
Assert-LogContains -LogPath $bootstrapLog -Label "Bootstrap install" -Patterns @(
    "agent bootstrap:"
)

Write-Section "Turn 1"
$turn1Log = Join-Path $workDir "04-learning-turn-1.log"
Invoke-LoggedCommand -Name "learning turn 1" -LogPath $turn1Log -FilePath $exePath -ArgumentList @(
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
    "--memory-project", "qwen-nomic-multistep",
    "--plan-scope", "project",
    "--agent-trace",
    "--plan-show-summary",
    "--prompt", $LearningPromptOne,
    "-ngl", "0"
)
Assert-LogContains -LogPath $turn1Log -Label "Turn 1" -Patterns @(
    "reflection completed",
    "memory_learn summary="
)

Write-Section "Turn 2"
$turn2Log = Join-Path $workDir "05-learning-turn-2.log"
Invoke-LoggedCommand -Name "learning turn 2" -LogPath $turn2Log -FilePath $exePath -ArgumentList @(
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
    "--memory-project", "qwen-nomic-multistep",
    "--plan-scope", "project",
    "--agent-trace",
    "--plan-show-summary",
    "--prompt", $LearningPromptTwo,
    "-ngl", "0"
)
Assert-LogContains -LogPath $turn2Log -Label "Turn 2" -Patterns @(
    "memory_learn summary="
)

Write-Section "Export"
$exportLog = Join-Path $workDir "06-export.log"
Invoke-LoggedCommand -Name "agent export" -LogPath $exportLog -FilePath $exePath -ArgumentList @(
    "chat",
    "--memory-db", $memoryDb,
    "--plan-db", $planDb,
    "--model", $ChatModel,
    "--embedding-model", $EmbeddingModel,
    "--agent-profile", "safe",
    "--agent-bootstrap", "none",
    "--thinking-mode", "reflective",
    "--agent-export", $exportFile,
    "--memory-project", "qwen-nomic-multistep",
    "--plan-scope", "project",
    "--prompt", "Export the reusable procedures and blueprints for later reuse.",
    "-ngl", "0"
)
Assert-PathExists -Path $exportFile -Label "Export package"
Assert-LogContains -LogPath $exportLog -Label "Export" -Patterns @(
    "agent export written"
)
$exportJson = Get-Content -LiteralPath $exportFile -Raw | ConvertFrom-Json
if (($exportJson.procedures | Measure-Object).Count -lt 1 -or ($exportJson.blueprints | Measure-Object).Count -lt 1) {
    throw "Export package is still empty; bootstrap install did not populate procedures and blueprints"
}

Write-Section "Import Reuse"
Invoke-LoggedCommand -Name "memory add reuse" -LogPath (Join-Path $reuseWorkDir "01-add.log") -FilePath $exePath -ArgumentList @(
    "add",
    "--memory-db", $reuseMemoryDb,
    "--embedding-model", $EmbeddingModel,
    "--id", "note-1",
    "--kind", "fact",
    "--content", "The reuse run should prove that the exported package can bootstrap a fresh session."
)

Invoke-LoggedCommand -Name "memory search reuse" -LogPath (Join-Path $reuseWorkDir "02-search.log") -FilePath $exePath -ArgumentList @(
    "search",
    "--memory-db", $reuseMemoryDb,
    "--embedding-model", $EmbeddingModel,
    "--query", "What should the reuse run prove?"
)

$reuseLog = Join-Path $reuseWorkDir "03-reuse-chat.log"
Invoke-LoggedCommand -Name "reuse chat" -LogPath $reuseLog -FilePath $exePath -ArgumentList @(
    "chat",
    "--memory-db", $reuseMemoryDb,
    "--plan-db", $reusePlanDb,
    "--model", $ChatModel,
    "--embedding-model", $EmbeddingModel,
    "--agent-profile", "safe",
    "--agent-bootstrap", "none",
    "--agent-import", $exportFile,
    "--thinking-mode", "reflective",
    "--memory-learn", "off",
    "--memory-project", "qwen-nomic-reuse",
    "--plan-scope", "project",
    "--agent-trace",
    "--plan-show-summary",
    "--prompt", $ReusePrompt,
    "-ngl", "0"
)
Assert-LogContains -LogPath $reuseLog -Label "Reuse" -Patterns @(
    "agent bootstrap:",
    "agent blueprint auto-selected:",
    "plan: id="
)

Write-Section "Done"
Write-Host "Logs written under: $workDir"
Write-Host "Reuse logs written under: $reuseWorkDir"
Write-Host "Export written to: $exportFile"
