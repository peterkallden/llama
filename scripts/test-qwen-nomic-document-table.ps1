[CmdletBinding()]
param(
    [string]$BuildDir = "C:\llama-builds\agent-resource-tools-msvc-debug-cuda12b",
    [string]$Configuration = "Release",
    [string]$ChatModel = "C:\Users\kalld\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    [string]$EmbeddingModel = "C:\Users\kalld\models\nomic-embed-text-v1.5.Q4_K_M.gguf",
    [string]$WorkSubdir = "work\qwen-nomic-document-table",
    [ValidateSet("reflective", "research")]
    [string]$ThinkingMode = "research",
    [switch]$Build
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-PathExists {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path)) { throw "$Label not found: $Path" }
}

function Quote-Argument {
    param([string]$Value)
    $Value = $Value -replace '[\r\n]+', ' '
    if ($Value -notmatch '[\s"]') { return $Value }
    $escaped = $Value -replace '(\\*)"', '$1$1\"'
    $escaped = $escaped -replace '(\\+)$', '$1$1'
    return '"' + $escaped + '"'
}

function Invoke-LoggedCommand {
    param([string]$Name, [string]$LogPath, [string]$FilePath, [string[]]$ArgumentList)
    $stdoutPath = "$LogPath.stdout"
    $stderrPath = "$LogPath.stderr"
    Remove-Item -LiteralPath $stdoutPath,$stderrPath -Force -ErrorAction SilentlyContinue
    New-Item -ItemType File -Force -Path $stdoutPath,$stderrPath | Out-Null
    $arguments = ($ArgumentList | ForEach-Object { Quote-Argument $_ }) -join ' '
    $toolPath = 'C:\tools;C:\tools\LLVM\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;C:\Users\kalld\Documents\Codex\llama-dyn\work\cozo-release\win;C:\Windows\System32;C:\Windows;C:\Program Files\Git\cmd;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
    $command = ('"{0}" {1} 1> "{2}" 2> "{3}"' -f $FilePath, $arguments, $stdoutPath, $stderrPath)
    & cmd.exe /d /c ('set "Path=" & set PATH={0} & {1}' -f $toolPath, $command)
    $exitCode = $LASTEXITCODE
    $lines = @(Get-Content -LiteralPath $stdoutPath) + @(Get-Content -LiteralPath $stderrPath)
    [System.IO.File]::WriteAllText($LogPath, ($lines -join [Environment]::NewLine))
    $lines | ForEach-Object { Write-Host $_ }
    if ($exitCode -ne 0) { throw "$Name failed with exit code $exitCode. See $LogPath" }
}

function Assert-LogContains {
    param([string]$LogPath, [string[]]$Patterns)
    $text = Get-Content -LiteralPath $LogPath -Raw
    foreach ($pattern in $Patterns) {
        if ($text -notmatch [regex]::Escape($pattern)) { throw "Smoke log missing expected text: $pattern" }
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$workDir = Join-Path $repoRoot ("{0}-{1}" -f $WorkSubdir, $stamp)
$resourceRoot = Join-Path $workDir "resources"
$dataDb = Join-Path $workDir "data.cozo"
$document = Join-Path $repoRoot "pocs\agent\smoke\data\fixtures\document-table\document-table-model.json"
$buildRoot = if ([System.IO.Path]::IsPathRooted($BuildDir)) {
    [System.IO.Path]::GetFullPath($BuildDir)
} else {
    Join-Path $repoRoot $BuildDir
}
$agentExe = Join-Path $buildRoot "bin\$Configuration\llama-agent.exe"
if (-not (Test-Path -LiteralPath $agentExe)) {
    $agentExe = Join-Path $buildRoot "bin\llama-agent.exe"
}
$logPath = Join-Path $workDir "document-table-agent.log"

Assert-PathExists $ChatModel "Chat model"
Assert-PathExists $EmbeddingModel "Embedding model"
Assert-PathExists $document "Document fixture"
New-Item -ItemType Directory -Force -Path $resourceRoot | Out-Null

if ($Build) {
    & cmake --build $BuildDir --config $Configuration --target llama-agent-cli --parallel 1
    if ($LASTEXITCODE -ne 0) { throw "llama-agent build failed with exit code $LASTEXITCODE" }
}
Assert-PathExists $agentExe "llama-agent executable"

$prompt = @"
Use the attached JSON document representation. First call document.tables for the
attached resource. Then call document.table using the unique table name "Budget
summary". Finally call data.aggregate on the returned dataset and calculate the
sum of the amount column. Report the table name, dataset reference and total.
Use the tools and do not guess. The expected total is 200.
"@

Invoke-LoggedCommand -Name "Qwen/Nomic document table" -LogPath $logPath -FilePath $agentExe -ArgumentList @(
    "run", "--backend", "in-memory",
    "--plan-backend", "in-memory",
    "--data-backend", "cozo", "--data-db", $dataDb,
    "--model", $ChatModel, "--embedding-model", $EmbeddingModel,
    "--agent-profile", "research", "--tool-profile", "research",
    "--thinking-mode", $ThinkingMode, "--max-reflection-rounds", "1",
    "--max-research-iterations", "1", "--agent-plan", "auto",
    "--max-plan-revisions", "1", "--max-tool-rounds", "1",
    "--resource-blob-backend", "fs", "--resource-blob-root", $resourceRoot,
    "--resource-metadata-backend", "in-memory",
    "--resource", $document, "--resource-mime-type", "application/json",
    "--memory-project", "qwen-nomic-document-table", "--plan-scope", "project",
    "--agent-trace", "--plan-show-summary", "--prompt", $prompt,
    "-n", "256", "--context-size", "2048", "--threads", "4", "-ngl", "0")

Assert-LogContains $logPath @("document.tables", "document.table", "data.aggregate", "200")
Write-Host "Qwen/Nomic document-table smoke passed. Log: $logPath"
