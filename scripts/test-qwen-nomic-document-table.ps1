[CmdletBinding()]
param(
    [string]$BuildDir = "build-agent-resource-tools-msvc-debug",
    [string]$Configuration = "Release",
    [string]$ChatModel = "models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    [string]$EmbeddingModel = "models\nomic-embed-text-v1.5.Q4_K_M.gguf",
    [string]$WorkSubdir = "work\qwen-nomic-document-table",
    [ValidateSet("reflective", "deliberate", "research")]
    [string]$ThinkingMode = "research",
    [uint32]$InferenceStepTimeoutMs = 60000,
    [ValidateRange(1, 8)]
    [int]$MaxToolRounds = 4,
    [ValidateRange(128, 2048)]
    [int]$NPredict = 256,
    [switch]$Build
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-PathExists {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path)) { throw "$Label not found: $Path" }
}

function Invoke-LoggedCommand {
    param([string]$Name, [string]$LogPath, [string]$FilePath, [string[]]$ArgumentList)
    $stdoutPath = "$LogPath.stdout"
    $stderrPath = "$LogPath.stderr"
    Remove-Item -LiteralPath $stdoutPath,$stderrPath -Force -ErrorAction SilentlyContinue
    New-Item -ItemType File -Force -Path $stdoutPath,$stderrPath | Out-Null
    & $FilePath @ArgumentList 1> $stdoutPath 2> $stderrPath
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

function Assert-LogDoesNotContain {
    param([string]$LogPath, [string[]]$Patterns)
    $text = Get-Content -LiteralPath $LogPath -Raw
    foreach ($pattern in $Patterns) {
        if ($text -match [regex]::Escape($pattern)) { throw "Smoke log contained unexpected text: $pattern" }
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$workBase = if ([System.IO.Path]::IsPathRooted($WorkSubdir)) {
    [System.IO.Path]::GetFullPath($WorkSubdir)
} else {
    Join-Path $repoRoot $WorkSubdir
}
$workDir = "{0}-{1}" -f $workBase, $stamp
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
    "--max-research-iterations", "1",
    "--max-plan-revisions", "1", "--max-tool-rounds", $MaxToolRounds.ToString(),
    "--resource-blob-backend", "fs", "--resource-blob-root", $resourceRoot,
    "--resource-metadata-backend", "in-memory",
    "--resource", $document, "--resource-mime-type", "application/json",
    "--memory-project", "qwen-nomic-document-table", "--plan-scope", "project",
    "--agent-trace", "--generation-trace", "--inference-step-timeout-ms", $InferenceStepTimeoutMs,
    "--plan-show-summary", "--prompt", $prompt,
    "-n", $NPredict.ToString(), "--context-size", "3072", "--threads", "4", "-ngl", "0")

Assert-LogContains $logPath @("document.tables", "document.table", "data.aggregate", "200")
Assert-LogContains $logPath @(
    "stage=tool kind=completed",
    "tool=document.tables",
    "tool=document.table",
    "tool=data.aggregate")
Assert-LogDoesNotContain $logPath @(
    "stage=tool kind=failed",
    "tool_call_limit_reached",
    "document representation is unavailable")
Write-Host "Qwen/Nomic document-table smoke passed. Log: $logPath"
