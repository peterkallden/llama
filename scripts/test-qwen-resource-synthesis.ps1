[CmdletBinding()]
param(
    [string]$BuildDir = "build-plan-cozo-ssl",
    [string]$ChatModel = "C:\Users\kalld\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    [string]$Fixture = "tests\data\agent-resource-synthesis.txt",
    [string]$WorkSubdir = "work\qwen-resource-synthesis",
    [ValidateRange(1, 64)]
    [int]$Threads = 2
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

$resolvedBuildDir = if ([System.IO.Path]::IsPathRooted($BuildDir)) {
    [System.IO.Path]::GetFullPath($BuildDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $BuildDir))
}
$fixturePath = if ([System.IO.Path]::IsPathRooted($Fixture)) {
    [System.IO.Path]::GetFullPath($Fixture)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Fixture))
}
$exePath = Join-Path $resolvedBuildDir "bin\Release\llama-agent.exe"
$workDir = Join-Path $repoRoot $WorkSubdir
$logPath = Join-Path $workDir "resource-synthesis.log"

foreach ($required in @(
        @{ Path = $ChatModel; Label = "Chat model" },
        @{ Path = $fixturePath; Label = "Fixture" },
        @{ Path = $exePath; Label = "llama-agent executable" })) {
    if (-not (Test-Path -LiteralPath $required.Path)) {
        throw "$($required.Label) not found: $($required.Path)"
    }
}

New-Item -ItemType Directory -Force -Path $workDir | Out-Null
$paragraphs = (Get-Content -LiteralPath $fixturePath -Raw) -split "\r?\n\r?\n" |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

if ($paragraphs.Count -ne 4) {
    throw "Expected exactly four fixed synthesis chunks, got $($paragraphs.Count)"
}

$observations = for ($index = 0; $index -lt $paragraphs.Count; ++$index) {
    "[chunk $($index + 1)/$($paragraphs.Count) parent=agent-resource-synthesis]`n$($paragraphs[$index].Trim())"
}

$prompt = @"
Synthesize the following four resource observations into one concise answer.
Preserve every factual requirement. Include all three binary variants, the test
gate, the Agent CI package requirement, and the three separate test result
categories. Do not invent facts and do not mention chunking.

$($observations -join "`n`n")
"@

Write-Host "Build dir: $resolvedBuildDir"
Write-Host "Fixture: $fixturePath"
Write-Host "Chunks: $($observations.Count)"
Write-Host "Running Qwen resource synthesis"

$arguments = @(
    "chat",
    "--model", $ChatModel,
    "--agent-profile", "static",
    "--agent-bootstrap", "none",
    "--prompt", $prompt,
    "-t", $Threads.ToString(),
    "-ngl", "0"
)

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    $output = @(& $exePath @arguments 2>&1)
} finally {
    $ErrorActionPreference = $previousErrorActionPreference
}
$exitCode = $LASTEXITCODE
$output | Set-Content -LiteralPath $logPath

if ($exitCode -ne 0) {
    throw "Qwen resource synthesis failed with exit code $exitCode. See $logPath"
}

$answer = ($output -join [Environment]::NewLine)
$requiredFactPatterns = @("CPU", "CUDA", "Vulkan", "Agent CI", "passed", "failed", "not[- ]run")
$missingFacts = @($requiredFactPatterns | Where-Object { $answer -notmatch $_ })
if ($missingFacts.Count -gt 0) {
    throw "Qwen synthesis omitted required facts: $($missingFacts -join ', '). See $logPath"
}

Write-Host "resource_chunks=4"
Write-Host "resource_synthesis=passed"
Write-Host "Log: $logPath"
