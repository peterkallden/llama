[CmdletBinding()]
param(
    [string]$BuildDir = "build-plan-cozo-ssl",
    [string]$Configuration = "Release",
    [string]$ChatModel = "C:\Users\kalld\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    [string]$EmbeddingModel = "C:\Users\kalld\models\nomic-embed-text-v1.5.Q4_K_M.gguf",
    [string]$OrdersCsv = "",
    [string]$CustomersCsv = "",
    [ValidateSet("reflective", "deliberate")]
    [string]$ThinkingMode = "reflective",
    [string]$WorkSubdir = "work\qwen-nomic-data-$ThinkingMode",
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
    # cmd.exe treats literal newlines in a /c command as command separators.
    # Normalize multi-line prompts before constructing the redirected command.
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
    if (Test-Path -LiteralPath $stdoutPath) { Remove-Item -LiteralPath $stdoutPath -Force }
    if (Test-Path -LiteralPath $stderrPath) { Remove-Item -LiteralPath $stderrPath -Force }
    New-Item -ItemType File -Force -Path $stdoutPath,$stderrPath | Out-Null
    $arguments = ($ArgumentList | ForEach-Object { Quote-Argument $_ }) -join ' '
    $toolPath = 'E:\progs\bin;C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem;C:\Program Files\Git\cmd;C:\Program Files\nodejs;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin'
    $command = ('"{0}" {1} 1> "{2}" 2> "{3}"' -f $FilePath, $arguments, $stdoutPath, $stderrPath)
    & cmd.exe /d /c ('set "Path=" & set PATH={0} & {1}' -f $toolPath, $command)
    $exitCode = $LASTEXITCODE
    $lines = @()
    if (Test-Path -LiteralPath $stdoutPath) { $lines += Get-Content -LiteralPath $stdoutPath }
    if (Test-Path -LiteralPath $stderrPath) { $lines += Get-Content -LiteralPath $stderrPath }
    if (-not (Test-Path -LiteralPath $stdoutPath)) { New-Item -ItemType File -Force -Path $stdoutPath | Out-Null }
    if (-not (Test-Path -LiteralPath $stderrPath)) { New-Item -ItemType File -Force -Path $stderrPath | Out-Null }
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

function Get-SeededCount {
    param([string]$LogPath, [string]$Name)
    $text = Get-Content -LiteralPath $LogPath -Raw
    $match = [regex]::Match($text, [regex]::Escape($Name) + '=([0-9]+)')
    if (-not $match.Success -or [int]$match.Groups[1].Value -lt 1) {
        throw "Cozo seed did not report a positive $Name count"
    }
    return [int]$match.Groups[1].Value
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$runStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$workDir = Join-Path $repoRoot ("{0}-{1}" -f $WorkSubdir, $runStamp)
$datasetsDir = Join-Path $workDir "datasets"
$dataDb = Join-Path $workDir "data.cozo"
$memoryDb = Join-Path $workDir "memory.cozo"
$planDb = Join-Path $workDir "plan.cozo"
$ordersPath = if ($OrdersCsv) { (Resolve-Path -LiteralPath $OrdersCsv).Path } else { Join-Path $datasetsDir "orders.csv" }
$customersPath = if ($CustomersCsv) { (Resolve-Path -LiteralPath $CustomersCsv).Path } else { Join-Path $datasetsDir "customers.csv" }
$buildRoot = if ([System.IO.Path]::IsPathRooted($BuildDir)) {
    [System.IO.Path]::GetFullPath($BuildDir)
} else {
    Join-Path $repoRoot $BuildDir
}
$agentExe = Join-Path $buildRoot "bin\$Configuration\llama-agent.exe"
$seedExe = Join-Path $buildRoot "bin\$Configuration\llama-agent-data-store-cozo-seed.exe"
if (-not (Test-Path -LiteralPath $agentExe)) {
    $agentExe = Join-Path $buildRoot "bin\llama-agent.exe"
}
if (-not (Test-Path -LiteralPath $seedExe)) {
    $seedExe = Join-Path $buildRoot "bin\llama-agent-data-store-cozo-seed.exe"
}
$seedLog = Join-Path $workDir "01-seed.log"
$agentLog = Join-Path $workDir "02-agent-data.log"

Assert-PathExists -Path $ChatModel -Label "Chat model"
Assert-PathExists -Path $EmbeddingModel -Label "Embedding model"
New-Item -ItemType Directory -Force -Path $datasetsDir | Out-Null

if (-not $OrdersCsv) {
    @("order_id,customer_id,amount", "1,10,12", "2,10,8", "3,11,20") | Set-Content -Encoding ascii -LiteralPath $ordersPath
} else { Assert-PathExists -Path $ordersPath -Label "Orders CSV" }
if (-not $CustomersCsv) {
    @("customer_id,segment", "10,enterprise", "11,consumer") | Set-Content -Encoding ascii -LiteralPath $customersPath
} else { Assert-PathExists -Path $customersPath -Label "Customers CSV" }

if ($Build) {
    & cmake --build $BuildDir --config $Configuration --target llama-agent-cli --parallel 1
    if ($LASTEXITCODE -ne 0) { throw "llama-agent build failed with exit code $LASTEXITCODE" }
    & cmake --build $BuildDir --config $Configuration --target llama-agent-data-store-cozo-seed --parallel 1
    if ($LASTEXITCODE -ne 0) { throw "Cozo seed build failed with exit code $LASTEXITCODE" }
}
Assert-PathExists -Path $agentExe -Label "llama-agent executable"
Assert-PathExists -Path $seedExe -Label "Cozo seed executable"

Invoke-LoggedCommand -Name "Cozo CSV seed" -LogPath $seedLog -FilePath $seedExe -ArgumentList @(
    "--db", $dataDb, "--orders", $ordersPath, "--customers", $customersPath)
$seededOrders = Get-SeededCount -LogPath $seedLog -Name "seeded_orders"
$seededCustomers = Get-SeededCount -LogPath $seedLog -Name "seeded_customers"

$prompt = @"
Perform a bounded research/data-analysis task using the datasets in the workspace.
Discover the two CSV datasets, inspect them, join orders with customers on customer_id,
then aggregate the joined rows by segment using sum(amount). Also use statistics.describe
on amount. Report the exact totals and mention the tools/results you used. Do not guess;
use the data tools.
"@

Invoke-LoggedCommand -Name "Qwen/Nomic $ThinkingMode data research" -LogPath $agentLog -FilePath $agentExe -ArgumentList @(
    "chat", "--backend", "cozo", "--memory-db", $memoryDb,
    "--plan-backend", "cozo", "--plan-db", $planDb,
    "--data-backend", "cozo", "--data-db", $dataDb,
    "--model", $ChatModel, "--embedding-model", $EmbeddingModel,
    "--agent-profile", "research", "--tool-profile", "all-configured",
    "--thinking-mode", $ThinkingMode, "--max-reflection-rounds", "1",
    "--max-research-iterations", "1", "--agent-plan", "auto",
    "--max-plan-revisions", "2",
    "--repository-root", $workDir, "--max-tool-rounds", "4",
    "--memory-project", "qwen-nomic-data", "--plan-scope", "project",
    "--agent-trace", "--plan-show-summary", "--prompt", $prompt, "-n", "96",
    "--threads", "2", "-ngl", "0")

$orders = Import-Csv -LiteralPath $ordersPath
$expectedTotal = ($orders | ForEach-Object { [double]$_.amount } | Measure-Object -Sum).Sum
$expectedTotalText = $expectedTotal.ToString([System.Globalization.CultureInfo]::InvariantCulture)
Assert-LogContains -LogPath $agentLog -Patterns @("data.join", "data.aggregate", $expectedTotalText)
Write-Host "Qwen/Nomic CSV data smoke passed for thinking mode: $ThinkingMode (orders=$seededOrders customers=$seededCustomers sum=$expectedTotalText)"
Write-Host "Logs written under: $workDir"
