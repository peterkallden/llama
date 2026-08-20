[CmdletBinding()]
param(
    [string]$BuildDir = "build-plan-cozo-ssl",
    [string]$Configuration = "Release",
    [string]$ChatModel = "C:\Users\kalld\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    [string]$EmbeddingModel = "C:\Users\kalld\models\nomic-embed-text-v1.5.Q4_K_M.gguf",
    [string]$OrdersCsv = "",
    [string]$CustomersCsv = "",
    [ValidateRange(1, 64)]
    [int]$Threads = 2,
    [switch]$Build
)

& (Join-Path $PSScriptRoot "test-qwen-nomic-agent-data.ps1") `
    -BuildDir $BuildDir `
    -Configuration $Configuration `
    -ChatModel $ChatModel `
    -EmbeddingModel $EmbeddingModel `
    -OrdersCsv $OrdersCsv `
    -CustomersCsv $CustomersCsv `
    -Threads $Threads `
    -ThinkingMode deliberate `
    -Build:$Build

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
