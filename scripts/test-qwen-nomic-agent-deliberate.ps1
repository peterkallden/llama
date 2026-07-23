[CmdletBinding()]
param(
    [string]$BuildDir = "build-plan-cozo-ssl",
    [string]$ChatModel = "C:\Users\kalld\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    [string]$EmbeddingModel = "C:\Users\kalld\models\nomic-embed-text-v1.5.Q4_K_M.gguf",
    [string]$WorkSubdir = "work\qwen-nomic-deliberate-test",
    [string]$AgentPrompt = "Make a short plan, inspect the available context, and then reply OK only.",
    [switch]$Build,
    [switch]$SkipAddSearch,
    [switch]$SkipStaticChat,
    [switch]$SkipAgentChat,
    [switch]$SkipLearningChat
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptPath = Join-Path $PSScriptRoot "test-qwen-nomic-agent.ps1"
& $scriptPath `
    -BuildDir $BuildDir `
    -ChatModel $ChatModel `
    -EmbeddingModel $EmbeddingModel `
    -WorkSubdir $WorkSubdir `
    -AgentPrompt $AgentPrompt `
    -ThinkingMode deliberate `
    -Build:$Build `
    -SkipAddSearch:$SkipAddSearch `
    -SkipStaticChat:$SkipStaticChat `
    -SkipAgentChat:$SkipAgentChat `
    -SkipLearningChat:$SkipLearningChat

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
