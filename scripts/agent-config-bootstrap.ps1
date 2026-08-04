[CmdletBinding()]
param(
    [string]$Output = 'agent-daemon-config.json',
    [string]$Model = 'models/model.gguf',
    [string]$EmbeddingModel = '',
    [string]$CozoRoot = 'data',
    [string]$RepositoryRoot = '.',
    [ValidateSet('minimal', 'research', 'developer-read', 'all-configured')]
    [string]$ToolProfile = 'all-configured',
    [ValidateRange(1, 1048576)] [int]$Threads = 4,
    [ValidateRange(0, 1048576)] [int]$GpuLayers = 0,
    [ValidateRange(1, 1048576)] [int]$WorkerCount = 2,
    [ValidateRange(1, 1048576)] [int]$QueueCapacity = 8,
    [ValidateRange(1, 1048576)] [int]$InferenceMaxActive = 1,
    [ValidateSet('chat', 'agent')] [string]$DefaultMode = 'agent',
    [ValidateSet('auto', 'reflective', 'deliberate', 'research')]
    [string]$ThinkingMode = 'auto',
    [ValidateSet('none', 'docker', 'kubernetes')] [string]$Sandbox = 'none',
    [string]$ProvidersFile
)

$providers = @()
if ($ProvidersFile) {
    if (-not (Test-Path -LiteralPath $ProvidersFile -PathType Leaf)) {
        throw "Provider file not found: $ProvidersFile"
    }
    $providers = @(Get-Content -LiteralPath $ProvidersFile -Raw | ConvertFrom-Json)
}

$config = [ordered]@{
    schema_version = 1
    model = [ordered]@{ backend = 'server-context'; path = $Model; embedding_model = $EmbeddingModel }
    runtime = [ordered]@{
        context_size = 0; n_predict = 128; n_threads = $Threads; n_gpu_layers = $GpuLayers
        default_mode = $DefaultMode; thinking_mode = $ThinkingMode; max_reflection_rounds = 2
        max_plan_revisions = 3; max_research_iterations = 4; memory_learn = 'post-turn'
        agent_plan = 'auto'; agent_trace = $true
    }
    stores = [ordered]@{
        memory = [ordered]@{ backend = 'cozo'; path = "$CozoRoot/memory.cozo" }
        plan = [ordered]@{ backend = 'cozo'; path = "$CozoRoot/plan.cozo" }
        data = [ordered]@{ backend = 'cozo'; path = "$CozoRoot/structured.cozo" }
    }
    resources = [ordered]@{
        blob_backend = 'fs'; blob_root = "$CozoRoot/resources"
        metadata_backend = 'cozo'; metadata_db = "$CozoRoot/resources.cozo"
    }
    tools = [ordered]@{ profile = $ToolProfile; repository_root = $RepositoryRoot; providers = $providers }
    sandbox = [ordered]@{
        backend = $Sandbox
        workspace = [ordered]@{
            root = "$CozoRoot/workspaces"; artifact_root = "$CozoRoot/artifacts"
            operation_mode = 'ephemeral'; project_mode = 'persistent'
        }
    }
    diagnostics = [ordered]@{
        semantic_backend = 'auto'; clang_executable = 'clang'; clangd_executable = 'clangd'
        compile_commands = 'auto'
    }
    limits = [ordered]@{
        queue_capacity = $QueueCapacity; worker_count = $WorkerCount
        inference_max_active = $InferenceMaxActive; turn_timeout_ms = 120000
        inference_step_timeout_ms = 0; tool_timeout_ms = 30000
        mcp_connect_timeout_ms = 5000; mcp_request_timeout_ms = 30000
        mcp_shutdown_timeout_ms = 2000; max_tool_rounds = 0
    }
}

$parent = Split-Path -Parent $Output
if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
$config | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $Output -Encoding UTF8
Write-Output "Wrote $Output"
