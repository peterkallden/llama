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
    [ValidateSet('none', 'docker', 'kubernetes', 'lxc')] [string]$Sandbox = 'none',
    [string]$SandboxExecutable = 'docker',
    [string]$LxcExecutable = 'lxc',
    [string]$LxcImage = 'ubuntu:24.04',
    [ValidateSet('none', 'profile')] [string]$LxcNetworkMode = 'none',
    [string]$LxcNetworkProfile = '',
    [ValidateSet('none', 'dns_only', 'allowlisted', 'package_registry', 'research_web')]
    [string]$LxcNetworkProfileScope = 'none',
    [ValidateSet('stdio', 'mcp-http', 'jsonl-tcp', 'jsonl-unix')] [string]$Transport = 'stdio',
    [string]$Listen = '127.0.0.1',
    [ValidateRange(1, 65535)] [int]$Port = 8080,
    [string]$UnixSocket = 'run/llama-agent.sock',
    [ValidateSet('none', 'opaque', 'jwt')] [string]$AuthMode = 'none',
    [string]$EnableTools = '',
    [switch]$ListTools,
    [string]$TokenEnv = '',
    [string]$TokenProfile = '',
    [string]$JwtIssuer = '',
    [string]$JwtAudience = '',
    [string]$JwtJwksUri = '',
    [string]$JwtToolProfile = '',
    [string]$JwtScope = '',
    [ValidateSet('disabled', 'local_preferred', 'local_required', 'sandbox_preferred', 'sandbox_required')]
    [string]$PdfPageImageExecution = 'disabled',
    [ValidateSet('auto', 'local', 'docker', 'kubernetes', 'lxc')]
    [string]$PdfPageImageBackend = 'auto',
    [string]$PdfPageImageExecutable = 'mutool',
    [string]$PdfPageImageExpectedVersion = '',
    [ValidateSet('disabled', 'local_preferred', 'local_required', 'sandbox_preferred', 'sandbox_required')]
    [string]$OcrTesseractExecution = 'disabled',
    [ValidateSet('auto', 'local', 'docker', 'kubernetes', 'lxc')]
    [string]$OcrTesseractBackend = 'auto',
    [string]$OcrTesseractExecutable = 'tesseract',
    [string]$OcrTesseractExpectedVersion = '',
    [ValidateSet('disabled', 'local_preferred', 'local_required', 'sandbox_preferred', 'sandbox_required')]
    [string]$PandocExecution = 'disabled',
    [ValidateSet('auto', 'local', 'docker', 'kubernetes', 'lxc')]
    [string]$PandocBackend = 'auto',
    [string]$PandocExecutable = 'pandoc',
    [string]$PandocExpectedVersion = '',
    [string]$ProvidersFile
)

if ($ListTools) {
    @(
        'Internal agent capabilities: memory, planning, deliberation, reflection, research, resources'
        'External catalog tools: calculator, time_now, memory_search, memory_get, memory_inspect, memory_conflict_check'
        '  repository.list, repository.search, repository.read, repository.diff, repository.log, repository.status'
        '  workspace.list, workspace.read, workspace.search, workspace.patch'
        '  diagnostics.compile, diagnostics.symbol, diagnostics.references, diagnostics.call_hierarchy'
        '  dataset.list, dataset.inspect, dataset.schema, dataset.sample, dataset.validate'
        '  data.query, data.filter, data.aggregate, data.join, data.transform'
        '  statistics.describe, statistics.outliers, statistics.value_counts, artifact.export, resource_read'
        '  web_search, web_fetch, development.build, development.test'
    ) | Write-Output
    exit 0
}

if ($AuthMode -eq 'opaque' -and ([string]::IsNullOrWhiteSpace($TokenEnv) -or [string]::IsNullOrWhiteSpace($TokenProfile))) {
    throw 'AuthMode opaque requires TokenEnv and TokenProfile'
}
if ($AuthMode -eq 'jwt' -and ([string]::IsNullOrWhiteSpace($JwtIssuer) -or [string]::IsNullOrWhiteSpace($JwtAudience) -or [string]::IsNullOrWhiteSpace($JwtJwksUri) -or [string]::IsNullOrWhiteSpace($JwtToolProfile))) {
    throw 'AuthMode jwt requires JwtIssuer, JwtAudience, JwtJwksUri and JwtToolProfile'
}
if ($EnableTools -and $AuthMode -eq 'none') {
    throw 'EnableTools requires AuthMode opaque or jwt'
}
if ($Sandbox -eq 'lxc' -and [string]::IsNullOrWhiteSpace($LxcNetworkProfile)) {
    throw 'LxcNetworkProfile is required when Sandbox is lxc'
}
if ($LxcNetworkMode -eq 'profile' -and [string]::IsNullOrWhiteSpace($LxcNetworkProfile)) {
    throw 'LxcNetworkProfile is required when LxcNetworkMode is profile'
}
if ($LxcNetworkMode -eq 'none' -and $LxcNetworkProfileScope -ne 'none') {
    throw 'LxcNetworkMode none requires LxcNetworkProfileScope none'
}

$processorPolicies = [ordered]@{}
if ($PdfPageImageExecution -ne 'disabled') {
    $processorPolicies['pdf.page_image'] = [ordered]@{
        execution = $PdfPageImageExecution; backend = $PdfPageImageBackend
        executable = $PdfPageImageExecutable; expected_version = $PdfPageImageExpectedVersion
    }
}
if ($OcrTesseractExecution -ne 'disabled') {
    $processorPolicies['ocr.tesseract'] = [ordered]@{
        execution = $OcrTesseractExecution; backend = $OcrTesseractBackend
        executable = $OcrTesseractExecutable; expected_version = $OcrTesseractExpectedVersion
    }
}
if ($PandocExecution -ne 'disabled') {
    foreach ($policyId in @('docx.text', 'odt.text', 'html.text')) {
        $processorPolicies[$policyId] = [ordered]@{
            execution = $PandocExecution; backend = $PandocBackend
            executable = $PandocExecutable; expected_version = $PandocExpectedVersion
        }
    }
}

$providers = @()
if ($ProvidersFile) {
    if ($ProvidersFile -eq '-') {
        $providers = @([Console]::In.ReadToEnd() | ConvertFrom-Json)
    } elseif (-not (Test-Path -LiteralPath $ProvidersFile -PathType Leaf)) {
        throw "Provider file not found: $ProvidersFile"
    } else {
        $providers = @(Get-Content -LiteralPath $ProvidersFile -Raw | ConvertFrom-Json)
    }
}

$config = [ordered]@{
    schema_version = 1
    model = [ordered]@{ backend = 'server-context'; path = $Model; embedding_model = $EmbeddingModel }
    runtime = [ordered]@{
        context_size = 3072; n_predict = 128; n_threads = $Threads; n_gpu_layers = $GpuLayers
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
        docker = [ordered]@{ executable = $SandboxExecutable; default_image = 'llama-agent-dev:latest' }
        lxc = [ordered]@{ executable = $LxcExecutable; default_image = $LxcImage; network_mode = $LxcNetworkMode; network_profile = $LxcNetworkProfile; network_profile_scope = $LxcNetworkProfileScope; cleanup = $true }
        kubernetes = [ordered]@{ namespace = 'llama-agent'; service_account = 'llama-agent-runner'; runtime_class = 'standard'; cleanup = $true }
        workspace = [ordered]@{
            root = "$CozoRoot/workspaces"; artifact_root = "$CozoRoot/artifacts"
            operation_mode = 'ephemeral'; project_mode = 'persistent'
        }
        defaults = [ordered]@{ timeout_ms = 60000; cpu_count = 1; max_output_bytes = 65536; network = 'none'; filesystem = 'readonly'; allow_artifacts = $true }
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

if ($processorPolicies.Count -gt 0) { $config.resources.processor_policies = $processorPolicies }

$enabledTools = @()
if ($EnableTools -and $EnableTools -ne 'none') { $enabledTools = @($EnableTools -split ',' | ForEach-Object { $_.Trim() }) }
$inbound = [ordered]@{ enabled = ($Transport -eq 'mcp-http'); listen = $Listen; port = $Port; path = '/mcp'; agent_tools = $false; max_delegation_depth = 1 }
if ($AuthMode -eq 'none') {
    $inbound.tokens = @()
} elseif ($AuthMode -eq 'opaque') {
    $token = [ordered]@{ id = 'bootstrap-client'; token_env = $TokenEnv; audience = 'llama-agent'; namespace = 'local'; project = 'default'; tool_profile = $TokenProfile; allow_writes = $false }
    if ($EnableTools) { $token.allowed_tools = $enabledTools }
    $inbound.tokens = @($token)
} else {
    $inbound.authorization = [ordered]@{ mode = 'jwt'; issuer = $JwtIssuer; audience = $JwtAudience; jwks_uri = $JwtJwksUri; allowed_algorithms = @('RS256'); required_scopes = @($JwtScope); tool_profile = $JwtToolProfile; allowed_tools = $enabledTools; allow_writes = $false }
}
$config.mcp = [ordered]@{ inbound = $inbound }
$config.jsonl = [ordered]@{
    tcp = [ordered]@{ enabled = ($Transport -eq 'jsonl-tcp'); listen = $Listen; port = $Port; max_line_bytes = 1048576; idle_timeout_seconds = 300 }
    unix_socket = [ordered]@{ enabled = ($Transport -eq 'jsonl-unix'); path = $UnixSocket; mode = 432 }
}

$parent = Split-Path -Parent $Output
if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
$config | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $Output -Encoding UTF8
Write-Output "Wrote $Output"
