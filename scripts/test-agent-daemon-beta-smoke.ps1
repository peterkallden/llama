[CmdletBinding()]
param(
    [string]$BuildDir = "build-plan-resident-cozo-debug-3",
    [string]$Configuration = "Release",
    [string]$ChatModel = "$HOME\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    [int]$Port = 18090,
    [int]$TimeoutSeconds = 120,
    [switch]$IncludeCTest,
    [switch]$KeepLogs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$buildRoot = if ([IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $repoRoot $BuildDir }
$bin = Join-Path $buildRoot "bin\$Configuration"
$runId = [guid]::NewGuid().ToString("N")
$logRoot = Join-Path $env:TEMP "llama-agent-beta-test-$runId"
$results = [System.Collections.Generic.List[object]]::new()

function Invoke-TestProcess([string]$Name, [string]$FileName, [string]$Arguments = "") {
    $stdout = Join-Path $logRoot "$Name.stdout.log"
    $stderr = Join-Path $logRoot "$Name.stderr.log"
    $start = [Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $FileName
    $start.Arguments = $Arguments
    $start.WorkingDirectory = $repoRoot
    $start.UseShellExecute = $false
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $start
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $exitCode = $null
    $failure = $null
    try {
        [void]$process.Start()
        $outTask = $process.StandardOutput.ReadToEndAsync()
        $errTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            $process.Kill()
            $failure = "timeout after ${TimeoutSeconds}s"
        } else {
            $process.WaitForExit()
            $exitCode = $process.ExitCode
            if ($exitCode -ne 0) { $failure = "exit code $exitCode" }
        }
        [IO.File]::WriteAllText($stdout, $outTask.Result)
        [IO.File]::WriteAllText($stderr, $errTask.Result)
    } catch {
        $failure = $_.Exception.Message
    } finally {
        if (-not $process.HasExited) { $process.Kill() }
        $process.Dispose()
        $timer.Stop()
    }
    $status = if ($null -eq $failure) { "passed" } else { "failed" }
    $results.Add([pscustomobject]@{ Suite = $Name; Status = $status; DurationMs = $timer.ElapsedMilliseconds; Log = $stderr; Error = $failure })
    if ($failure) { Write-Warning "suite=$Name status=failed error=$failure log=$stderr" }
    else { Write-Output "suite=$Name status=passed duration_ms=$($timer.ElapsedMilliseconds)" }
}

function Assert-Executable([string]$Name) {
    $path = Join-Path $bin "$Name.exe"
    if (-not (Test-Path -LiteralPath $path)) { throw "Required executable not found: $path" }
    return $path
}

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
try {
    $deterministic = @(
        "llama-agent-runtime-operation-manager-smoke",
        "llama-agent-runtime-session-manager-smoke",
        "llama-agent-daemon-dispatcher-smoke",
        "llama-agent-daemon-protocol-smoke",
        "llama-agent-daemon-jsonl-protocol-smoke",
        "llama-agent-daemon-mcp-config-smoke",
        "llama-agent-deliberation-policy-smoke",
        "llama-agent-deliberate-runtime-smoke",
        "llama-agent-research-runtime-smoke",
        "test-agent-research-contract",
        "llama-agent-resource-store-smoke",
        "llama-agent-mcp-tool-provider-smoke",
        "llama-agent-mcp-stdio-client-smoke",
        "llama-agent-mcp-http-client-smoke",
        "llama-agent-mcp-http-inbound-dispatcher-smoke",
        "llama-agent-mcp-http-vertical-smoke",
        "llama-agent-mcp-agent-tools-smoke",
        "llama-agent-daemon-wait-events-smoke"
    )
    foreach ($target in $deterministic) {
        Invoke-TestProcess $target (Assert-Executable $target)
    }

    if ($IncludeCTest) {
        $ctest = (Get-Command ctest.exe -ErrorAction Stop).Source
        Invoke-TestProcess "ctest-agent" $ctest "--test-dir `"$buildRoot`" -C $Configuration -L agent --output-on-failure"
    }

    $tcp = Join-Path $repoRoot "scripts\test-agent-daemon-tcp-smoke.ps1"
    Invoke-TestProcess "daemon-tcp" (Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe") "-NoProfile -ExecutionPolicy Bypass -File `"$tcp`" -BuildDir `"$BuildDir`" -Configuration $Configuration -ChatModel `"$ChatModel`" -Port $Port"

    $daemon = Join-Path $repoRoot "scripts\test-agent-daemon-smoke.ps1"
    Invoke-TestProcess "daemon-foreground" (Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe") "-NoProfile -ExecutionPolicy Bypass -File `"$daemon`" -BuildDir `"$BuildDir`" -Configuration $Configuration -ChatModel `"$ChatModel`""

    $failed = @($results | Where-Object Status -eq "failed")
    Write-Output "agent_test_pack_suites=$($results.Count)"
    Write-Output "agent_test_pack_failed=$($failed.Count)"
    if ($failed.Count -ne 0) { throw "Agent beta test pack failed; logs: $logRoot" }
    Write-Output "agent_test_pack=passed"
} finally {
    if ($KeepLogs) {
        Write-Output "agent_test_pack_logs=$logRoot"
    } else {
        Remove-Item -LiteralPath $logRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
