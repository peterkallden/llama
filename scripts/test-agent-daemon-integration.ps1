[CmdletBinding()]
param(
    [string]$BuildDir = "build-plan-resident-debug",
    [string]$Configuration = "Release",
    [string]$ChatModel = "$HOME\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    [string]$EmbeddingModel = "$HOME\models\nomic-embed-text-v1.5.Q4_K_M.gguf",
    [switch]$Build
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -TypeDefinition @"
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading;

public sealed class AgentDaemonHarness : IDisposable {
    private readonly Process process;
    private readonly ConcurrentQueue<string> stdoutLines = new ConcurrentQueue<string>();
    private readonly ConcurrentQueue<string> stderrLines = new ConcurrentQueue<string>();
    private bool disposed = false;

    public AgentDaemonHarness(string exePath, string arguments, string workingDirectory) {
        var psi = new ProcessStartInfo();
        psi.FileName = exePath;
        psi.Arguments = arguments;
        psi.WorkingDirectory = workingDirectory;
        psi.UseShellExecute = false;
        psi.RedirectStandardInput = true;
        psi.RedirectStandardOutput = true;
        psi.RedirectStandardError = true;
        psi.CreateNoWindow = true;

        process = new Process();
        process.StartInfo = psi;
        process.OutputDataReceived += (sender, args) => {
            if (args.Data != null) stdoutLines.Enqueue(args.Data);
        };
        process.ErrorDataReceived += (sender, args) => {
            if (args.Data != null) stderrLines.Enqueue(args.Data);
        };

        if (!process.Start()) {
            throw new InvalidOperationException("Failed to start daemon process");
        }

        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
    }

    public bool HasExited {
        get { return process.HasExited; }
    }

    public int ExitCode {
        get { return process.HasExited ? process.ExitCode : 0; }
    }

    public void SendLine(string line) {
        process.StandardInput.WriteLine(line);
        process.StandardInput.Flush();
    }

    public bool TryReadStdoutLine(out string line) {
        return stdoutLines.TryDequeue(out line);
    }

    public string DrainStderrTail(int maxLines) {
        var lines = new List<string>();
        string line;
        while (stderrLines.TryDequeue(out line)) {
            if (!string.IsNullOrWhiteSpace(line)) {
                lines.Add(line);
            }
        }

        if (lines.Count == 0) {
            return string.Empty;
        }

        var start = Math.Max(0, lines.Count - maxLines);
        var builder = new StringBuilder();
        for (var i = start; i < lines.Count; ++i) {
            if (builder.Length > 0) builder.AppendLine();
            builder.Append(lines[i]);
        }
        return builder.ToString();
    }

    public bool WaitForExit(int milliseconds) {
        return process.WaitForExit(milliseconds);
    }

    public void Kill() {
        if (!process.HasExited) {
            process.Kill();
        }
    }

    public void Dispose() {
        if (disposed) return;
        disposed = true;
        try { process.CancelOutputRead(); } catch { }
        try { process.CancelErrorRead(); } catch { }
        try {
            if (!process.HasExited) {
                process.Kill();
                process.WaitForExit(5000);
            }
        } catch { }
        process.Dispose();
    }
}
"@

function Assert-PathExists {
    param(
        [string]$Path,
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label not found: $Path"
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

function Quote-CmdArg {
    param([string]$Value)

    if ($Value -notmatch '[\s"]') {
        return $Value
    }

    return '"' + $Value.Replace('"', '\"') + '"'
}

function Get-DaemonDiagnostics {
    param($Client)

    if ($null -eq $Client) {
        return ""
    }
    return $Client.DrainStderrTail(80)
}

function Read-DaemonResponse {
    param(
        $Client,
        [int]$TimeoutSeconds = 60
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        $line = $null
        if ($Client.TryReadStdoutLine([ref]$line)) {
            if ([string]::IsNullOrWhiteSpace($line)) {
                continue
            }

            try {
                return $line | ConvertFrom-Json
            } catch {
                $diagnostics = Get-DaemonDiagnostics $Client
                throw "Daemon emitted a non-JSON protocol line: $line`n$diagnostics"
            }
        }

        if ($Client.HasExited) {
            $diagnostics = Get-DaemonDiagnostics $Client
            throw "Daemon exited before returning a response (exit_code=$($Client.ExitCode))`n$diagnostics"
        }

        Start-Sleep -Milliseconds 50
    }

    $diagnostics = Get-DaemonDiagnostics $Client
    throw "Timed out waiting for daemon response`n$diagnostics"
}

function Start-AgentDaemon {
    param(
        [string]$ExePath,
        [string[]]$ArgumentList
    )

    $arguments = ($ArgumentList | ForEach-Object { Quote-CmdArg $_ }) -join ' '
    $client = [AgentDaemonHarness]::new($ExePath, $arguments, (Split-Path -Parent $ExePath))

    $ready = Read-DaemonResponse -Client $client -TimeoutSeconds $script:ReadyTimeoutSeconds
    $readyJson = $ready | ConvertTo-Json -Compress -Depth 10
    $readyOkMember = $ready | Get-Member -Name "ok" -MemberType NoteProperty -ErrorAction SilentlyContinue
    $readyOk = if ($null -ne $readyOkMember) {
        [bool]$ready.ok
    } else {
        $false
    }
    $readyEventMember = $ready | Get-Member -Name "event" -MemberType NoteProperty -ErrorAction SilentlyContinue
    $readyEvent = if ($null -ne $readyEventMember) {
        [string]$ready.event
    } else {
        ""
    }
    if (-not $readyOk -or $readyEvent -ne "ready") {
        $diagnostics = Get-DaemonDiagnostics $client
        throw "Unexpected daemon ready response: $readyJson`n$diagnostics"
    }

    return $client
}

function Send-DaemonCommand {
    param(
        $Client,
        $Command,
        [int]$TimeoutSeconds = 60
    )

    $json = $Command | ConvertTo-Json -Compress -Depth 10
    $Client.SendLine($json)
    while ($true) {
        $response = Read-DaemonResponse -Client $Client -TimeoutSeconds $TimeoutSeconds
        $messageTypeMember = $response | Get-Member -Name "message_type" -MemberType NoteProperty -ErrorAction SilentlyContinue
        if ($null -eq $messageTypeMember -or [string]$response.message_type -ne "event") {
            return $response
        }
    }
}

function Stop-AgentDaemon {
    param($Client)

    if ($null -eq $Client) {
        return
    }

    if (-not $Client.HasExited) {
        try {
            $shutdown = Send-DaemonCommand -Client $Client -Command @{
                command = "shutdown"
                request_id = "shutdown"
            } -TimeoutSeconds 20
            if (-not $shutdown.ok -or $shutdown.event -ne "shutdown") {
                throw "Unexpected shutdown response: $($shutdown | ConvertTo-Json -Compress)"
            }
        } catch {
            try {
                $Client.Kill()
            } catch {
            }
        }
    }

    if (-not $Client.HasExited) {
        $Client.WaitForExit(10000) | Out-Null
        if (-not $Client.HasExited) {
            $Client.Kill()
        }
    }

    $Client.Dispose()
}

function Assert-True {
    param(
        [bool]$Condition,
        [string]$Message
    )

    if (-not $Condition) {
        throw $Message
    }
}

function Invoke-ToolingConfiguredScenario {
    param(
        [string]$ExePath,
        [string]$FakeServerPath,
        [string[]]$BaseArgs
    )

    $client = $null
    try {
        $toolArgs = @($BaseArgs) + @(
            "--tool-profile", "minimal",
            "--repository-root", $repoRoot,
            "--mcp-tool-command", $FakeServerPath,
            "--mcp-tool-server-name", "github",
            "--mcp-tool-prefix", "github",
            "--max-tool-rounds", "2"
        )

        $client = Start-AgentDaemon -ExePath $ExePath -ArgumentList $toolArgs

        $turn1 = Send-DaemonCommand -Client $client -Command @{
            request_id = "tooling-turn-1"
            mode = "chat"
            prompt = "Reply with OK only."
            session_id = "integration-tooling"
            namespace_id = "integration"
            project_id = "repo-tooling"
            memory_scope = "project"
            plan_scope = "project"
        } -TimeoutSeconds $chatTurnTimeoutSeconds
        Assert-True ($turn1.ok -and $turn1.response -eq "OK" -and -not $turn1.runtime_reused) "First tooling-configured turn should succeed without runtime reuse"

        $turn2 = Send-DaemonCommand -Client $client -Command @{
            request_id = "tooling-turn-2"
            mode = "chat"
            prompt = "Reply with DONE only."
            session_id = "integration-tooling"
            namespace_id = "integration"
            project_id = "repo-tooling"
            memory_scope = "project"
            plan_scope = "project"
        } -TimeoutSeconds $chatTurnTimeoutSeconds
        Assert-True ($turn2.ok -and $turn2.response -eq "DONE" -and $turn2.runtime_reused) "Second tooling-configured turn should reuse the resident runtime"

        $status = Send-DaemonCommand -Client $client -Command @{
            command = "status"
            request_id = "tooling-status"
        }
        Assert-True ($status.ok -and $status.sessions -eq 1) "Tooling-configured daemon should report one active session"

        Write-Host "tooling_configured=ok"
    }
    finally {
        Stop-AgentDaemon $client
    }
}

function Invoke-ToolingProbeScenario {
    param(
        [string]$ProbeExePath
    )

    $output = cmd /d /c "`"$ProbeExePath`""
    if ($LASTEXITCODE -ne 0) {
        throw "Tooling probe executable failed with exit code $LASTEXITCODE"
    }

    $joined = ($output -join [Environment]::NewLine)
    Assert-True ($joined.Contains("daemon_mcp_ready=true")) "Tooling probe did not report ready daemon state"
    Assert-True ($joined.Contains("daemon_tooling_tools=")) "Tooling probe did not report resolved tooling count"
    Write-Host "tooling_probe=ok"
}

function Invoke-ChatLifecycleScenario {
    param(
        [string]$ExePath,
        [string[]]$BaseArgs
    )

    $client = $null
    try {
        $client = Start-AgentDaemon -ExePath $ExePath -ArgumentList $BaseArgs

        $status1 = Send-DaemonCommand -Client $client -Command @{
            command = "status"
            request_id = "chat-status-1"
        }
        Assert-True ($status1.ok -and $status1.sessions -eq 0) "Initial daemon status should report zero sessions"

        $turn1 = Send-DaemonCommand -Client $client -Command @{
            request_id = "chat-turn-1"
            mode = "chat"
            prompt = "Reply with OK only."
            session_id = "integration-chat"
            namespace_id = "integration"
            project_id = "repo-chat"
            memory_scope = "project"
            plan_scope = "project"
        } -TimeoutSeconds $chatTurnTimeoutSeconds
        Assert-True ($turn1.ok -and $turn1.response -eq "OK" -and -not $turn1.runtime_reused) "First chat turn should succeed without runtime reuse"

        $turn2 = Send-DaemonCommand -Client $client -Command @{
            request_id = "chat-turn-2"
            mode = "chat"
            prompt = "Reply with DONE only."
            session_id = "integration-chat"
            namespace_id = "integration"
            project_id = "repo-chat"
            memory_scope = "project"
            plan_scope = "project"
        } -TimeoutSeconds $chatTurnTimeoutSeconds
        Assert-True ($turn2.ok -and $turn2.response -eq "DONE" -and $turn2.runtime_reused) "Second chat turn should reuse the resident runtime"

        $reset = Send-DaemonCommand -Client $client -Command @{
            command = "reset_session"
            request_id = "chat-reset"
            session_id = "integration-chat"
            namespace_id = "integration"
        }
        Assert-True ($reset.ok -and $reset.event -eq "session_reset") "reset_session should succeed"

        $turn3 = Send-DaemonCommand -Client $client -Command @{
            request_id = "chat-turn-3"
            mode = "chat"
            prompt = "Reply with AGAIN only."
            session_id = "integration-chat"
            namespace_id = "integration"
            project_id = "repo-chat"
            memory_scope = "project"
            plan_scope = "project"
        } -TimeoutSeconds $chatTurnTimeoutSeconds
        Assert-True ($turn3.ok -and $turn3.response -eq "AGAIN" -and -not $turn3.runtime_reused) "Turn after reset should rebuild the resident runtime"

        $close = Send-DaemonCommand -Client $client -Command @{
            command = "close_session"
            request_id = "chat-close"
            session_id = "integration-chat"
            namespace_id = "integration"
        }
        Assert-True ($close.ok -and $close.event -eq "session_closed") "close_session should succeed"

        $status2 = Send-DaemonCommand -Client $client -Command @{
            command = "status"
            request_id = "chat-status-2"
        }
        Assert-True ($status2.ok -and $status2.sessions -eq 0) "Final daemon status should report zero sessions after close"

        Write-Host "chat_lifecycle=ok"
    }
    finally {
        Stop-AgentDaemon $client
    }
}

function Invoke-ProjectSwitchScenario {
    param(
        [string]$ExePath,
        [string[]]$BaseArgs
    )

    $client = $null
    try {
        $client = Start-AgentDaemon -ExePath $ExePath -ArgumentList $BaseArgs

        $turnA = Send-DaemonCommand -Client $client -Command @{
            request_id = "project-turn-a"
            mode = "chat"
            prompt = "Reply with A only."
            session_id = "shared-session"
            namespace_id = "integration"
            project_id = "project-a"
            memory_scope = "project"
            plan_scope = "project"
        } -TimeoutSeconds $chatTurnTimeoutSeconds
        Assert-True ($turnA.ok -and $turnA.response -eq "A" -and -not $turnA.runtime_reused) "First project-bound turn should succeed without runtime reuse"

        $turnB = Send-DaemonCommand -Client $client -Command @{
            request_id = "project-turn-b"
            mode = "chat"
            prompt = "Reply with B only."
            session_id = "shared-session"
            namespace_id = "integration"
            project_id = "project-b"
            memory_scope = "project"
            plan_scope = "project"
        } -TimeoutSeconds $chatTurnTimeoutSeconds
        Assert-True ($turnB.ok -and $turnB.response -eq "B" -and -not $turnB.runtime_reused) "Project switch should rebuild the resident runtime"

        $status = Send-DaemonCommand -Client $client -Command @{
            command = "status"
            request_id = "project-status"
        }
        Assert-True ($status.ok -and $status.sessions -eq 1) "Project-switch scenario should leave one tracked session"
        Assert-True ($status.session_keys.Count -eq 1 -and $status.session_keys[0].project_id -eq "project-b") "Tracked session should reflect the latest project binding"

        Write-Host "project_switch=ok"
    }
    finally {
        Stop-AgentDaemon $client
    }
}

function Invoke-TraceScenario {
    param(
        [string]$ExePath,
        [string[]]$BaseArgs
    )

    $client = $null
    try {
        $client = Start-AgentDaemon -ExePath $ExePath -ArgumentList $BaseArgs

        $turn = Send-DaemonCommand -Client $client -Command @{
            request_id = "trace-turn-1"
            mode = "agent"
            prompt = "Reply with OK only after making a tiny plan."
            session_id = "integration-trace"
            namespace_id = "integration"
            project_id = "repo-trace"
            memory_scope = "project"
            plan_scope = "project"
        } -TimeoutSeconds $agentTurnTimeoutSeconds

        Assert-True ($turn.ok -and $turn.response -eq "OK") "Trace scenario turn should succeed"
        Assert-True ($turn.trace_count -gt 0) "Trace scenario should expose trace entries"
        Assert-True ($turn.trace.Count -gt 0) "Trace scenario should include a non-empty trace array"

        $stages = @($turn.trace | ForEach-Object { $_.stage })
        Assert-True ($stages -contains "plan") "Trace scenario should include a plan trace stage"
        Assert-True ($stages -contains "reflection") "Deliberate trace scenario should include a reflection trace stage"
        Assert-True ($stages -contains "response") "Trace scenario should include a response trace stage"

        Write-Host "trace_visible=ok"
    }
    finally {
        Stop-AgentDaemon $client
    }
}

function Invoke-AgentLearningScenario {
    param(
        [string]$ExePath,
        [string[]]$BaseArgs
    )

    $client = $null
    try {
        $client = Start-AgentDaemon -ExePath $ExePath -ArgumentList $BaseArgs

        $turn1 = Send-DaemonCommand -Client $client -Command @{
            request_id = "agent-turn-1"
            mode = "agent"
            prompt = "Say OK after making a tiny plan about remembering that the project codename is Maple."
            session_id = "integration-agent"
            namespace_id = "integration"
            project_id = "repo-agent"
            memory_scope = "project"
            plan_scope = "project"
        } -TimeoutSeconds $agentTurnTimeoutSeconds
        Assert-True ($turn1.ok -and $turn1.response -eq "OK" -and -not $turn1.runtime_reused) "First agent turn should succeed without runtime reuse"
        Assert-True (-not [string]::IsNullOrWhiteSpace($turn1.plan_id)) "First agent turn should expose a plan_id"
        Assert-True (-not [string]::IsNullOrWhiteSpace($turn1.memory_learning_summary)) "First agent turn should expose a memory learning summary"

        $turn2 = Send-DaemonCommand -Client $client -Command @{
            request_id = "agent-turn-2"
            mode = "agent"
            prompt = "Say DONE after making a tiny plan about remembering that the daemon target is llama-agent-daemon."
            session_id = "integration-agent"
            namespace_id = "integration"
            project_id = "repo-agent"
            memory_scope = "project"
            plan_scope = "project"
        } -TimeoutSeconds $agentTurnTimeoutSeconds
        Assert-True ($turn2.ok -and $turn2.response -eq "DONE" -and $turn2.runtime_reused) "Second agent turn should reuse the resident runtime"
        Assert-True ($turn2.plan_id -eq $turn1.plan_id) "Agent scenario should preserve plan_id across turns"
        Assert-True (-not [string]::IsNullOrWhiteSpace($turn2.memory_learning_summary)) "Second agent turn should expose a memory learning summary"

        Write-Host "agent_learning=ok"
    }
    finally {
        Stop-AgentDaemon $client
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

$cmake = Resolve-CMake
$resolvedBuildDir = if ([System.IO.Path]::IsPathRooted($BuildDir)) {
    [System.IO.Path]::GetFullPath($BuildDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $BuildDir))
}
$daemonPath = Join-Path $resolvedBuildDir "bin\$Configuration\llama-agent-daemon.exe"
$fakeServerPath = Join-Path $resolvedBuildDir "bin\$Configuration\llama-agent-mcp-stdio-fake-server.exe"
$toolingProbePath = Join-Path $resolvedBuildDir "bin\$Configuration\llama-agent-daemon-mcp-config-smoke.exe"
$isDebugConfiguration = $Configuration -ieq "Debug"
$script:ReadyTimeoutSeconds = if ($isDebugConfiguration) { 120 } else { 60 }
$chatTurnTimeoutSeconds = if ($isDebugConfiguration) { 300 } else { 180 }
$agentTurnTimeoutSeconds = if ($isDebugConfiguration) { 480 } else { 240 }

Write-Host "Repo root: $repoRoot"
Write-Host "Build dir: $resolvedBuildDir"
Write-Host "Configuration: $Configuration"
Write-Host "Chat model: $ChatModel"
Write-Host "Embedding model: $EmbeddingModel"

Assert-PathExists -Path $ChatModel -Label "Chat model"

$haveEmbeddingModel = Test-Path -LiteralPath $EmbeddingModel

if ($Build) {
    $targets = @("llama-agent-daemon", "llama-agent-mcp-stdio-fake-server", "llama-agent-daemon-mcp-config-smoke")
    & $cmake --build $resolvedBuildDir --config $Configuration --target $targets -j 1
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }
}

Assert-PathExists -Path $daemonPath -Label "Daemon executable"
Assert-PathExists -Path $fakeServerPath -Label "Fake MCP server executable"
Assert-PathExists -Path $toolingProbePath -Label "Tooling probe executable"

$chatArgs = @(
    "--model", $ChatModel,
    "--default-mode", "chat",
    "-n", "32",
    "-ngl", "0"
)

Invoke-ChatLifecycleScenario -ExePath $daemonPath -BaseArgs $chatArgs
Invoke-ProjectSwitchScenario -ExePath $daemonPath -BaseArgs $chatArgs
Invoke-ToolingConfiguredScenario -ExePath $daemonPath -FakeServerPath $fakeServerPath -BaseArgs $chatArgs
Invoke-ToolingProbeScenario -ProbeExePath $toolingProbePath

$traceArgs = @(
    "--model", $ChatModel,
    "--default-mode", "agent",
    "--thinking-mode", "deliberate",
    "--max-reflection-rounds", "1",
    "--max-plan-revisions", "1",
    "--agent-plan", "auto",
    "-n", "64",
    "-ngl", "0"
)

Invoke-TraceScenario -ExePath $daemonPath -BaseArgs $traceArgs

if ($haveEmbeddingModel) {
    $agentArgs = @(
        "--model", $ChatModel,
        "--embedding-model", $EmbeddingModel,
        "--default-mode", "agent",
        "--thinking-mode", "reflective",
        "--memory-learn", "post-turn",
        "--agent-plan", "auto",
        "-n", "64",
        "-ngl", "0"
    )
    Invoke-AgentLearningScenario -ExePath $daemonPath -BaseArgs $agentArgs
} else {
    Write-Host "agent_learning=skipped"
}

Write-Host ""
Write-Host "Agent daemon integration test complete."
