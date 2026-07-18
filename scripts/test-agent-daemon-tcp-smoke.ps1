[CmdletBinding()]
param(
    [string]$BuildDir = "build-plan-resident-cozo-debug-3",
    [string]$Configuration = "Release",
    [string]$ChatModel = "$HOME\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    [int]$Port = 18090
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$daemon = Join-Path $repoRoot "$BuildDir\bin\$Configuration\llama-agent-daemon.exe"
$config = Join-Path $repoRoot "docs\examples\agent-host-config-jsonl-tcp.json"
if (-not (Test-Path -LiteralPath $daemon)) { throw "Daemon not found: $daemon" }
if (-not (Test-Path -LiteralPath $ChatModel)) { throw "Chat model not found: $ChatModel" }

$readToken = "tcp-smoke-read-$([guid]::NewGuid().ToString('N'))"
$adminToken = "tcp-smoke-admin-$([guid]::NewGuid().ToString('N'))"
$env:LLAMA_AGENT_TCP_READ_TOKEN = $readToken
$env:LLAMA_AGENT_TCP_ADMIN_TOKEN = $adminToken
$process = $null
$client = $null
$writer = $null
$reader = $null
try {
    $start = [Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $daemon
    $start.WorkingDirectory = $repoRoot
    $start.UseShellExecute = $false
    $start.RedirectStandardInput = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $start.Arguments = '--config "{0}" --model "{1}" --tcp-port {2}' -f $config, $ChatModel, $Port
    $process = [Diagnostics.Process]::Start($start)

    $client = [Net.Sockets.TcpClient]::new()
    $deadline = [DateTime]::UtcNow.AddSeconds(15)
    while (-not $client.Connected -and [DateTime]::UtcNow -lt $deadline) {
        try { $null = $client.ConnectAsync("127.0.0.1", $Port).Wait(250) } catch { }
        if (-not $client.Connected) { Start-Sleep -Milliseconds 100 }
    }
    if (-not $client.Connected) { throw "TCP daemon did not accept a connection" }

    $stream = $client.GetStream()
    $reader = [IO.StreamReader]::new($stream)
    $writer = [IO.StreamWriter]::new($stream)
    $writer.AutoFlush = $true

    $ready = $reader.ReadLine() | ConvertFrom-Json
    if (-not $ready.ok -or $ready.event -ne "ready") { throw "TCP daemon did not report ready" }
    $writer.WriteLine((@{ authorization = "Bearer $readToken" } | ConvertTo-Json -Compress))
    $auth = $reader.ReadLine() | ConvertFrom-Json
    if (-not $auth.ok -or $auth.event -ne "status") { throw "TCP authentication/status handshake failed" }
    $writer.WriteLine('{"command":"status"}')
    $status = $reader.ReadLine() | ConvertFrom-Json
    if (-not $status.ok -or -not $status.ready) { throw "TCP status request failed" }
    $writer.WriteLine('{"command":"drain"}')
    $denied = $reader.ReadLine() | ConvertFrom-Json
    if ($denied.ok -or $denied.error -notlike '*does not allow daemon administration*') { throw "TCP read-only admin rejection failed" }
    $writer.Dispose(); $reader.Dispose(); $client.Dispose()
    $client = [Net.Sockets.TcpClient]::new()
    $client.Connect("127.0.0.1", $Port)
    $stream = $client.GetStream()
    $reader = [IO.StreamReader]::new($stream)
    $writer = [IO.StreamWriter]::new($stream)
    $writer.AutoFlush = $true
    $readyAgain = $reader.ReadLine() | ConvertFrom-Json
    if (-not $readyAgain.ok -or $readyAgain.event -ne "ready") { throw "TCP admin connection did not report ready" }
    $writer.WriteLine((@{ authorization = "Bearer $adminToken" } | ConvertTo-Json -Compress))
    $null = $reader.ReadLine() | ConvertFrom-Json
    $writer.WriteLine('{"command":"shutdown"}')
    $shutdown = $reader.ReadLine() | ConvertFrom-Json
    if (-not $shutdown.ok -or $shutdown.event -ne "shutdown") { throw "TCP admin shutdown failed" }
    if (-not $process.WaitForExit(10000)) { throw "TCP daemon did not exit after shutdown" }
    if ($process.ExitCode -ne 0) { throw "TCP daemon exited with code $($process.ExitCode)" }
    Write-Output "tcp_jsonl_ready=true"
    Write-Output "tcp_jsonl_authenticated=true"
    Write-Output "tcp_jsonl_readonly_admin_denied=true"
    Write-Output "tcp_jsonl_status=true"
    Write-Output "tcp_jsonl_shutdown=true"
}
finally {
    if ($writer) { $writer.Dispose() }
    if ($reader) { $reader.Dispose() }
    if ($client) { $client.Dispose() }
    if ($process) {
        if (-not $process.HasExited) { $process.Kill($true) }
        $process.Dispose()
    }
}
