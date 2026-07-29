[CmdletBinding()]
param(
    [string]$BuildDir = $(if ($env:LLAMA_AGENT_BUILD_DIR) { $env:LLAMA_AGENT_BUILD_DIR } else { 'build-agent' }),
    [string]$Configuration = $(if ($env:LLAMA_AGENT_CONFIGURATION) { $env:LLAMA_AGENT_CONFIGURATION } else { 'RelWithDebInfo' }),
    [string[]]$Target = @(),
    [ValidateRange(1, 256)]
    [int]$Parallel = $(if ($env:LLAMA_AGENT_BUILD_PARALLEL_LEVEL) { [int]$env:LLAMA_AGENT_BUILD_PARALLEL_LEVEL } else { 2 }),
    [switch]$VerboseOutput
)

$ErrorActionPreference = 'Stop'

# Add the repository's local CMake/LLVM tools and normalize Path/PATH for MSBuild.
. (Join-Path $PSScriptRoot 'agent-build-env.ps1')

$cmakePath = Join-Path $env:LLAMA_AGENT_CMAKE_BIN 'cmake.exe'
if (-not (Test-Path -LiteralPath $cmakePath -PathType Leaf)) {
    throw "CMake was not found at $cmakePath"
}

$arguments = @('--build', $BuildDir, '--config', $Configuration, '--parallel', $Parallel)
if ($Target.Count -gt 0) {
    $arguments += @('--target') + $Target
}
if ($VerboseOutput) {
    $arguments += '--verbose'
}

Write-Host "Building '$BuildDir' ($Configuration) with $Parallel worker(s)."
if ($Target.Count -gt 0) {
    Write-Host "Target(s): $($Target -join ', ')"
}

# cmd.exe exposes only one case-variant of Path to avoid MSBuild MSB6001.
$cleanPath = $env:Path
$argumentString = ($arguments | ForEach-Object {
    $value = [string]$_
    if ($value -match '\s') { '"' + $value + '"' } else { $value }
}) -join ' '

& cmd.exe /d /c ('set "Path=" & set "PATH=" & set "Path={0}" & "{1}" {2}' -f $cleanPath, $cmakePath, $argumentString)
exit $LASTEXITCODE
