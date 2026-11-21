# Configure the toolchain used by agent builds and smokes.
#
# Usage from PowerShell:
#   . .\scripts\agent-build-env.ps1
#   cmake --version
#   clang --version
# If the local execution policy blocks scripts, start a process-scoped shell:
#   powershell -NoProfile -ExecutionPolicy Bypass
#   . .\scripts\agent-build-env.ps1
#
# The environment changes are limited to the current PowerShell process. The
# script does not modify the user or system PATH permanently.

function Resolve-AgentToolDirectory {
    param(
        [string]$Override,
        [string]$Executable
    )

    if ($Override) {
        if (Test-Path -LiteralPath $Override -PathType Container) {
            $candidate = Join-Path $Override $Executable
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                return (Resolve-Path -LiteralPath $Override).Path
            }
            throw "$Executable was not found under LLAMA_AGENT tool directory: $Override"
        }

        $command = Get-Command $Override -ErrorAction Stop
        return Split-Path -Parent $command.Source
    }

    $command = Get-Command $Executable -ErrorAction Stop
    return Split-Path -Parent $command.Source
}

$cmakeBin = Resolve-AgentToolDirectory -Override $env:LLAMA_AGENT_CMAKE_BIN -Executable 'cmake.exe'
$llvmBin = Resolve-AgentToolDirectory -Override $env:LLAMA_AGENT_LLVM_BIN -Executable 'clang.exe'

foreach ($requiredPath in @($cmakeBin, $llvmBin)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Container)) {
        throw "Required toolchain directory was not found: $requiredPath"
    }
}

$pathEntries = @($llvmBin, $cmakeBin) + ($env:Path -split ';' | Where-Object { $_.Trim() })
$uniquePathEntries = [System.Collections.Generic.List[string]]::new()
$seenPathEntries = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)

foreach ($entry in $pathEntries) {
    $normalizedEntry = $entry.Trim().TrimEnd('\')
    if ($seenPathEntries.Add($normalizedEntry)) {
        $uniquePathEntries.Add($normalizedEntry)
    }
}

$env:Path = $uniquePathEntries -join ';'
$env:PATH = $env:Path
$env:LLAMA_AGENT_LLVM_BIN = $llvmBin
$env:LLAMA_AGENT_CMAKE_BIN = $cmakeBin

function Invoke-AgentBuild {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$BuildDirectory,
        [string]$Configuration = 'Release',
        [string]$Target,
        [int]$Parallel = 1
    )

    $arguments = @('--build', "`"$BuildDirectory`"", '--config', $Configuration, '--parallel', $Parallel)
    if ($Target) {
        $arguments += @('--target', $Target)
    }

    $cmakePath = Join-Path $cmakeBin 'cmake.exe'
    $argumentString = ($arguments | ForEach-Object {
        if ([string]$_ -match '\s') { "`"$_`"" } else { [string]$_ }
    }) -join ' '

    # cmd.exe lets us replace both case variants before MSBuild inherits the
    # environment.
    $cleanPath = $env:Path
    & cmd.exe /d /c ('set "Path=" & set "PATH=" & set "Path={0}" & "{1}" {2}' -f $cleanPath, $cmakePath, $argumentString)
    return $LASTEXITCODE
}

Write-Host "Agent build environment configured for this PowerShell session."
Write-Host "  LLVM:  $llvmBin"
Write-Host "  CMake: $cmakeBin"
