# Configure the local toolchain used by agent builds and smokes.
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

$cmakeCandidates = @(
    'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin',
    'C:\Users\kalld\AppData\Roaming\Python\Python312\Scripts'
)
$cmakeBin = $cmakeCandidates | Where-Object {
    Test-Path -LiteralPath (Join-Path $_ 'cmake.exe')
} | Select-Object -First 1
$llvmBin = 'E:\progs\bin'

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
    # environment.  This avoids the duplicate Path/PATH dictionary key seen
    # by MSBuild in the Codex Windows process.
    $cleanPath = $env:Path
    & cmd.exe /d /c ('set "Path=" & set "PATH=" & set "Path={0}" & "{1}" {2}' -f $cleanPath, $cmakePath, $argumentString)
    return $LASTEXITCODE
}

Write-Host "Agent build environment configured for this PowerShell session."
Write-Host "  LLVM:  $llvmBin"
Write-Host "  CMake: $cmakeBin"
