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

$cmakeBin = 'C:\Users\kalld\AppData\Roaming\Python\Python312\Scripts'
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

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = Join-Path $cmakeBin 'cmake.exe'
    $psi.Arguments = ($arguments | ForEach-Object {
        if ($_ -match '\s') { "`"$_`"" } else { [string]$_ }
    }) -join ' '
    $psi.UseShellExecute = $false
    $psi.WorkingDirectory = (Get-Location).Path

    # ProcessStartInfo receives a fresh, case-insensitive environment map.
    # This avoids MSBuild seeing both Path and PATH from the parent process.
    $psi.EnvironmentVariables.Clear()
    foreach ($entry in [Environment]::GetEnvironmentVariables('Process').GetEnumerator()) {
        if ($entry.Key -ine 'Path' -and $entry.Key -ine 'PATH') {
            $psi.EnvironmentVariables.Set_Item($entry.Key, [string]$entry.Value)
        }
    }
    $psi.EnvironmentVariables.Set_Item('Path', $env:Path)

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    [void]$process.Start()
    $process.WaitForExit()
    return $process.ExitCode
}

Write-Host "Agent build environment configured for this PowerShell session."
Write-Host "  LLVM:  $llvmBin"
Write-Host "  CMake: $cmakeBin"
