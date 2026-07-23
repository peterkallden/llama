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

$pathEntries = @($llvmBin, $cmakeBin) + ($env:Path -split ';' | Where-Object { $_ })
$uniquePathEntries = [System.Collections.Generic.List[string]]::new()

foreach ($entry in $pathEntries) {
    if (-not $uniquePathEntries.Contains($entry)) {
        $uniquePathEntries.Add($entry)
    }
}

$env:Path = $uniquePathEntries -join ';'
$env:LLAMA_AGENT_LLVM_BIN = $llvmBin
$env:LLAMA_AGENT_CMAKE_BIN = $cmakeBin

Write-Host "Agent build environment configured for this PowerShell session."
Write-Host "  LLVM:  $llvmBin"
Write-Host "  CMake: $cmakeBin"
