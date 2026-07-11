[CmdletBinding()]
param(
    [string]$BuildDir = "build-plan",
    [string]$Configuration = "Release",
    [string]$ChatModel = "$HOME\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    [switch]$Build
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

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

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

$cmake = Resolve-CMake
$exePath = Join-Path $repoRoot "$BuildDir\bin\$Configuration\llama-agent-daemon-dispatcher-smoke.exe"

Write-Host "Repo root: $repoRoot"
Write-Host "Build dir: $BuildDir"
Write-Host "Configuration: $Configuration"
Write-Host "Chat model: $ChatModel"

Assert-PathExists -Path $ChatModel -Label "Chat model"

if ($Build) {
    & $cmake --build $BuildDir --config $Configuration --target llama-agent-daemon-dispatcher-smoke -j 1
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }
}

Assert-PathExists -Path $exePath -Label "Dispatcher smoke executable"

function Show-Diagnostics {
    param([string]$Path)

    if (Test-Path -LiteralPath $Path) {
        $diag = @(Get-Content -LiteralPath $Path)
        if ($diag.Count -gt 0) {
            Write-Host ""
            Write-Host "Dispatcher diagnostics:"
            $diag | Select-Object -First 60 | ForEach-Object { Write-Host $_ }
        }
    }
}

$runId = [guid]::NewGuid().ToString("N")
$stdoutPath = Join-Path $env:TEMP "llama-agent-daemon-dispatcher-cancel-$runId-stdout.log"
$stderrPath = Join-Path $env:TEMP "llama-agent-daemon-dispatcher-cancel-$runId-stderr.log"
Remove-Item -LiteralPath $stdoutPath -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $stderrPath -ErrorAction SilentlyContinue

try {
    $cmd = "`"$exePath`" --model `"$ChatModel`" --default-mode chat -n 32 -ngl 0 1> `"$stdoutPath`" 2> `"$stderrPath`""
    cmd /d /c $cmd | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Show-Diagnostics -Path $stderrPath
        throw "Dispatcher cancel smoke failed with exit code $LASTEXITCODE"
    }

    $lines = @(Get-Content -LiteralPath $stdoutPath)
    if ($lines.Count -lt 4) {
        Show-Diagnostics -Path $stderrPath
        throw "Dispatcher cancel smoke produced too few summary lines"
    }

    Write-Host ""
    Write-Host "Agent daemon dispatcher cancel smoke test complete."
    $lines | Select-Object -First 4 | ForEach-Object { Write-Host $_ }
}
finally {
    Remove-Item -LiteralPath $stdoutPath -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stderrPath -ErrorAction SilentlyContinue
}
