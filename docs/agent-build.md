# Agent Build and Development Guide

This document describes the supported developer workflow for configuring,
building, and testing the llama agent runtime. It covers the model-free agent
contracts, the Cozo-backed runtime, the Docker and Kubernetes sandbox tests,
and the longer model-backed end-to-end checks.

Recorded verification results are maintained separately in
[Agent Assurance](agent-assurance.md). This page describes how to reproduce a
build or test run; it is not an assurance report.

## Build profiles

Use the following profiles for local development:

| Profile | Purpose |
| --- | --- |
| Windows Debug | MSVC diagnostics, local agent development, and focused CTest runs |
| Linux RelWithDebInfo | CI-compatible agent build with Cozo and Kubernetes support |
| Linux or Windows Release | Packaging and release verification |

The agent tests are intentionally split into separate groups. Contract tests
are deterministic and do not require a model. Sandbox tests require an
external Docker or Kubernetes runtime. End-to-end tests require a model and
additional runtime configuration.

## Windows prerequisites

Install or make available:

- Visual Studio C++ Build Tools with MSVC and the Windows SDK
- CMake
- `clang` and `clangd`
- Git
- Docker Desktop
- Kubernetes enabled in Docker Desktop
- `kubectl`
- The Cozo C API header, import library, and runtime DLL built for Windows

Start from a Visual Studio Developer PowerShell or an equivalent environment
that exposes MSVC, the Windows SDK, the linker, and CMake. The repository also
contains [scripts/agent-build-env.ps1](../scripts/agent-build-env.ps1), which
adds the local CMake and LLVM tools without modifying the user or system PATH
permanently.

On Windows, keep `Path` and `PATH` normalized to one environment variable
before invoking MSBuild. Duplicate case variants can cause MSBuild to fail
before compilation with an `MSB6001` error.

## Linux prerequisites

Install or make available:

- CMake
- Ninja
- GCC or Clang
- `clang` and `clangd`
- Git, curl, OpenSSL development files, and zlib development files
- Docker, when running Docker sandbox tests
- Kubernetes and `kubectl`, when running Kubernetes sandbox tests

The Linux dependency and Cozo setup used by CI is defined in
[agent-ci.yml](../.github/workflows/agent-ci.yml). It provisions Cozo 0.7.6,
uses the `alpine:3.20` Kubernetes smoke image, and builds with four parallel
workers.

## Windows configure and build

Use the Visual Studio generator with an out-of-source build directory. The
source directory is the current directory; the build directory should remain
separate from the source tree.

```powershell
cmake -S . -B build-agent-windows `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DLLAMA_BUILD_TESTS=ON `
  -DBUILD_TESTING=ON `
  -DLLAMA_TESTS_INSTALL=OFF `
  -DLLAMA_AGENT_RUNTIME=ON `
  -DLLAMA_MEMORY=ON `
  -DLLAMA_MEMORY_COZO=ON `
  -DLLAMA_PLAN=ON `
  -DLLAMA_PLAN_COZO=ON `
  -DLLAMA_AGENT_TOOLS_CLANG=ON `
  -DLLAMA_BUILD_SERVER=ON `
  -DLLAMA_SUBPROCESS=ON `
  -DCOZO_INCLUDE_DIR="PATH_TO_COZO" `
  -DCOZO_LIBRARY="PATH_TO_COZO\\win\\libcozo_c.lib" `
  -DLLAMA_AGENT_CLANG_EXECUTABLE="PATH_TO_CLANG\\clang.exe" `
  -DLLAMA_AGENT_CLANGD_EXECUTABLE="PATH_TO_CLANGD\\clangd.exe"
```

Build with four workers:

```powershell
cmake --build build-agent-windows `
  --config Debug `
  --parallel 4
```

For a package-oriented build, use the agent build-pack target when it is
available in the configured tree:

```powershell
cmake --build build-agent-windows `
  --config Release `
  --parallel 4 `
  --target llama-agent-build-pack
```

`Debug` is intended for development and diagnostics. `Release` is intended
for packaging and release validation; it should not be used as a substitute
for the Debug assurance run.

## Linux configure and build

The following configuration follows the working agent CI shape. Cozo paths
are supplied by the provisioning step or by the local environment.

```bash
cmake -S . -B build-agent-linux -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLAMA_BUILD_TESTS=ON \
  -DBUILD_TESTING=ON \
  -DLLAMA_TESTS_INSTALL=OFF \
  -DLLAMA_AGENT_PACKAGE_INSTALL=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_TOOLS_INSTALL=OFF \
  -DLLAMA_MEMORY=ON \
  -DLLAMA_MEMORY_COZO=ON \
  -DLLAMA_PLAN=ON \
  -DLLAMA_PLAN_COZO=ON \
  -DLLAMA_AGENT_RUNTIME=ON \
  -DLLAMA_AGENT_TOOLS_CLANG=ON \
  -DCOZO_INCLUDE_DIR="$COZO_INCLUDE_DIR" \
  -DCOZO_LIBRARY="$COZO_LIBRARY"

cmake --build build-agent-linux --parallel 4 \
  --target llama-agent-build-pack
```

## CTest labels

CTest labels select test categories independently from CMake build targets.
Always configure with `LLAMA_BUILD_TESTS=ON`; enabling `BUILD_TESTING=ON` alone
does not guarantee that the agent tests are built.

### Agent contract and runtime tests

These tests are model-free unless a test explicitly says otherwise:

```powershell
ctest --test-dir build-agent-windows `
  -C Debug `
  -L agent `
  --output-on-failure
```

On Linux:

```bash
ctest --test-dir build-agent-linux \
  -L agent \
  --output-on-failure \
  --timeout 900
```

The `agent` label includes agent contracts, runtime JSON and lifecycle tests,
tool and MCP tests, daemon protocol tests, and Cozo-backed data-store tests
when Cozo support is enabled.

### Docker sandbox

Docker Desktop must be running and the Docker CLI must be able to reach the
Docker engine:

```powershell
docker info

ctest --test-dir build-agent-windows `
  -C Debug `
  -L sandbox-docker `
  --output-on-failure
```

The Docker smoke uses the configured sandbox image and verifies command
execution plus artifact materialization inside the sandbox workspace.

### Kubernetes sandbox

Kubernetes must be enabled and `kubectl` must point at the intended cluster.
For the Docker Desktop Kubernetes cluster, the local API certificate may need
to be skipped by the smoke:

```powershell
kubectl config current-context
kubectl get nodes --insecure-skip-tls-verify=true

$env:LLAMA_AGENT_KUBERNETES_INSECURE_SKIP_TLS_VERIFY = "1"

ctest --test-dir build-agent-windows `
  -C Debug `
  -L sandbox-kubernetes `
  --output-on-failure
```

On Linux CI, the workflow creates an ephemeral kind cluster, prepares the
`llama-agent` namespace and local-path storage, and then runs the same CTest
label.

## End-to-end tests

End-to-end tests are separate from the deterministic CTest contract suite.
They require:

- a compatible model file
- a configured build directory and configuration
- a writable work directory
- Cozo database paths when persistent memory, plans, data, or resources are
  enabled
- Docker or Kubernetes when the selected tools require a sandbox
- longer timeouts and retained logs

The Windows beta test pack runs the deterministic agent executables and can
optionally include CTest:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\test-agent-daemon-beta-smoke.ps1 `
  -BuildDir build-agent-windows `
  -Configuration Debug `
  -ChatModel PATH_TO_MODEL `
  -IncludeCTest `
  -KeepLogs
```

The matching Linux entry point is
`scripts/test-agent-daemon-beta-smoke.sh`. Model-backed runs should be
reported separately from model-free tests because they depend on model
availability, prompt behavior, runtime duration, and host resources.

For daemon configuration and request examples, see
[Agent daemon usage](agent-daemon-usage.md). For the full runtime behavior and
protocol contracts, see [Agent runtime](agent-runtime.md).

## Troubleshooting

### MSVC reports duplicate `Path`/`PATH`

Start a fresh Visual Studio Developer PowerShell and ensure only one case
variant of the PATH variable is inherited. Do not repair the user or system
PATH registry values for this problem. Use a clean process environment or the
repository build-environment script.

### CTest reports no agent tests

Check both configuration and discovery:

```powershell
Select-String build-agent-windows\CMakeCache.txt `
  -Pattern "LLAMA_BUILD_TESTS|BUILD_TESTING|LLAMA_AGENT_RUNTIME"

ctest --test-dir build-agent-windows -C Debug -N
```

If the tests are absent, configure a fresh build tree with
`LLAMA_BUILD_TESTS=ON`.

### Cozo tests fail to start

Verify that the Cozo header, import library, and runtime DLL all belong to
the same ABI/configuration. The selected Cozo library must be reachable at
runtime, not only during linking.

### Sandbox tests are skipped

The sandbox tests use an explicit skip return code when their external runtime
is unavailable. Treat a skipped Docker or Kubernetes test as not run, not as
evidence that sandbox execution passed. Start the runtime and rerun the label.

## Related files

- [Agent assurance](agent-assurance.md)
- [Agent runtime](agent-runtime.md)
- [Agent daemon usage](agent-daemon-usage.md)
- [Agent CI workflow](../.github/workflows/agent-ci.yml)
- [Agent dynamic analysis workflow](../.github/workflows/agent-dynamic-analysis.yml)
- [Agent release workflow](../.github/workflows/agent-release.yml)
- [Local build environment](../scripts/agent-build-env.ps1)
