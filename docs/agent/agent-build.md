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

The agent verification is layered by dependency and failure mode. The layers
are related, but they are not all independent CTest suites:

| Layer | Source | Character | Model required |
| --- | --- | --- | --- |
| Core agent contracts | `tests/` | deterministic C++ contracts for loops, planning, memory, bootstrap, and Cozo | No |
| Agent runtime smokes | `pocs/agent/smoke/` | model-free runtime, CLI/MCP, daemon, resource, data, and processor behavior | No |
| Sandbox backends | `pocs/agent/smoke/` | Docker/Podman and Kubernetes process/backend checks | No, external runtime required |
| Model-backed text flows | `scripts/test-qwen-*.sh` | Qwen chat, planning, resource synthesis, tools, Cozo, and Nomic embeddings | Qwen/Nomic |
| Multimodal flows | `scripts/test-agent-multimodal-smoke.sh` | capability-gated native image/audio and OCR fallback | multimodal model/projector |

The root CTest project aggregates the current agent-labelled contract and POC
tests. The exact inventory is build/configuration dependent; do not hard-code
layer counts in automation. In the current Linux packaging build,
`ctest --test-dir BUILD -L agent` discovers 73 agent tests, including the
OpenAPI catalog/provider and host materialization tests. The two sandbox
backend tests deliberately use the separate `sandbox-docker` and
`sandbox-kubernetes` labels.

For the complete unique agent CTest set, use one root invocation:

```bash
ctest --test-dir build-agent-packaging \
  -L 'agent|sandbox-docker|sandbox-kubernetes' \
  --output-on-failure
```

This discovers the complete configured set without relying on overlapping
directory views. Running CTest directly in `build-agent-packaging/pocs/agent`
is useful for isolating POC tests, but it may repeat tests already included by
the root `agent` label.

Model-backed shell smokes and multimodal smokes are intentionally outside
CTest. They have model availability, inference quality, projector capability,
and longer timeout behavior that should be reported separately from
deterministic CTest results.

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
contains [scripts/agent-build-env.ps1](../../scripts/agent-build-env.ps1), which
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
[agent-ci.yml](../../.github/workflows/agent-ci.yml). It provisions Cozo 0.7.6,
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
  -DLLAMA_MEMORY_COZO=ON `
  -DLLAMA_PLAN_COZO=ON `
  -DLLAMA_AGENT_TOOLS_CLANG=ON `
  -DLLAMA_BUILD_SERVER=ON `
  -DLLAMA_SUBPROCESS=ON `
  -DCOZO_INCLUDE_DIR="PATH_TO_COZO" `
  -DCOZO_LIBRARY="PATH_TO_COZO\\win\\libcozo_c.lib" `
  -DLLAMA_AGENT_CLANG_EXECUTABLE="PATH_TO_CLANG\\clang.exe" `
  -DLLAMA_AGENT_CLANGD_EXECUTABLE="PATH_TO_CLANGD\\clangd.exe"
```

`LLAMA_AGENT_RUNTIME=ON` automatically enables the agent memory and planning
libraries. The lower-level `LLAMA_MEMORY` and `LLAMA_PLAN` options remain
available for standalone builds, but are not needed in an agent configuration.

Build with one worker on MSVC. The agent tree is intentionally serial here:
Cozo-enabled Debug builds can exhaust the MSVC compiler heap when several large
translation units compile concurrently.

```powershell
cmake --build build-agent-windows `
  --config Debug `
  --parallel 1
```

For a package-oriented build, use the agent build-pack target when it is
available in the configured tree:

```powershell
cmake --build build-agent-windows `
  --config Release `
  --parallel 1 `
  --target llama-agent-build-pack
```

`Debug` is intended for development and diagnostics. `Release` is intended
for packaging and release validation; it should not be used as a substitute
for the Debug assurance run.

## Generic build scripts

The repository provides small wrappers for building an already configured
tree. They do not configure CMake; this keeps compiler, Cozo, and backend
options explicit in the platform-specific configure step above.

On Windows, use PowerShell from the repository root:

```powershell
.\scripts\agent-build.ps1 `
  -BuildDir E:\llama-builds\agent-tool-profiles-debug `
  -Configuration Debug `
  -Target llama-agent-cli, llama-agent-daemon `
  -Parallel 2
```

On Linux, use Bash from the repository root:

```bash
bash scripts/agent-build.sh \
  --build-dir build-agent-linux \
  --config RelWithDebInfo \
  --target llama-agent-build-pack \
  --parallel 4
```

Both scripts accept repeated target values and a verbose mode. Defaults can
also be supplied through `LLAMA_AGENT_BUILD_DIR`,
`LLAMA_AGENT_CONFIGURATION`, and `LLAMA_AGENT_BUILD_PARALLEL_LEVEL`.

### Agent CMake configuration manifests

The agent CMake install manifest keeps its host-configuration example list
explicit in `pocs/agent/CMakeLists.txt`. It intentionally does not use a
`CONFIGURE_DEPENDS` glob for these files: with the CMake/Ninja combination
used by the Windows agent build, an unchanged glob can leave the generated
verification output absent and cause the whole project to be reconfigured on
every build invocation. When a new `agent-host-config-*.json` example is
added, add it to that explicit list as part of the same change. This is a
local agent-build precaution and does not alter the repository-wide CMake
policy.

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
  -DLLAMA_MEMORY_COZO=ON \
  -DLLAMA_PLAN_COZO=ON \
  -DLLAMA_AGENT_RUNTIME=ON \
  -DLLAMA_AGENT_TOOLS_CLANG=ON \
  -DCOZO_INCLUDE_DIR="$COZO_INCLUDE_DIR" \
  -DCOZO_LIBRARY="$COZO_LIBRARY"

cmake --build build-agent-linux --parallel 4 \
  --target llama-agent-build-pack
```

## Debian and Ubuntu package build

The repository also contains a native Debian source package under
[`debian`](../../debian). Build it through `dpkg-buildpackage`; Debian's
`debian/rules` invokes `dh_auto_configure`, which creates the CMake build tree
with the package flags, and then invokes CMake directly to build the
`llama-agent-build-pack` target. The explicit CMake invocation is required
because `--target` is a CMake build option, not a `dh_auto_build` option.

Agent builds automatically enable the memory and planning libraries through
`LLAMA_AGENT_RUNTIME=ON`; only the optional Cozo backend switches need to be
specified explicitly.

The package uses Cozo-backed stores by default, so provide the Cozo C API
header and shared library explicitly. Debian backend selection is controlled
through `LLAMA_AGENT_BACKEND`; the default is portable CPU-only output:

```bash
sudo apt-get update
sudo apt-get install --yes \
  build-essential clang clangd cmake debhelper-compat dpkg-dev ninja-build \
  libssl-dev pkg-config zlib1g-dev

export COZO_INCLUDE_DIR=/path/to/cozo/cozo-lib-c
export COZO_LIBRARY=/path/to/libcozo_c.so
export LLAMA_AGENT_BACKEND=cpu
dpkg-buildpackage -us -uc -b
```

`LLAMA_AGENT_BACKEND` accepts `cpu`, `cuda`, `vulkan`, or `auto`. CPU is always
included; the selected backend adds CUDA or Vulkan support. Use explicit
backend selection for release packages so builds are reproducible:

```bash
LLAMA_AGENT_BACKEND=cuda \
LLAMA_AGENT_NATIVE=off \
dpkg-buildpackage -us -uc -b
```

`auto` detects the available CUDA and Vulkan development toolchains and is
intended mainly for local builds. The current Debian/Ubuntu workflow selects
`LLAMA_AGENT_BACKEND=cpu`; CUDA and Vulkan package variants will be added
separately. The installed runtime paths are identical across variants, so the
variants are mutually exclusive at the Debian package level.

`LLAMA_AGENT_NATIVE` defaults to `off` to keep Debian binaries portable. A
downstream maintainer building for a controlled host can set it to `on`. The
resulting package installs the daemon and bootstrap helper under `/usr/sbin`,
client binaries under `/usr/bin`, and examples under
`/usr/share/doc/llama-agent/examples`. The systemd unit is installed but not
enabled or started until `/etc/llama-agent/config.json` has been created.

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

The Docker-compatible sandbox backend can use either Docker or Podman. Docker
is the default; select another executable through the environment when running
the Linux smoke. Podman is daemonless for direct `run` commands, so no Podman
service is required.

For Docker:

```powershell
docker info

ctest --test-dir build-agent-windows `
  -C Debug `
  -L sandbox-docker `
  --output-on-failure
```

The Docker smoke uses the configured sandbox image and verifies command
execution plus artifact materialization inside the sandbox workspace.

For rootless Podman on Linux:

```bash
podman info
LLAMA_AGENT_SANDBOX_EXECUTABLE=podman \
  ctest --test-dir build-agent-linux \
    -L sandbox-docker --output-on-failure
```

The backend remains named `docker` in the agent contract; only the host-owned
container executable changes. The executable can also be set in generated
configuration with `--sandbox-executable podman`.

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

The Linux beta test pack runs the equivalent foreground daemon smoke when
`LLAMA_AGENT_CHAT_MODEL` points to a model file. Without that variable, the
model-backed smoke is reported as skipped while the deterministic pack still
runs:

```bash
export LLAMA_AGENT_CHAT_MODEL=/path/to/model.gguf
./scripts/test-agent-daemon-beta-smoke.sh --include-ctest --keep-logs
```

The matching Linux entry point is
`scripts/test-agent-daemon-beta-smoke.sh`. Model-backed runs should be
reported separately from model-free tests because they depend on model
availability, prompt behavior, runtime duration, and host resources.

For the focused Linux model-backed checks, use the Bash scripts below. They
share the same model environment and return `77` when a required model is not
available:

```bash
export LLAMA_AGENT_MODEL=/home/prbm/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
export LLAMA_AGENT_EMBEDDING_MODEL=/home/prbm/models/nomic-embed-text-v1.5.Q4_K_M.gguf
scripts/test-qwen-nomic-agent.sh
scripts/test-qwen-resource-synthesis.sh
scripts/test-agent-server-context-smoke.sh
scripts/test-qwen-nomic-agent-data.sh
scripts/test-qwen-nomic-document-table.sh
scripts/test-agent-daemon-integration.sh
```

Set `LLAMA_AGENT_BUILD=1` to build `llama-agent` before a script runs. These
scripts are intentionally outside the fast CTest gate because they depend on
model output and local CPU/memory capacity.

For daemon configuration and request examples, see
[Agent daemon usage](agent-daemon-usage.md). For the full runtime behavior and
protocol contracts, see [Agent runtime](agent-runtime.md). For the multi-model
loader, profile pinning, residency limits and scheduler integration, see
[Agent model residency and multi-model scheduling](agent-model-residency.md).

### Multi-model verification boundary

The current model-backed scripts exercise one configured generation model and
an optional separate embedding model. Process-wide multi-model residency is
covered by the model-free residency contract smoke, while loading two real
GGUF files is kept operator-supplied. The optional two-model verification must
remain separate from the ordinary Qwen/Nomic smoke and cover, at minimum:

- two named generation profiles with distinct model identities;
- profile reuse and profile isolation across two session lanes;
- eviction of an idle profile and refusal to evict a pinned profile;
- cancellation/deadline while waiting for residency or inference capacity;
- CLI and `server-context` loader selection with explicit unsupported-profile
  errors;
- continuation/checkpoint rejection when the profile identity has changed.

The target is `llama-agent-two-model-smoke` and accepts two explicit paths:

```bash
./bin/llama-agent-two-model-smoke \
  --model-1 /models/model-a.gguf \
  --model-2 /models/model-b.gguf \
  --backend server-context
```

`LLAMA_AGENT_MODEL_1` and `LLAMA_AGENT_MODEL_2` are supported for scripted
runs. The target is built but not registered in default CTest because it
loads real operator-provided files. It must not make Qwen a contract
requirement, and a daemon must not silently use the old `model.path` fallback
when a named catalog profile was requested.

The repository wrapper is:

```bash
./scripts/test-agent-two-model-smoke.sh \
  --build-dir /tmp/llama-agent-config-lifecycle-cozo \
  --model-1 /models/model-a.gguf \
  --model-2 /models/model-b.gguf
```

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
- [Agent CI workflow](../../.github/workflows/agent-ci.yml)
- [Agent dynamic analysis workflow](../../.github/workflows/agent-dynamic-analysis.yml)
- [Agent release workflow](../../.github/workflows/agent-release.yml)
- [Local build environment](../../scripts/agent-build-env.ps1)
