# llama-agent

[![Agent verification (Linux)](https://github.com/peterkallden/llama/actions/workflows/agent-ci.yml/badge.svg?branch=feature%2Fllama-agent)](https://github.com/peterkallden/llama/actions/workflows/agent-ci.yml)
[![Agent verification (sanitizers)](https://github.com/peterkallden/llama/actions/workflows/agent-analysis.yml/badge.svg?branch=feature%2Fllama-agent)](https://github.com/peterkallden/llama/actions/workflows/agent-analysis.yml)
[![Agent development package (.tar.gz)](https://github.com/peterkallden/llama/actions/workflows/agent-package-archive.yml/badge.svg)](https://github.com/peterkallden/llama/actions/workflows/agent-package-archive.yml)
[![Agent development package (Debian/Ubuntu)](https://github.com/peterkallden/llama/actions/workflows/agent-package-debian.yml/badge.svg)](https://github.com/peterkallden/llama/actions/workflows/agent-package-debian.yml)
[![Agent development package (Windows)](https://github.com/peterkallden/llama/actions/workflows/agent-package-windows.yml/badge.svg)](https://github.com/peterkallden/llama/actions/workflows/agent-package-windows.yml)
[![Agent container images](https://github.com/peterkallden/llama/actions/workflows/agent-container-images.yml/badge.svg?branch=feature%2Fllama-agent)](https://github.com/peterkallden/llama/actions/workflows/agent-container-images.yml)
[![Latest agent tag](https://img.shields.io/github/v/tag/peterkallden/llama?filter=agent-v*&sort=semver)](https://github.com/peterkallden/llama/tags)

`llama-agent` is a resident agent runtime built on llama.cpp. It provides
host-controlled tools and capabilities, persistent memory and planning,
bounded reflection and research, resource handling, daemon operation, and MCP
integration.

## Current workflow state

The agent workflows provide separate signals for build confidence and
distribution readiness:

| Workflow | State it represents |
| --- | --- |
| [Agent verification (Linux)](../../.github/workflows/agent-ci.yml) | Linux agent contracts, Cozo integration, and Kubernetes sandbox verification |
| [Agent verification (sanitizers)](../../.github/workflows/agent-analysis.yml) | Linux AddressSanitizer and UndefinedBehaviorSanitizer checks |
| [Agent development package (.tar.gz)](../../.github/workflows/agent-package-archive.yml) | Verified development archive from the Linux staging package |
| [Agent development package (Debian/Ubuntu)](../../.github/workflows/agent-package-debian.yml) | Native Ubuntu/Debian package after the verified development archive |
| [Agent development package (Windows)](../../.github/workflows/agent-package-windows.yml) | Windows CPU dev ZIP/MSI after Windows package tests and staging validation |
| [Agent release packages](../../.github/workflows/agent-release.yml) | Versioned CPU, CUDA, and Vulkan packages from `agent-v*` tags |
| [Agent container images](../../.github/workflows/agent-container-images.yml) | Development image from the `.tar.gz` package and backend images from releases |

Workflow badges show the latest workflow result for the selected ref. The
evidence-based milestone record is maintained in
[Agent Assurance](agent-assurance.md).

## Distribution

- [Agent releases](https://github.com/peterkallden/llama/releases) contain
  versioned CPU, CUDA, and Vulkan packages published from `agent-v*` tags.
- [Development `.tar.gz` packages](https://github.com/peterkallden/llama/actions/workflows/agent-package-archive.yml)
- [Debian development packages](https://github.com/peterkallden/llama/actions/workflows/agent-package-debian.yml)
- [Windows development packages](https://github.com/peterkallden/llama/actions/workflows/agent-package-windows.yml)
  are dated workflow artifacts for testing non-master development branches.
- [Container images](https://github.com/peterkallden/llama/pkgs/container/llama-agent)
  are published to GitHub Container Registry. Pull a release image with:

  ```bash
  docker pull ghcr.io/peterkallden/llama/llama-agent:cpu-amd64-latest
  docker pull ghcr.io/peterkallden/llama/llama-agent:cuda-amd64-latest
  docker pull ghcr.io/peterkallden/llama/llama-agent:vulkan-amd64-latest
  ```

  The development CPU image is published separately:

  ```bash
  docker pull ghcr.io/peterkallden/llama/llama-agent-dev:cpu-amd64-latest
  ```

  Podman uses the same image names; replace `docker pull` with `podman pull`.

## Quick start

For prerequisites, supported build profiles, environment setup, CMake options,
and CTest labels, see [Agent Build](agent-build.md). A first daemon
configuration can be generated with
[`scripts/agent-config-bootstrap.sh`](../../scripts/agent-config-bootstrap.sh):

```bash
./scripts/agent-config-bootstrap.sh \
  --model models/model.gguf \
  --embedding-model models/embedding.gguf \
  --cozo-root data \
  --output agent-daemon-config.json
```

The daemon and its JSONL control plane are described in
[Agent Daemon Usage](agent-daemon-usage.md). That guide also covers
persistent stores, model mounts, daemon logs, authentication, tool exposure,
remote MCP providers, status/readiness, and Docker usage.

## Runtime areas

- **Runtime orchestration** — sessions, turns, thinking modes, reflection,
  deliberate planning, research, cancellation, deadlines, and bounded async
  execution. See [Agent Runtime](agent-runtime.md).
- **Tools and capabilities** — host-owned profiles, local tools, diagnostics,
  repository/workspace operations, data tools, resources, and optional remote
  MCP providers.
- **Memory, plans, and resources** — persistent Cozo-backed stores and a
  resource boundary for turn inputs, generated artifacts, and resource
  references.
- **Multimodal runtime** — staged native image/audio support through the
  server-context backend, with host-owned resource resolution and OCR/page-
  image fallback. See [Agent multimodal runtime](agent-multimodal.md).
- **Daemon and protocols** — foreground JSONL administration, TCP and Unix
  transports, inbound MCP, readiness/status reporting, and systemd examples.
- **Sandbox workspaces** — controlled Docker/Kubernetes execution and artifact
  handling. See [Agent Sandbox Workspaces](agent-sandbox-workspaces.md).

## MCP integration

The runtime can expose approved agent tools to MCP clients and can connect to
approved external MCP providers. Stdio, HTTP, JSONL, authentication,
delegation, and provider configuration are covered by
[Agent Daemon Usage](agent-daemon-usage.md). The bounded agent-to-agent MCP
design is documented in the
[Remote MCP Scheduler Plan](agent-remote-mcp-scheduler-plan.md).

## Configuration and examples

Copyable configurations and protocol fixtures are available in
[`docs/examples`](../examples):

- [`agent-config.example.json`](../examples/agent-config.example.json)
- [`agent-host-config-capabilities.json`](../examples/agent-host-config-capabilities.json)
- [`agent-host-config-remote-http.json`](../examples/agent-host-config-remote-http.json)
- [`agent-host-config-stdio.json`](../examples/agent-host-config-stdio.json)
- [`agent-host-config-jsonl-tcp.json`](../examples/agent-host-config-jsonl-tcp.json)
- [`agent-host-config-jsonl-unix.json`](../examples/agent-host-config-jsonl-unix.json)
- [`agent-daemon-requests.jsonl`](../examples/agent-daemon-requests.jsonl)
- [`llama-agent-daemon.service`](../examples/llama-agent-daemon.service)

The packaged Docker image uses persistent mounts for `/models`,
`/etc/llama-agent`, `/var/lib/llama-agent/data`, and `/var/log/llama-agent`.
The entrypoint bootstraps `/etc/llama-agent/config.json` only when it is
missing. Published images are
available through the repository's [Packages](https://github.com/peterkallden/llama/pkgs/container/llama-agent)
page when the corresponding workflows have completed successfully.

## Testing and assurance

Agent verification is split by dependency and execution cost:

- model-free contract and protocol tests use the `agent` CTest label;
- Docker and Kubernetes tests use their dedicated sandbox labels;
- model-backed resident and end-to-end checks are separate from contract
  assurance;
- release packaging runs only after the configured release checks pass.

See [Agent Assurance](agent-assurance.md) for recorded results and known
limitations, and [Agent Build](agent-build.md) for reproducible local and
CI-compatible commands.
