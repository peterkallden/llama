# llama-agent

[![Agent CI](https://github.com/peterkallden/llama/actions/workflows/agent-ci.yml/badge.svg?branch=feature%2Fllama-agent)](https://github.com/peterkallden/llama/actions/workflows/agent-ci.yml)
[![Dynamic analysis](https://github.com/peterkallden/llama/actions/workflows/agent-dynamic-analysis.yml/badge.svg?branch=feature%2Fllama-agent)](https://github.com/peterkallden/llama/actions/workflows/agent-dynamic-analysis.yml)
[![Development package](https://github.com/peterkallden/llama/actions/workflows/agent-package.yml/badge.svg?branch=feature%2Fllama-agent)](https://github.com/peterkallden/llama/actions/workflows/agent-package.yml)
[![Docker images](https://github.com/peterkallden/llama/actions/workflows/agent-docker-image.yml/badge.svg?branch=feature%2Fllama-agent)](https://github.com/peterkallden/llama/actions/workflows/agent-docker-image.yml)
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
| [Agent CI](../../.github/workflows/agent-ci.yml) | Linux agent contracts, Cozo integration, and Kubernetes sandbox verification |
| [Dynamic analysis](../../.github/workflows/agent-dynamic-analysis.yml) | Linux AddressSanitizer and UndefinedBehaviorSanitizer checks |
| [Development package](../../.github/workflows/agent-package.yml) | Verified development package from non-master branches |
| [Windows development package](../../.github/workflows/agent-windows-dev-package.yml) | Windows CPU dev ZIP/MSI from a tested per-user-oriented staging tree |
| [Agent release](../../.github/workflows/agent-release.yml) | Versioned CPU, CUDA, and Vulkan packages from `agent-v*` tags |
| [Docker images](../../.github/workflows/agent-docker-image.yml) | Dev image from a development package and backend images from releases |

Workflow badges show the latest workflow result for the selected ref. The
evidence-based milestone record is maintained in
[Agent Assurance](agent-assurance.md).

## Distribution

- [Agent releases](https://github.com/peterkallden/llama/releases) contain
  versioned CPU, CUDA, and Vulkan packages published from `agent-v*` tags.
- [Development packages](https://github.com/peterkallden/llama/actions/workflows/agent-package.yml)
- [Windows development package](https://github.com/peterkallden/llama/actions/workflows/agent-windows-dev-package.yml)
  are dated workflow artifacts for testing non-master development branches.

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
