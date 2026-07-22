# Agent sandbox workspaces

Sandbox execution is host-owned. Tools select a semantic execution class and
provide bounded arguments; they do not select host paths, containers or shell
commands.

## Shared workspace identity

Research, developer and data-analysis work share the same logical identity:

```text
namespace / project / session / turn
        |
        +-- workspace context
              |
              +-- ephemeral operation
                    +-- source
                    +-- writable
                    +-- artifacts
```

Research keeps its own gaps, sources and evidence state. A research workspace
can be adapted to the shared workspace context, and `resource://` inputs can
be materialized into an operation source directory through host resource
authority.

## Host configuration

Workspace roots and execution classes belong in host configuration:

```json
{
  "sandbox": {
    "backend": "docker",
    "docker": {
      "default_image": "llama-agent-dev:latest"
    },
    "workspace": {
      "root": "C:/agent-workspaces",
      "artifact_root": "C:/agent-artifacts",
      "operation_mode": "ephemeral",
      "project_mode": "persistent"
    },
    "defaults": {
      "timeout_ms": 60000,
      "cpu_count": 1,
      "max_output_bytes": 65536,
      "network": "none",
      "filesystem": "readonly",
      "allow_artifacts": true
    },
    "classes": {
      "developer-build": {
        "image": "llama-agent-dev:latest",
        "timeout_ms": 1200000,
        "cpu_count": 4,
        "memory_bytes": 8589934592,
        "filesystem": "workspace_write"
      }
    }
  }
}
```

The values in `sandbox.defaults` are copied to every execution class when the
host configuration is loaded. A class may override only the values it needs.
An execution class may provide its own image; if it is omitted, the Docker
backend uses `sandbox.docker.default_image`. Kubernetes uses the image from
the selected execution class or default policy. The request itself may not
override this host-owned selection. A value of zero is invalid for CPU,
timeout and output limits; zero means no configured limit only for memory and
process count.

The Docker backend runs each operation as an ephemeral container with a
read-only root filesystem, no network by default, bounded resource limits,
and only the host-created workspace mounts. The `none` backend remains useful
for hosts that only want validation: it returns `sandbox.backend_unavailable`
instead of falling back to an unsandboxed process. Broader network scopes are
rejected by the Docker backend until explicit allowlists are implemented.

## Kubernetes backend

The Kubernetes backend creates one ephemeral `batch/v1 Job` per operation and
waits for completion through `kubectl`. Its host configuration is:

```json
{
  "sandbox": {
    "backend": "kubernetes",
    "kubernetes": {
      "executable": "kubectl",
      "namespace": "llama-agent-jobs",
      "service_account": "llama-agent-runner",
      "runtime_class": "standard",
      "cleanup": true
    }
  }
}
```

The first slice uses `hostPath` volumes for the logical source, writable and
artifact mounts. This is suitable for a local Kubernetes installation where
the workspace is visible on the worker node. A remote cluster needs a shared
volume or object-store materializer before these mounts can be used safely.
The backend currently supports `network: none` by applying a deny-egress
`NetworkPolicy` alongside the Job; the cluster's CNI must enforce
NetworkPolicy. It also applies non-root and read-only-rootfs security settings,
collects logs and artifacts, enforces the operation timeout, and deletes the
Job and policy when cleanup is enabled.

## Semantic developer tools

`build_target` and `test_run` create bounded `developer-build` requests. The
native adapter emits semantic commands such as `agent.build_target` and
`agent.test_run`; Docker translates those requests into container execution
details. Kubernetes translates the same semantic request into a Job.

Operation directories are host-created and may be ephemeral. Source is
normally read-only, the writable directory is used for build/analysis work,
and artifacts are returned through explicit artifact/resource references.

The workspace modes are host policy, not tool arguments. `operation_mode:
ephemeral` gives each operation its own generated directory; `project_mode:
persistent` keeps the project identity stable across turns while still using
separate operation directories. Build and test requests may carry bounded
`resource_refs`; the helper resolves those through the host resource store and
materializes them under the operation's `source/` directory before execution.
