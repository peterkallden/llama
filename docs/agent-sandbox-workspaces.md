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
      "executable": "docker",
      "default_image": "llama-agent-dev:latest"
    },
    "workspace": {
      "root": "C:/agent-workspaces",
      "artifact_root": "C:/agent-artifacts"
    },
    "classes": {
      "developer-build": {
        "timeout_ms": 120000,
        "memory_bytes": 8589934592,
        "cpu_count": 4,
        "network": "none",
        "filesystem": "workspace_write",
        "allow_artifacts": true
      }
    }
  }
}
```

The Docker backend runs each operation as an ephemeral container with a
read-only root filesystem, no network by default, bounded resource limits,
and only the host-created workspace mounts. The `none` backend remains useful
for hosts that only want validation: it returns `sandbox.backend_unavailable`
instead of falling back to an unsandboxed process. Broader network scopes are
rejected by the Docker backend until explicit allowlists are implemented.

## Semantic developer tools

`build_target` and `test_run` create bounded `developer-build` requests. The
native adapter emits semantic commands such as `agent.build_target` and
`agent.test_run`; Docker translates those requests into container execution
details. Kubernetes will use the same semantic contract later.

Operation directories are host-created and may be ephemeral. Source is
normally read-only, the writable directory is used for build/analysis work,
and artifacts are returned through explicit artifact/resource references.
