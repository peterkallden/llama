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
    "lxc": {
      "executable": "lxc",
      "default_image": "ubuntu:24.04",
      "network_mode": "none",
      "network_profile": "llama-agent-network-none",
      "network_profile_scope": "none",
      "cleanup": true
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

The Docker-compatible backend runs each operation as an ephemeral container with a
read-only root filesystem, no network by default, bounded resource limits,
and only the host-created workspace mounts. Its executable defaults to `docker`;
Podman can be selected as a host-owned compatible executable without changing
the backend contract. The `none` backend remains useful
for hosts that only want validation: the host omits sandbox-backed tools from
the effective tool view, while direct sandbox execution returns
`sandbox.backend_unavailable` instead of falling back to an unsandboxed
process. Broader network scopes are
rejected by the Docker-compatible backend until explicit allowlists are implemented.

## Backend capabilities and LXC

`docker`, `kubernetes` and `lxc` are sandbox backends; `none` is an explicit
unavailable state and never means “run locally”. Each backend advertises the
capabilities it can enforce: process isolation, filesystem scope, resource
limits, artifact collection and network scope. The host validates the requested
policy against that capability declaration and fails closed when it cannot be
honoured.

LXC is an optional lightweight fallback for Linux hosts with LXC/Incus. It
creates an ephemeral container, mounts the host-created source, writable and
artifact directories at the standard workspace paths, and removes the container
after completion when `cleanup` is enabled. LXC limits are applied explicitly
with `limits.cpu`, `limits.memory` and `limits.processes`; if LXC rejects one of
those settings, execution fails closed instead of reporting an unenforced
limit.

LXC also requires an operator-managed profile for every execution. The profile
must be named in `network_profile`, and `network_profile_scope` must describe
the scope that profile actually enforces: `none`, `dns_only`, `allowlisted`,
`package_registry` or `research_web`. A profile name by itself grants no
network capability. The default example uses a profile that removes network
access. A profile that permits network access must be configured with the
matching declared scope; the model and tool cannot select or broaden it. This
is an operator assertion about a separately managed LXC/Incus profile, not a
claim that LXC provides Kubernetes NetworkPolicy semantics.

Docker and Kubernetes advertise only capabilities backed by their generated
runtime arguments/manifests. Docker currently supports `network=none`; its
CPU, memory and process limits map to `--cpus`, `--memory` and `--pids-limit`.
Kubernetes currently supports `network=none`; its Job manifest sets CPU and
memory limits, read-only/security constraints and a deny-egress NetworkPolicy.
It does not advertise a per-pod process limit because `podPidsLimit` is a
kubelet/node setting outside this runtime. Broader network scopes remain
unavailable for those backends until explicit allowlist enforcement exists.

The general [agent-config.example.json](../examples/agent-config.example.json)
shows all backend configuration sections while keeping the selected backend
as `none`. The focused
[agent-host-config-capabilities.json](../examples/agent-host-config-capabilities.json)
selects Docker and therefore demonstrates `memory_bytes` and `process_count`.
Do not copy that process limit unchanged to a Kubernetes configuration; the
host will reject it because Kubernetes does not advertise that capability.
Importable LXC/Incus profile examples are in
[examples/lxc-profiles](../examples/lxc-profiles/README.md). The networkless
profile is the ready-to-use example; broader network profiles remain templates
until their host-side enforcement has been verified.

Processor execution modes have deterministic fallback semantics. With
`backend: auto`, `local_preferred` tries local then sandbox, while
`sandbox_preferred` tries sandbox then local. `local_required` and
`sandbox_required` fail rather than cross the requested trust boundary. An
explicit `backend` such as `lxc` restricts the choice to that backend.

## Kubernetes backend

The Kubernetes backend creates one ephemeral `batch/v1 Job` per operation and
waits for completion through `kubectl`. Its host configuration is:

```json
{
  "sandbox": {
    "backend": "kubernetes",
    "kubernetes": {
      "executable": "kubectl",
      "kubeconfig": "",
      "context": "",
      "insecure_skip_tls_verify": false,
      "namespace": "llama-agent-jobs",
      "service_account": "llama-agent-runner",
      "runtime_class": "standard",
      "storage_class": "",
      "workspace_storage_size": "4Gi",
      "artifact_storage_size": "1Gi",
      "staging_image": "alpine:3.20",
      "pvc_retention": "project",
      "staging_timeout_ms": 120000,
      "cleanup": true
    }
  }
}
```

The Kubernetes backend uses project- or session-scoped PVCs for the logical
source, writable and artifact mounts. A staging pod copies the host-created
operation into the workspace claim and copies artifacts back after the Job.
The default claim sizes are `4Gi` for workspace data and `1Gi` for artifacts;
an empty `storage_class` uses the cluster default. PVCs are retained across
operations and are not deleted by normal Job cleanup.
The Kubernetes executable, namespace, service account, runtime class, storage
class, claim sizes, staging image, kubeconfig/context, TLS verification and
cleanup/retention policy are host-owned and cannot be selected by a client or
tool request. `insecure_skip_tls_verify` is intended only for isolated local
testing and must remain `false` in normal deployments.

The current backend does not require a Helm chart. It creates an ephemeral Job
for each processor or sandbox operation, so the processor worker is supplied as
an image reference on the host-owned policy for that operation. The worker
image and `staging_image` have different responsibilities: the worker image
contains the processor implementation and its runtime dependencies, while the
staging image only copies workspace data into or out of the PVC-backed
operation workspace. A local Docker Desktop image such as
`llama-agent-pdf-worker:local` is sufficient for the local Kubernetes smoke;
remote clusters should use an image available in their registry, preferably
addressed by an immutable digest.

Helm becomes useful if the deployment later needs a repeatable installation of
cluster-scoped or long-lived resources, such as namespaces, service accounts,
RBAC, storage defaults, NetworkPolicies, registry configuration, or a resident
worker/controller. Those packaging concerns must remain outside the generic
resource-processor contract. Introducing a chart is not required for the
current ephemeral Job model and must not create a second scheduler or worker
lifecycle.

PVC names, `ReadWriteOnce`, operation `subPath` values and the container paths
`/workspace/source`, `/workspace/writable` and `/workspace/artifacts` remain
backend implementation details.
The backend currently supports `network: none` by applying a deny-egress
`NetworkPolicy` alongside the Job; the cluster's CNI must enforce
NetworkPolicy. It also applies non-root and read-only-rootfs security settings,
collects logs and artifacts, enforces the operation timeout, and deletes the
Job and policy when cleanup is enabled.

## Semantic developer tools

development.build and development.test create bounded developer-build requests.
The model supplies a logical target identifier, never a command line. The host resolves
that identifier through sandbox.development_targets and builds a fixed argv vector for
the configured sandbox. Configuration names and test-filter names are accepted only when
the selected target profile defines them.

Example host configuration:

    {
      "sandbox": {
        "development_targets": {
          "agent-build-pack": {
            "build_program": "cmake",
            "build_arguments": ["--build", "/workspace/writable/build", "--target", "llama-agent-build-pack"],
            "test_program": "ctest",
            "test_arguments": ["--test-dir", "/workspace/writable/build", "--output-on-failure"],
            "working_directory": "/workspace/source",
            "configurations": {"Release": ["--config", "Release"]},
            "test_filters": {"agent-contracts": ["-L", "agent"]},
            "max_timeout_ms": 1200000
          }
        }
      }
    }

Each target is an operator-owned allowlist entry. Arguments are passed directly to the
configured process runner; no shell is involved. A missing target, configuration, filter
or operation fails closed. Images must provide the configured executables. A Docker or
Kubernetes backend does not invent a project-specific command translation.

Execution operation IDs are unique per invocation. The workspace manager creates an
operation directory and materializes explicitly supplied resource inputs. A
`repository.snapshot` resource uses `code-source-snapshot.v1`: the manifest points to
immutable, scoped resources containing captured file bytes, modes and SHA-256 hashes.
When a complete snapshot is passed in `source_snapshot_ref`, the host verifies every file entry
and materializes the exact bytes beneath `/workspace/source` before running the configured
target. The execution receipt records the snapshot URI and manifest hash. Incomplete,
escaping, duplicate, oversized or hash-mismatched snapshots are rejected before execution.
The snapshot is bounded to 8,192 files, 64 MiB per file and 256 MiB total. Symlinks,
submodules and unreadable files make it incomplete; it cannot be used for source-bound
execution. Tracked paths that are intentionally absent are listed as `deleted_paths`
and omitted from the staged tree; they do not make a snapshot incomplete.
With resource storage configured, receipts and bounded stdout/stderr excerpts are retained
as scoped resources. Existing storeless profiles can still run without resource inputs,
but do not produce durable evidence. Requested inputs fail before execution when the
resource store is unavailable. Output
retention failures are explicit. Tool results include short output previews and references
to retained logs, and preserve backend artifact references even on failure. Receipt and
log names are content-addressed so repeated operation IDs cannot overwrite earlier evidence.
After a source snapshot is staged, additional resource inputs cannot overwrite its files.
Generic resource-processing retries retain their existing behavior. File-resource size is checked before
bounded reads, preventing a resource prefix from being accepted as a complete source file.
The host records the selected logical target, configuration, test filter, configured
executable, status, timing, usage, artifact references and output hashes; it does not
retain command credentials or raw command argument vectors.

The binding is exact for the immutable staged source tree, not for external dependencies,
compiler images, generated inputs or the host toolchain. Those are covered only when the
operator-owned profile and runner provide that provenance. A successful build is not
evidence that requested acceptance criteria passed; test outcomes remain separate
verification receipts. A target that does not consume `/workspace/source` is not source-
bound even when a snapshot was staged, so profiles must be reviewed accordingly.

For mutations, `workspace.patch` can take `source_snapshot_ref`. The host checks that the
current exact file hash still matches the selected snapshot before applying the patch. It
then returns a `workspace-change-receipt.v1` resource with the snapshot reference,
expected/before/after hashes, requested line operations and applied/unchanged status. A
stale snapshot is rejected even if a caller supplies the current file hash; the workflow
must capture a new baseline and review the changed source first. After the last patch,
capture a separate post-change snapshot for build/test execution. Reusing the pre-change
snapshot would verify the old source rather than the patch.

CTest och smoke

The backend checks are split into a model-free contract smoke and backend
smokes:

```powershell
ctest --test-dir build-agent-tool-profiles-debug -C Debug -R sandbox-contract
ctest --test-dir build-agent-tool-profiles-debug -C Debug -L sandbox-docker
ctest --test-dir build-agent-tool-profiles-debug -C Debug -L sandbox-kubernetes
```

The Docker test is skipped when Docker is not reachable. The Kubernetes test
always runs its contract checks, but only creates a real Job when explicitly
enabled:

```powershell
$env:LLAMA_AGENT_KUBERNETES_SMOKE = "1"
$env:LLAMA_AGENT_KUBERNETES_IMAGE = "alpine:3.20"
$env:LLAMA_AGENT_KUBERNETES_NAMESPACE = "llama-agent-jobs"
ctest --test-dir build-agent-tool-profiles-debug -C Debug -L sandbox-kubernetes
```

The real Kubernetes smoke assumes that `kubectl`, the configured namespace,
the image, a usable StorageClass and a trusted kubeconfig are available to the
current host.

Operation directories are host-created and may be ephemeral. Source is
normally read-only, the writable directory is used for build/analysis work,
and artifacts are returned through explicit artifact/resource references.

The workspace modes are host policy, not tool arguments. `operation_mode:
ephemeral` gives each operation its own generated directory; `project_mode:
persistent` keeps the project identity stable across turns while still using
separate operation directories. Build and test requests may carry bounded
`resource_refs`; the helper resolves those through the host resource store and
materializes them under the operation's `source/` directory before execution.
