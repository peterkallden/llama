# LXC/Incus profiles for llama-agent

These files are operator-managed examples for the LXC/Incus sandbox backend.
They are not agent configuration files and they are not selected by the model.
The agent receives only the profile name and the explicitly declared network
scope from host configuration.

The `network-none` profile is the safe starting point. It deliberately has no
network device; the runtime adds only the workspace mounts it needs. Use it
with:

```bash
lxc profile create llama-agent-network-none
lxc profile edit llama-agent-network-none < lxc-profile-network-none.yaml
lxc profile show llama-agent-network-none
```

For Incus, replace `lxc` with `incus`. The profile name must match the value
passed to `--lxc-network-profile` or `lxc.network_profile`.

The restricted-network file is a template only. A NIC plus a descriptive
profile name does not enforce DNS-only, allowlisted, package-registry or web
access. Those scopes require an independently verified host bridge, firewall,
proxy or network policy. Do not configure a broader
`lxc_network_profile_scope` until that enforcement has been tested on the
target host. A profile name alone never grants a capability.

The runtime applies CPU, memory and process limits separately with LXC
instance limits. Do not add workspace disks to these profiles: the runtime
creates the source, writable and artifact mounts for each execution.

