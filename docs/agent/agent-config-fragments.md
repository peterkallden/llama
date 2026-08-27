# Agent configuration fragments

The daemon accepts one main JSON configuration file. Provider definitions may
also be split into a directory using `tools.include_dir`:

```json
{
  "schema_version": 1,
  "model": {"path": "models/model.gguf"},
  "tools": {"include_dir": "providers.d"}
}
```

The directory is resolved relative to the main configuration file. Every
regular file whose name ends in `.json` is read in lexical filename order. A
fragment contains exactly one provider object, using the same contract as an
entry in `tools.providers`:

```json
{
  "type": "mcp",
  "id": "github",
  "transport": "stdio",
  "command": ["/opt/tools/github-mcp"]
}
```

OpenAPI fragments use the provider contract documented in
[OpenAPI tool provider](agent-openapi-tool-provider.md). Inline providers and
directory providers may be used together. Provider IDs must be unique across
both sources. Non-JSON files are ignored, and nested include directories are
not followed.

The loader expands fragments into the normal in-memory provider catalog and
then validates the complete configuration. A missing directory, invalid JSON,
unsupported provider type, duplicate ID, or invalid provider definition
rejects the complete candidate. The previously active daemon configuration is
kept intact.

## Deployment and reload

The daemon does not watch the directory. Publish a new complete directory
snapshot and then issue the existing administrative reload command against the
main configuration. For deployment, write fragments to a staging directory and
atomically rename that directory into place before reloading. A transient
invalid snapshot is rejected and never partially applied.

Relative paths inside the current provider contract are relative to the main
configuration file. Secrets remain environment references such as `token_env`;
they must not be placed in fragments.

`agent_host_config_to_json()` emits a canonical effective configuration while
preserving `tools.include_dir` and does not duplicate providers loaded from
that directory. It is suitable for diagnostics, not as a replacement for the
source fragment layout.

This mechanism deliberately handles provider definitions first. It does not
perform a generic deep merge of arbitrary JSON objects. MCP tools are
discovered from their MCP server at runtime; an MCP fragment can narrow that
discovery with `allowed_tools`, but cannot define a remote MCP tool by itself.
