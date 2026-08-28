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

Family descriptions belong in the main configuration because a family may
contain tools from several providers:

```json
{
  "tools": {
    "families": {
      "eurostat": {
        "description": "Query Eurostat statistical datasets and JSON-stat data"
      }
    }
  }
}
```

The family id is derived from the tool namespace (for example,
`eurostat.getData` becomes `eurostat`). This section changes only the compact
family preflight text; it does not create tools, grant capabilities or replace
the operation-specific tool contract. Descriptions are single-line and
bounded to 240 characters. The bootstrap script accepts repeated
`--family-description ID=TEXT` options for generating this section.

The family descriptions are part of the host-owned tooling snapshot. A JSONL
`reload_config` validates them, but changing `tools.families` is reported as
`restart_required`; restart the daemon before the new text is used for model
preflight. This avoids changing the model-facing tool snapshot halfway through
an active session.

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
## Provider-helper

För en enskild provider kan `scripts/agent-provider-bootstrap.sh` skapa ett
fragment utan att secrets hamnar i JSON. OpenAPI kräver en base URL och kan få
en lokal spec med `--spec`:

```sh
scripts/agent-provider-bootstrap.sh \
  --type openapi --id sales --base-url https://sales.example.test \
  --spec examples/sales-openapi.json \
  --auth-type bearer --token-env SALES_API_TOKEN \
  --output config/providers/sales.json
```

Om `--spec` utelämnas försöker helpern i ordning `/openapi.json`,
`/swagger.json` och `/api-docs`, sparar det första giltiga JSON-dokumentet med
`--spec-output` (eller `<id>-openapi.json`) och skriver den upptäckta adressen
till stderr. Discovery följer aldrig redirects.

Ett HTTP-MCP-fragment skapas exempelvis så här:

```sh
scripts/agent-provider-bootstrap.sh \
  --type mcp --id github --transport streamable_http \
  --url https://mcp.example.test/mcp --server-name github \
  --auth-type bearer --token-env GITHUB_MCP_TOKEN \
  --allowed-tool search_issues --output config/providers/github.json
```

Lägg sedan katalogen på `tools.include_dir` i huvudkonfigurationen. Helpern
bygger bara hostens providerobjekt; den ändrar inte det modellvända
toolkontraktet och den gör ingen implicit tool-allowlist för OpenAPI.
