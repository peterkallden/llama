# HTTP inference profiles

The daemon may run its normal agent runtime on one host while sending text generation to a remote `llama-server`. Planning, tool authorization and execution, MCP providers, memory, the project graph, resource storage, and session state remain on the daemon host. The remote server receives the model-facing messages, declared tool schemas, and generated tool results included in later messages. Do not place confidential prompts or tool output on a remote endpoint unless its operator and transport are trusted. Prefer HTTPS or a private network.

Local `cli` and `server-context` profiles remain the default. HTTP is selected only by an explicit model profile; there is no automatic fallback between local and remote inference.

## Example

On the model host, start llama-server on an address reachable from the agent host:

```sh
llama-server -m /models/instruct.gguf -c 32768 --host 0.0.0.0 --port 8080
```

Use firewall rules or a private network to restrict access. If the server requires bearer authentication, set the variable on the agent host before starting the daemon. Only the variable name is stored in the host configuration; its value is not serialized, included in profile identity, or logged.

```sh
export LLAMA_SERVER_TOKEN='...'
```

Put this object in the daemon host configuration's top-level `models` field. Leave `model.profile` unset to use `routing.default_profile`, or set it to `remote-agent` explicitly:

```json
{
  "schema_version": 1,
  "directory": "",
  "bases": {
    "remote-generation": {
      "kind": "generation",
      "backend": "http",
      "endpoint": "https://model-host.example:8080",
      "model_id": "local-model-alias",
      "token_env": "LLAMA_SERVER_TOKEN",
      "connect_timeout_ms": 5000,
      "request_timeout_ms": 120000,
      "supports_json_schema": true,
      "supports_native_tool_calls": true,
      "load": "lazy"
    }
  },
  "profiles": {
    "remote-agent": {
      "base": "remote-generation",
      "context_size": 32768,
      "tool_output_format": "native",
      "n_parallel": 1,
      "n_sequences": 1,
      "load": "lazy"
    }
  },
  "routing": {
    "default_profile": "remote-agent"
  },
  "limits": {
    "max_loaded_generation_models": 1,
    "model_eviction": "lru"
  }
}
```

`endpoint` is the server base URL; the adapter posts to `/v1/chat/completions` (including any configured path prefix). `model_id` is sent in each request. `context_size` is required for the profile and is the capacity used by agent budgeting; configure the server's context capacity to match or exceed it. Exact prompt-token counting is unavailable for HTTP, so the agent uses its conservative budget and reports context errors returned by the server.

Set `tool_output_format` explicitly for the remote profile. The example uses native tool calls and requires `supports_native_tool_calls`; `jsonl` is another option for textual tool calls. The inherited default may be `compact_dsl`; HTTP uses the text fallback described below when the server lacks GBNF.

An HTTP model base may set `"map_tool_names": true` for servers that restrict function names to ASCII letters, digits, underscores and hyphens. The adapter escapes other bytes, including underscores, so names remain distinct; for example, `memory.search` becomes `memory_2esearch` and `memory_search` becomes `memory_5fsearch`. Names exceeding 64 characters after encoding are rejected. Historical tool calls use the same mapping, and returned names must match a tool declared in the current request. This option only maps names: it does not enable OpenAI-specific generation parameters or establish compatibility with a particular model.

For a server that requires OpenAI-style generation parameters, a base may additionally specify `"generation": {"token_limit_parameter": "max_completion_tokens", "reasoning_effort": "none"}`. The agent's per-request `n_predict` supplies the limit; it is not a second independent budget. The default is `max_tokens` and no reasoning-effort field. Supported reasoning values are `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, and `max`; actual availability depends on the server and model. `max_completion_tokens` may include non-visible generated tokens. These options do not imply that a model supports tool calling at every reasoning effort.

For compact DSL with an HTTP server that lacks GBNF, supported DSL selection, planning and reflection requests use plain text with host-side parsing and validation. The HTTP backend logs each fallback; no extra setting is needed. This does not add grammar support to the server or permit arbitrary grammar requests. Other grammar-dependent flows still report unsupported. Invalid DSL is rejected by the existing caller and may be regenerated within its normal budget.

When a server's strict schema subset cannot represent a required agent schema, explicitly set `"generation": {"strict_json_schema": false}` alongside any other generation options. This applies to all JSON-schema requests for the base: the server may generate non-conforming output, which the agent still validates against the original schema locally. The default remains strict; this setting does not weaken local plan, tool authorization, or argument validation.

## Supported behavior

- Text chat through complete, non-streaming OpenAI-compatible chat-completion responses.
- JSON-schema-constrained output and native function tool calls only when their profile flags are explicitly enabled. The agent still validates structured results and authorizes/binds tool calls locally.
- Existing scalar `generate_batch` fallback. This does not imply physical batching by the remote server.
- Existing daemon operations, planning, tool execution, MCP, memory, graph, and resource services remain local to the agent host.

GBNF grammar enforcement, images/audio, remote embeddings, local adapters/overlays, FlyDelta hidden-state capture and interventions, device diagnostics, residual patching, and teacher-forced scoring are unsupported by this transport. Grammar requests fail unless they use the explicitly logged DSL text fallback for a locally validated operation. There is no streaming of generated text to the client and no automatic retry after an uncertain transport failure.

An enabled daemon FlyDelta server binding requires a resident `server-context` model. For a daemon whose selected generation profile is HTTP, set `runtime.adaptation.flydelta.enabled` to `false`; ordinary planning and tool execution remain available.

Timeouts, server rejection, malformed responses, schema violations, context limits, and pre-submit cancellation fail the inference request with an error. A synchronous in-flight HTTP operation cannot be interrupted by the current transport; its configured request timeout bounds waiting.
