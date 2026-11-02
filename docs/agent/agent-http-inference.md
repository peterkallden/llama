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

Set `tool_output_format` explicitly for the remote profile. The example uses native tool calls and requires `supports_native_tool_calls`; `jsonl` is another option for textual tool calls. The inherited default may be `compact_dsl`, whose grammar-constrained planning requests are unsupported by this HTTP transport.

## Supported behavior

- Text chat through complete, non-streaming OpenAI-compatible chat-completion responses.
- JSON-schema-constrained output and native function tool calls only when their profile flags are explicitly enabled. The agent still validates structured results and authorizes/binds tool calls locally.
- Existing scalar `generate_batch` fallback. This does not imply physical batching by the remote server.
- Existing daemon operations, planning, tool execution, MCP, memory, graph, and resource services remain local to the agent host.

GBNF grammar requests, images/audio, remote embeddings, local adapters/overlays, FlyDelta hidden-state capture and interventions, device diagnostics, residual patching, and teacher-forced scoring are unsupported by this transport. A request requiring an unsupported feature fails explicitly; it is not silently weakened. There is no streaming of generated text to the client and no automatic retry after an uncertain transport failure.

An enabled daemon FlyDelta server binding requires a resident `server-context` model. For a daemon whose selected generation profile is HTTP, set `runtime.adaptation.flydelta.enabled` to `false`; ordinary planning and tool execution remain available.

Timeouts, server rejection, malformed responses, schema violations, context limits, and pre-submit cancellation fail the inference request with an error. A synchronous in-flight HTTP operation cannot be interrupted by the current transport; its configured request timeout bounds waiting.
