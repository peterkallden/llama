# Native dynamic plan and reflection PoC

## Quick start

Build the optional PoC once:

```powershell
cmake -S . -B build-plan -DLLAMA_MEMORY=ON -DLLAMA_PLAN=ON -DLLAMA_AGENT_REFLECTION=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build-plan --config Release
```

### 1. Bootstrap persistent memory and blueprints

Bootstrap is idempotent and happens as part of an agent run. With database paths present, the CLI automatically uses the compiled Cozo backends.

```powershell
.\build-plan\bin\Release\llama-agent.exe run `
  --memory-db .\work\agent-memory.cozo --plan-db .\work\agent-plan.cozo `
  --model .\models\chat.gguf --embedding-model .\models\embedding.gguf `
  --agent-bootstrap default --prompt "Set up the local agent workspace."
```

### 2. Run the agent

The default profile is the normal agent entry point: it plans, uses the scoped memory tools, reflects once, and can submit policy-gated memory proposals.

```powershell
.\build-plan\bin\Release\llama-agent.exe run `
  --memory-db .\work\agent-memory.cozo --plan-db .\work\agent-plan.cozo `
  --model .\models\chat.gguf --embedding-model .\models\embedding.gguf `
  --prompt "What did we decide, and what should happen next?"
```

Use the research profile when a task needs bounded repository inspection or public web research. `web_search` returns at most eight candidates from DuckDuckGo Lite; `web_fetch` extracts bounded text from a selected public HTTPS URL. They are read-only tools and remain subject to the normal per-turn tool-round budget.

```powershell
.\build-plan\bin\Release\llama-agent.exe run `
  --memory-db .\work\agent-memory.cozo --plan-db .\work\agent-plan.cozo `
  --model .\models\chat.gguf --embedding-model .\models\embedding.gguf `
  --agent-profile research `
  --prompt "Search for the llama.cpp repository, then fetch its README and summarize the build prerequisites."
```

### 3. Choose an agent profile

| Profile | Planning | Tools | Reflection | Post-turn learning |
| --- | --- | --- | --- | --- |
| `default` | `mini` | `memory` | `always` | off |
| `learning` | `mini` | `memory` | `always` | `post-turn` |
| `research` | `mini` | `research` | `always` | off |
| `safe` | `mini` | `memory-read` | off | off |
| `static` | off | none | off | off |

Use `--agent-profile research`, `--agent-profile learning`, or `--agent-profile safe` when the task calls for a different temperament. Durable post-turn learning is deliberately not part of `default`; `memory_remember` remains policy-gated in the profiles that expose it.

### Common variations

Use a reusable bootstrap blueprint explicitly, or let the selector choose one from the installed package:

```powershell
--agent-blueprint repository-change --plan-id change-42
--agent-blueprint auto --plan-id change-42
```

Use `--memory-scope session|project|global` and `--plan-scope turn|session|project|global` to choose lifecycle boundaries. `global` is intended for one local instance unless a tenant policy is added.

Use `--agent-plan auto` with a persistent plan store to let the bounded selector consider up to eight compatible active or blocked task plans. It may resume one only when its confidence reaches `0.75`; otherwise the runtime creates a new plan. An explicit `--plan-id` always takes precedence. Candidate filtering is native-owned and requires matching plan scope plus namespace and the applicable turn, session, or project identity.

Export the portable bootstrap definitions in the current scope without loading the chat model:

```powershell
.\build-plan\bin\Release\llama-agent.exe run `
  --memory-db .\work\agent-memory.cozo --plan-db .\work\agent-plan.cozo `
  --model unused.gguf --prompt export --agent-export .\work\agent-package.json
```

`--agent-export` writes the canonical package form containing scoped bootstrap `procedures` and `blueprints`; it excludes embeddings, runtime IDs, observations and event history. Blueprint purpose and non-default step contributions are retained as portable definition metadata. `--agent-import PATH` reads that same safe package form. The first export slice is intentionally limited to these portable definition types; broader memory/state migration and a full Cozo export/import round-trip test remain follow-up work.

## Advanced configuration and reference

This optional PoC builds a small control plane around existing llama.cpp inference helpers. It does not change GGML, `llama_decode`, model loading, sampling, the KV cache, or libllama.

```text
user request -> memory retrieval -> planner -> plan store
                                      |          |
                                      v          |
                                safe executor -> observation
                                      |          |
                                      v          |
                                  reflector -----+
                                      |
                                      v
                            final answer or one bounded revision
```

`llama-agent` is the preferred PoC entry point for agent turns. It owns the CLI parsing, profile resolution, planning, reflection, tool-profile and learning flow, while reusing the same memory store and chat helpers as the memory PoC. `llama-memory chat` remains as a compatibility wrapper that delegates to the same agent implementation; `llama-memory add/search/relate` remain memory-only operations.

`llama-memory` provides retrieved evidence only. A planner can propose typed plan operations, but `common_plan_policy` validates version checks, state transitions, dependencies, cycles, evidence requirements and limits before `common_plan_store` persists an update. Plan events are append-only short audit records. Plans are never memory records.

The first backend is `common_plan_in_memory_store`. When `LLAMA_PLAN_COZO=ON`, `common_plan_cozo_store` persists plan state in the separate Cozo relations `agent_plan` and `agent_plan_event`; it never uses the memory relations `memory` or `memory_edge`. The generic in-memory policy remains the mutation gate before an accepted state and its short audit event are persisted. If either persistence write fails, the cache is reloaded from Cozo; if the audit-event write fails after a state write, the prior state is restored before that reload.

### Tool catalog and bootstrap internals

`common_tool_catalog` is a declarative, versioned catalog of built-in capabilities. Its bootstrap is idempotent: it creates missing definitions and the standard profiles without rewriting existing entries. It deliberately contains metadata only (schemas, policy, limits, risk class and a native `executor_id`) and cannot load a DLL, script, command or other executable code. A future Cozo-backed catalog will persist these same records; the initial in-process catalog keeps the bootstrap semantics testable before introducing another database relation.

The separate agent bootstrap is also idempotent, but installs curated procedure memories and blueprint plans. It has a common logical JSON parser and serializer (`schema_version`, name, version, procedures and blueprints), including blueprint constraints and assumptions. The parser intentionally ignores unknown JSON fields for forward compatibility; it accepts only portable logical IDs and never imports persisted IDs, embeddings, task-plan state, observations or event history. The compiled default remains a small local fallback; `pocs/memory/bootstrap/default-v1.json` carries the equivalent starter definitions and can be loaded with `--agent-import PATH`. `--agent-export PATH` uses filtered store listing to recreate the same package form from scoped bootstrap definitions. Import and export currently cover `procedures` and `blueprints`; future memory or plan kinds require an explicit format and option.

The bootstrap profiles are `minimal`, `memory-read`, `memory` and `research`. The catalog includes read-only `calculator`, `time_now`, memory inspection/search, `plan_get`, repository inspection, and the bounded network-read tools `web_search` and `web_fetch`, alongside proposal-only memory/plan mutation definitions. `research` is the only built-in profile that exposes repository and network tools. Proposal tools require confirmation and remain subject to native memory or plan policy; an enabled catalog entry never grants execution authority by itself.

`common_register_native_tool_adapters` is the bridge from catalog to execution. It receives its memory store, memory scope, bound plan id and optional repository root from runtime-owned bindings. The current native set includes `calculator`, UTC `time_now`, scoped memory reads, `plan_get`, repository list/search/read/diff/log, the native `web_search` and `web_fetch` handlers, and the policy-gated `memory_remember` proposal. Other proposal tools remain unavailable even if their profile declares them.

`web_search` queries DuckDuckGo Lite over HTTPS and returns a bounded list of title, URL and snippet candidates. `web_fetch` accepts only HTTPS, rejects credentialed URLs and obvious local targets (`localhost`, `.local`, loopback and private literal IP ranges), follows the native client's redirects, and returns a capped HTML-to-text extraction with status and content type. It has a 10-second transport timeout; `web_search` caps the downloaded provider page at 128 KiB, and `web_fetch` accepts 1 to 500,000 input bytes. Neither tool accepts arbitrary headers, request methods, uploads, shell access, file access or a generic `curl` command. DNS resolution is not yet independently checked against private address ranges, so a future hardening slice should add resolved-address validation and redirect-by-redirect revalidation.

On Windows, these two agent network tools use WinHTTP and the Windows certificate store (`LLAMA_AGENT_WEB_USE_WINHTTP`, supplied by the agent CMake target). On Linux and other supported platforms they use the existing httplib transport and its configured OpenSSL-compatible TLS backend. This keeps the tool contract identical while using the platform's appropriate trust path.

`common_tool_profile_to_chat_tools` and `common_tool_dispatch_chat_calls` connect that native registry to llama.cpp's existing chat-template and tool-call parser layer. The former exposes only definitions that are both profile-approved and actually registered; the latter turns a parsed assistant tool call into a bounded `role: tool` message for the next generation. It assigns a stable runtime call id when a template omitted one, caps results, and never executes proposal or unregistered tools. Native handlers now return a structured `common_tool_execution_result` with `failure_code`, `failure_class`, `retryable`, `safe_summary`, and optional raw diagnostics. The chat bridge forwards only the safe structured fields into the tool message; raw diagnostics stay local to the caller. The first limit is one call per batch.

When built with `LLAMA_MEMORY=ON` and `LLAMA_AGENT_REFLECTION=ON`, `llama-agent run` exposes this path with `--tool-profile minimal|memory-read|memory|research`. It bootstraps the selected profile in process, binds memory scope and the optional embedding provider from CLI-owned runtime state, performs bounded registered-tool rounds, and completes with a tool-free final generation. `memory_remember` is available as a policy-gated proposal in `memory` and `research`; it is not an unrestricted write capability. The profile flag cannot be combined with the older `--memory-search-tool` or `--memory-remember-tool` flags. `llama-memory chat` accepts the same agent arguments for compatibility.

### Detailed overrides

`--agent-profile` is a convenience layer. An explicitly supplied `--tool-profile`, `--planning-mode`, `--reflection-mode` or `--memory-learn` overrides only that dimension of the chosen profile. This is useful for experiments such as `--agent-profile research --reflection-mode off`. Existing legacy memory-tool flags retain static behavior unless a named agent profile is explicitly selected, and they remain useful for simple non-planned chat tests.

### Model-backed chat loop

`llama-agent run --planning-mode mini` runs the bounded runtime around the already loaded chat model. The model-facing planner contract is deliberately compact: it returns `{goal,steps}`, where each step may include a `tool`, ordinary JSON `args`, optional `after` dependencies, an optional `id`, and a `mode`. It never has to JSON-encode an object inside `arguments_json`. Native code expands this compact proposal into the full internal operations, supplies IDs when they are omitted, supplies titles/objectives and empty metadata, normalizes safe shorthand such as one-string dependencies, infers a simple sequential dependency chain when `after` is omitted, and can add the final synthesis step automatically. The schema given to grammar-constrained local generation is intentionally narrower than the parser: it prefers `tool` objects and `after` arrays, while the parser still accepts safe shorthand for imported, persisted or older compact proposals.

### Purpose, goal and evidence

Each task plan has a caller-owned `purpose`, an executable `goal`, and bounded `success_criteria`. By default the user prompt supplies both purpose and desired outcome; an embedding API may instead provide `common_agent_objective` with an explicit purpose, desired outcome, criteria and constraints. The planner may refine the executable goal but cannot replace the caller-owned purpose. A compact proposal can describe a step's optional `contribution`; internally every step has an intended contribution, defaulting to its objective.

Before the scheduler marks a plan complete, native code evaluates it deterministically: mandatory non-final steps require a recorded result or matching tool/reasoning observation, and the plan must have purpose, goal and success criteria. This is a completion guard, not a second model judge or a separate goal database. The result remains ordinary plan state, observations and event history.

The pipeline is `strict parse -> safe normalization -> policy validation -> bounded fallback`. Normalization may only add deterministic structure; it never invents a tool, path, dependency, mutation payload, or ambiguous result selection. Invalid planner JSON still becomes a one-step, tool-free fallback plan. Output-format grammar begins at the first generated JSON token rather than consuming the chat template's assistant marker. A deterministic scheduler finds dependency- and evidence-ready pending steps in plan order and executes registered tools sequentially until its tool-batch budget is exhausted. Each result becomes a plan observation before the step is completed. Plan steps have an explicit mode: `tool`, `reasoning`, or `final_response`; a reasoning step produces one bounded JSON observation and only `final_response` produces the user-visible draft. `--reflection-mode always` may request a revision through compact fields such as `complete`, `activate`, `next_action` and `add_steps`; native code expands those into the small allowlisted lifecycle operations before policy evaluation. The model-facing repair schema keeps each compact list small, while the parser enforces the total normalized operation budget after expansion. Reflection remains sideband data and is never appended to normal chat history.

For a reasoning step, the model receives a bounded local context: its objective, direct dependencies, and observations produced by those dependencies. It does not receive unrelated plan history. The active tool profile is also a capability filter: the planner sees only registered tools from that profile, and registry validation checks required properties, types, enum values, string/item limits and numeric bounds before any adapter runs. Runtime defaults are intentionally narrow and read-only: for example `repository_search` can receive an empty relative path and `max_results=16`, `repository_read` gets line range `1..200`, and `web_search` gets `limit=5`. Prompt-derived query defaults are used only where bounded and deterministic; runtime never fabricates a write path or mutation data.

Tool arguments may use a structured binding from an earlier completed step's JSON observation. The runtime resolves it immediately before normal registered-tool schema validation; bindings are data-only and do not add a new executor or expression language.

```json
{
  "path": {
    "$from_step": "search",
    "$json_pointer": "/matches/0/path"
  }
}
```

The source step must be completed and have a JSON tool or reasoning observation. Binding nesting, IDs, JSON Pointer length, materialized argument size and source lookup are bounded; missing or malformed bindings become structured validation failures on the active step. This enables chains such as `repository_search` to `repository_read` without predeclaring the discovered path while still giving reflection/repair a stable failure observation.

```powershell
.\build-plan\bin\Debug\llama-agent.exe run `
  --backend cozo --memory-db .\work\agent.db `
  --model .\models\chat.gguf --embedding-model .\models\embedding.gguf `
  --prompt "What did we decide, and what is the next step?" `
  --tool-profile memory-read --planning-mode mini --reflection-mode always `
  --plan-backend cozo --plan-db .\work\agent-plan.cozo --plan-id feature-session `
  --plan-scope session --plan-show-summary --agent-trace --max-tool-rounds 1
```

The planner is deliberately fail-closed for actions but fail-soft for availability: invalid planner JSON becomes a one-step, tool-free fallback plan; an invalid reflection becomes `accept` of the generated draft. A planner-selected tool outside the registered profile is stripped before execution. `mini` rejects legacy tool flags because only registry-owned tools may be planned. `--plan-backend` is independently selectable from the memory backend. Both backends resolve to in-memory when no database path is supplied; `--memory-db PATH` and `--plan-db PATH` select Cozo automatically when the binary was compiled with `LLAMA_MEMORY_COZO` and `LLAMA_PLAN_COZO` respectively. Explicit backend flags win, and an in-memory backend combined with a DB path is rejected rather than silently ignoring persistence. With `--plan-db PATH --plan-id ID`, the runtime loads an existing compatible plan before planning; otherwise it creates that ID once. This makes a session, project, or global plan resumable across CLI processes. `plan_get` is bound when an explicit `--plan-id` is supplied.

`memory_remember` is available in the `memory` and `research` profiles as a policy-gated proposal, not as a generic write capability. Its native binding invokes the existing memory policy and audit path, returning the accept/reject/duplicate/conflict decision as a tool result. Generic plan runtime callers must explicitly set `allow_policy_gated_tool_proposals`; otherwise only read-only tools are eligible.

The bounded agent runtime supports the same pattern across plan steps: `max_tool_batches` limits tool executions per run, while reasoning steps do not consume that tool budget, and an active successful step is never re-executed after its observation is recorded. The scheduler distinguishes `runnable`, `blocked`, `complete` and `inactive`: a mandatory step waiting on missing evidence or a failed dependency is blocked, not terminal. Only completed, failed and cancelled plans are terminal. The scheduler, rather than reflection, normally progresses the DAG and therefore permits bounded chains such as `memory_search -> memory_get` without model-issued lifecycle operations. Reflection remains the path for repair, replanning and answer-quality review. This first scheduler slice is sequential and deterministic; it does not run ready steps in parallel or execute arbitrary shell commands.

An ordinary registered-tool handler failure records a capped JSON error observation and marks that active step `failed`; it does not abort the whole agent process. Registered-tool argument validation failures follow the same repairable lifecycle: the runtime records a `tool.invalid_arguments` observation, marks the step failed, emits a `tool_failure` learning signal, and then continues to drafting/reflection with the failure as evidence. The following draft and reflection can therefore report the limitation honestly or propose an allowed repair/retry. Unregistered tools and disallowed policy classes still stop execution fail-closed because they indicate a capability or authority boundary rather than a recoverable argument-shape problem. Runtime failure classification no longer depends on string-matching executor text: adapters return explicit native classes such as `validation`, `policy`, `not_found`, `timeout`, `network`, `execution`, or `limit`, and the runtime maps those to the agent-facing failure envelope.

Reflection is a sideband interface. Its JSON parser accepts only a short decision, readiness flag, confidence, revision guidance, optional learning hints and bounded compact repair fields. It neither requires nor stores chain-of-thought, and the agent runtime never puts reflection output into normal conversation history.

### Post-turn durable memory learning

`--memory-learn post-turn` adds one optional stage after a successful bounded agent turn. A separate model-backed candidate extractor returns either one constrained-JSON candidate or `null`; it is not part of the reflection parser and does not expose reflection text. The runtime then applies native shape, confidence (default `0.75`), expected-reuse (default `0.65`) and provenance gates before passing an accepted proposal through the existing `common_memory_evaluate_remember_request` policy and ordinary memory store. The policy remains the final authority for sensitive-content, scope, duplicate and conflict decisions.

```powershell
.\build-plan\bin\Release\llama-agent.exe run `
  --backend cozo --memory-db .\work\agent.db `
  --model .\models\chat.gguf --embedding-model .\models\embedding.gguf `
  --prompt "When testing persistent data, always close, reopen and read it back." `
  --planning-mode mini --plan-backend cozo --plan-db .\work\agent-plan.cozo `
  --memory-learn post-turn --memory-learn-show-candidate
```

Learning is off by default and currently requires `--planning-mode mini`. It chooses project scope only when the runtime has an explicit project id; otherwise it uses session scope. It never auto-selects global scope, and the model never receives scope or identity authority. `--memory-learn-show-candidate` prints the proposed content for local PoC inspection; the regular audit is structured and omits candidate content and reasoning. A procedure candidate needs at least one runtime-verified evidence item or completed plan step with an observation/result; model-provided provenance claims do not satisfy this gate. Ordinary one-off plans, transient next actions, malformed JSON and failed embedding/policy/persistence paths fail safely without writing memory.

The first feedback-aware slice also supplies native `tool_failure` signals to the post-turn candidate extractor when a registered plan-step tool fails and its capped error observation is recorded. If a plan later reaches `completed` after such a failure, the runtime adds a `successful_recovery` signal that points to the same evidence. A bounded reflection `learning_hint` is recorded the same way when reflection supplies one. Signals are evidence and metadata only: the model may cite their observation IDs for a reusable procedure candidate, but they never write memory directly. Accepted memories retain their signal types in `learning_signal_types` and their native tool names in `learning_tools` metadata. During a later reasoning step, the runtime reorders only the already retrieved procedure slice when its active/failing tool matches that metadata; semantic retrieval, scope filtering and the bounded three-procedure limit remain unchanged. Explicit user corrections remain a follow-up interface because they need an equally explicit turn/provenance boundary.

### Procedure memory, plan provenance and blueprints

Retrieved `procedure` memories may inform the model's initial plan or a bounded reflection `add_step` proposal. A proposed step can name `source_memory_ids`, but the runtime retains only IDs belonging to actually retrieved procedure memories, marks the surviving step `generated_from_memory=true`, and adds short `memory:<id>` evidence. The normal plan policy and store mutation path still decide whether the step is accepted; memory never mutates a plan directly.

Reflection uses a compact outer JSON schema to keep local grammar sampling bounded. Its model-facing repair fields (`complete`, `activate`, `next_action` and `add_steps`) are parsed and normalized natively into supported plan operations before they can mutate a plan; omitted repair step IDs are generated deterministically and omitted `after` fields chain added repair steps sequentially. The current model-facing schema allows up to two completed steps, two activated steps and two added repair steps, and the native parser rejects the result if the expanded operation list exceeds its total budget. The older explicit `operations` form remains parser-accepted locally, but the intended model-facing contract is the compact repair form so the model does not have to emit full internal operation payloads.

### Model-safe failure observations

Tool failures are rendered to the plan/reflection context as a bounded native envelope: stable `code`, `class`, `stage`, `tool`, `step_id`, `retryable`, `safe_summary` and evidence ID. The initial classifications are `validation`, `policy`, `not_found`, `timeout`, `network`, `execution`, and `limit`. Schema-contract diagnostics identify the rejected field when possible, for example `unexpected contract field: tool`, but arbitrary executor output is still kept out of the model-facing observation. This lets a later repair policy reason about a failure without treating raw tool output as instructions.

The same structured contract also reaches the chat-template tool bridge. A failed tool call becomes a bounded `role: tool` payload such as:

```json
{
  "ok": false,
  "error": {
    "code": "tool.web_fetch.request_failed",
    "message": "Web fetch request failed.",
    "retryable": true,
    "class": "network"
  }
}
```

Successful calls continue to return either parsed JSON under `result` or plain text under `result_text`.

### Contract and capability boundaries

Native tool invocation uses one bounded object-contract path: parse JSON, canonicalize its representation, validate the supported schema subset, apply runtime policy, then invoke the bound executor. The same parse/normalize/validate/policy pattern is the intended boundary for future reflection, import and procedure-patch contracts.

Tools retain three separate layers. The catalog declares versioned metadata and profile membership; the registry owns executable handlers; adapters bind a catalog definition to local runtime resources. A tool is model-visible only when the registered handler matches the catalog definition's name, version and `executor_id`. Catalog metadata alone never supplies executable code.

Reflection and memory-candidate outputs use the same bounded JSON readers for required strings, unit-range scores and short string arrays. Plan-specific shorthand, runtime tool defaults and observation-to-argument bindings remain separate normalizers because they perform distinct, explicitly allowed transformations.

### Explicit user corrections

The runtime does not infer corrections from arbitrary user text. An integration may instead supply an explicit `user_correction` with a bounded statement and mandatory `source_turn_id`. The runtime records it as an evidence-addressable `user_correction` observation and native learning signal. It may support a later procedure candidate only through the ordinary post-turn extractor and memory policy; it cannot directly alter a procedure, blueprint, plan scope or memory record.

Procedure metadata makes the current promotion lifecycle explicit: an accepted learned procedure starts as `candidate`, becomes `verified` after a completed distinct plan use, and becomes `promoted` when it reaches the configured verified-use threshold and a sanitized blueprint is persisted. A future `procedure_patch` or degradation path must remain an explicit, evidence-bound transition rather than an automatic rewrite.

`test-agent-lifecycle-scenarios` exercises the same lifecycle end to end in three bounded runs: a session-scoped plan is paused and resumed by `plan_id`; a transient tool failure is reflected into one policy-validated retry; and three distinct completed plans turn one post-turn procedure candidate into a promoted blueprint. `test-plan-cozo` separately verifies that the persisted plan state survives a store reopen.

A reusable blueprint is a normal persisted plan with `kind=blueprint`. A concrete task instance has `kind=task` plus `derived_from_plan_id`, and `common_plan_instantiate_blueprint()` copies the blueprint's structure into independently mutable IDs. It intentionally clears tool bindings so the current runtime/planner must bind only currently registered tools. Cozo persists these fields inside the existing plan state JSON, so there is no blueprint relation or separate database layer.

Blueprint selection can be explicit with `--agent-blueprint repository-change --plan-id change-42`, or use `--agent-blueprint auto --plan-id change-42`. Both use the same native selection-and-instantiation path: explicit mode is a fixed selector, while auto mode uses the llama-backed JSON adapter. When bootstrap is enabled for a mini-plan agent and no blueprint option is supplied, `auto` is now the default and the CLI derives a turn-local task-plan ID. Candidates are derived from installed package definitions and carry only logical ID, persisted ID and a short description. Selection is capped at 16 candidates, requires confidence of at least `0.75`, and otherwise falls back safely to compact free-plan creation. A supplied plan resumes only when it is a task plan with matching scope and session; the selector cannot choose scope, identity, task-plan ID, an arbitrary plan, or an uninstalled blueprint.

Enable the generic PoC:

```powershell
cmake -S . -B build-plan -DLLAMA_MEMORY=ON -DLLAMA_PLAN=ON -DLLAMA_AGENT_REFLECTION=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build-plan --config Release
ctest --test-dir build-plan -C Release --output-on-failure
```

The runtime is intentionally mockable: a planner creates a turn/session/project/global plan, an executor produces a draft, and a reflector may accept, request a single revision, or propose policy-validated updates. `global` is useful for a single local instance's reusable test or operational plan; multi-user deployments should apply an explicit namespace/tenant policy before enabling it. Configure `max_iterations` and `max_reflection_rounds` (defaults: 2 and 1) to keep the loop bounded.

Registered tools are explicit opt-in runtime dependencies. An active plan step can carry a structured tool name and object-shaped JSON arguments; the runtime executes one dependency-ready read-only tool step at a time, up to the configured batch limit, validates each through the registry, and records the capped result as a plan observation. It does not expose shell execution, file writes, CozoScript, or unrestricted native calls. The `research` profile now provides constrained `web_search` and `web_fetch` rather than a general-purpose `curl` escape hatch; their HTTPS, timeout, response-size and local-target rules live in the native registered handler.

Known limitations: the bounded chain is capped by `--max-tool-rounds` and `max_iterations`; and the current reflection operation set deliberately excludes arbitrary plan mutation. Cozo stores the full plan state and append-only event relation separately; a future migration can normalize individual steps, dependencies, observations, and assumptions into additional Cozo relations.
