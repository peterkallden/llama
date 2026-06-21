# Native dynamic plan and reflection PoC

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

`llama-memory` provides retrieved evidence only. A planner can propose typed plan operations, but `common_plan_policy` validates version checks, state transitions, dependencies, cycles, evidence requirements and limits before `common_plan_store` persists an update. Plan events are append-only short audit records. Plans are never memory records.

The first backend is `common_plan_in_memory_store`. When `LLAMA_PLAN_COZO=ON`, `common_plan_cozo_store` persists plan state in the separate Cozo relations `agent_plan` and `agent_plan_event`; it never uses the memory relations `memory` or `memory_edge`. The generic in-memory policy remains the mutation gate before an accepted state and its short audit event are persisted. If either persistence write fails, the cache is reloaded from Cozo; if the audit-event write fails after a state write, the prior state is restored before that reload.

## Tool catalog bootstrap (first slice)

`common_tool_catalog` is a declarative, versioned catalog of built-in capabilities. Its bootstrap is idempotent: it creates missing definitions and the standard profiles without rewriting existing entries. It deliberately contains metadata only (schemas, policy, limits, risk class and a native `executor_id`) and cannot load a DLL, script, command or other executable code. A future Cozo-backed catalog will persist these same records; the initial in-process catalog keeps the bootstrap semantics testable before introducing another database relation.

The separate agent bootstrap is also idempotent, but installs curated procedure memories and blueprint plans. `--agent-bootstrap default` deliberately uses a compiled stable package. `pocs/memory/bootstrap/default-v1.json` documents the equivalent external format and can be loaded explicitly with `--agent-bootstrap-file PATH`; keeping both provides a stable local default and a versioned file-based extension path.

The bootstrap profiles are `minimal`, `memory-read`, `memory` and `research`. The catalog includes read-only `calculator`, `time_now`, memory inspection/search, `plan_get`, proposal-only memory/plan mutation definitions, `mock_web_search`, and a declared-but-not-yet-wired `web_fetch` boundary. Proposal tools require confirmation and remain subject to native memory or plan policy; an enabled catalog entry never grants execution authority by itself.

`common_register_native_tool_adapters` is the first bridge from catalog to execution. It only registers implemented read-only executors and receives its memory store, memory scope and bound plan id from runtime-owned bindings. The current first set is `calculator`, UTC `time_now`, `memory_search`, scope-checked `memory_get`, and `plan_get`; proposal tools and web tools intentionally remain unavailable even if their profile declares them.

`common_tool_profile_to_chat_tools` and `common_tool_dispatch_chat_calls` connect that native registry to llama.cpp's existing chat-template and tool-call parser layer. The former exposes only definitions that are both profile-approved and actually registered; the latter turns a parsed assistant tool call into a bounded `role: tool` message for the next generation. It assigns a stable runtime call id when a template omitted one, caps results, and never executes proposal or unregistered tools. The first limit is one call per batch.

When built with `LLAMA_MEMORY=ON` and `LLAMA_AGENT_REFLECTION=ON`, the existing `llama-memory chat` PoC exposes this path with `--tool-profile minimal|memory-read|memory|research`. It bootstraps the selected profile in process, binds memory scope and the optional embedding provider from CLI-owned runtime state, performs one native read-only tool round, and completes with a tool-free final generation. The profile flag cannot be combined with the older `--memory-search-tool` or `--memory-remember-tool` flags.

## Model-backed chat loop (first vertical slice)

`llama-memory chat --planning-mode mini` now runs the existing bounded runtime around the already loaded chat model. The planner is constrained by the versioned plan JSON schema to return one to four ordered steps and then strictly policy-validated. Output-format grammar begins at the first generated JSON token rather than consuming the chat template's assistant marker. It lets only the active plan step invoke a registered tool, records the result as a plan observation, completes that step and activates the next dependency-ready step. The model-backed executor then drafts from the current plan, observations and retrieved memory. `--reflection-mode always` uses the same output-format boundary plus strict reflection parsing; it may request a revision or propose the small allowlisted lifecycle operations `complete_step`, `activate_step` and `set_next_action`. Reflection output remains sideband data and is never appended to normal chat history.

```powershell
.\build-plan\bin\Debug\llama-memory.exe chat `
  --backend cozo --memory-db .\work\agent.db `
  --model .\models\chat.gguf --embedding-model .\models\embedding.gguf `
  --prompt "What did we decide, and what is the next step?" `
  --tool-profile memory-read --planning-mode mini --reflection-mode always `
  --plan-backend cozo --plan-db .\work\agent-plan.cozo --plan-id feature-session `
  --plan-scope session --plan-show-summary --agent-trace --max-tool-rounds 1
```

The planner is deliberately fail-closed for actions but fail-soft for availability: invalid planner JSON becomes a one-step, tool-free fallback plan; an invalid reflection becomes `accept` of the generated draft. A planner-selected tool outside the registered profile is stripped before execution. `mini` rejects legacy tool flags because only registry-owned tools may be planned. `--plan-backend` is independently selectable from the memory backend. With `--plan-backend cozo --plan-db PATH --plan-id ID`, the runtime loads an existing compatible plan before planning; otherwise it creates that ID once. This makes a session, project, or global plan resumable across CLI processes. `plan_get` is bound when an explicit `--plan-id` is supplied.

`memory_remember` is available in the `memory` and `research` profiles as a policy-gated proposal, not as a generic write capability. Its native binding invokes the existing memory policy and audit path, returning the accept/reject/duplicate/conflict decision as a tool result. Generic plan runtime callers must explicitly set `allow_policy_gated_tool_proposals`; otherwise only read-only tools are eligible.

The bounded agent runtime supports the same pattern across plan steps: `max_tool_batches` limits executions per run, and an active step is never re-executed after its observation is recorded. A reflection may propose accepted plan operations such as `complete_step` followed by `activate_step`; if that newly active step has a read-only tool call and batch budget remains, the next iteration runs it. This permits bounded chains such as `memory_search → memory_get` without automatic execution of arbitrary pending steps.

Reflection is a sideband interface. Its JSON parser accepts only a short decision, readiness flag, confidence, and revision guidance. It neither requires nor stores chain-of-thought, and the agent runtime never puts reflection output into normal conversation history.

## Opt-in post-turn durable memory learning

`--memory-learn post-turn` adds one optional stage after a successful bounded agent turn. A separate model-backed candidate extractor returns either one constrained-JSON candidate or `null`; it is not part of the reflection parser and does not expose reflection text. The runtime then applies native shape, confidence (default `0.75`), expected-reuse (default `0.65`) and provenance gates before passing an accepted proposal through the existing `common_memory_evaluate_remember_request` policy and ordinary memory store. The policy remains the final authority for sensitive-content, scope, duplicate and conflict decisions.

```powershell
.\build-plan\bin\Release\llama-memory.exe chat `
  --backend cozo --memory-db .\work\agent.db `
  --model .\models\chat.gguf --embedding-model .\models\embedding.gguf `
  --prompt "When testing persistent data, always close, reopen and read it back." `
  --planning-mode mini --plan-backend cozo --plan-db .\work\agent-plan.cozo `
  --memory-learn post-turn --memory-learn-show-candidate
```

Learning is off by default and currently requires `--planning-mode mini`. It chooses project scope only when the runtime has an explicit project id; otherwise it uses session scope. It never auto-selects global scope, and the model never receives scope or identity authority. `--memory-learn-show-candidate` prints the proposed content for local PoC inspection; the regular audit is structured and omits candidate content and reasoning. A procedure candidate needs at least one runtime-verified evidence item or completed plan step with an observation/result; model-provided provenance claims do not satisfy this gate. Ordinary one-off plans, transient next actions, malformed JSON and failed embedding/policy/persistence paths fail safely without writing memory.

## Procedure memory, plan provenance and blueprints

Retrieved `procedure` memories may inform the model's initial plan or a bounded reflection `add_step` proposal. A proposed step can name `source_memory_ids`, but the runtime retains only IDs belonging to actually retrieved procedure memories, marks the surviving step `generated_from_memory=true`, and adds short `memory:<id>` evidence. The normal plan policy and store mutation path still decide whether the step is accepted; memory never mutates a plan directly.

A reusable blueprint is a normal persisted plan with `kind=blueprint`. A concrete task instance has `kind=task` plus `derived_from_plan_id`, and `common_plan_instantiate_blueprint()` copies the blueprint's structure into independently mutable IDs. It intentionally clears tool bindings so the current runtime/planner must bind only currently registered tools. Cozo persists these fields inside the existing plan state JSON, so there is no blueprint relation or separate database layer.

The first chat integration keeps selection explicit: `--agent-blueprint repository-change --plan-id change-42` instantiates the installed blueprint once, then the normal runtime resumes that task plan. Model-driven blueprint selection is intentionally deferred.

Enable the generic PoC:

```powershell
cmake -S . -B build-plan -DLLAMA_MEMORY=ON -DLLAMA_PLAN=ON -DLLAMA_AGENT_REFLECTION=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build-plan --config Release
ctest --test-dir build-plan -C Release --output-on-failure
```

The runtime is intentionally mockable: a planner creates a turn/session/project/global plan, an executor produces a draft, and a reflector may accept, request a single revision, or propose policy-validated updates. `global` is useful for a single local instance's reusable test or operational plan; multi-user deployments should apply an explicit namespace/tenant policy before enabling it. Configure `max_iterations` and `max_reflection_rounds` (defaults: 2 and 1) to keep the loop bounded.

Registered tools are explicit opt-in runtime dependencies. An active plan step can carry a structured tool name and object-shaped JSON arguments; the runtime executes at most one read-only registered tool batch, validates it through the registry, and records the capped result as a plan observation. It does not expose shell execution, file writes, CozoScript, or unrestricted native calls. A network tool should be a constrained capability such as `web_fetch` or `web_search`, with URL, timeout, response-size, and allowlist policy in its registered handler rather than a general-purpose `curl` escape hatch.

Known limitations: the bounded chain is capped by `--max-tool-rounds` and `max_iterations`; and the current reflection operation set deliberately excludes arbitrary plan mutation. Cozo stores the full plan state and append-only event relation separately; a future migration can normalize individual steps, dependencies, observations, and assumptions into additional Cozo relations.
