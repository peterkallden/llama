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

The first backend is `common_plan_in_memory_store`. When `LLAMA_PLAN_COZO=ON`, `common_plan_cozo_store` persists plan state in the separate Cozo relations `agent_plan` and `agent_plan_event`; it never uses the memory relations `memory` or `memory_edge`. The generic in-memory policy remains the mutation gate before an accepted state and its short audit event are persisted.

## Tool catalog bootstrap (first slice)

`common_tool_catalog` is a declarative, versioned catalog of built-in capabilities. Its bootstrap is idempotent: it creates missing definitions and the standard profiles without rewriting existing entries. It deliberately contains metadata only (schemas, policy, limits, risk class and a native `executor_id`) and cannot load a DLL, script, command or other executable code. A future Cozo-backed catalog will persist these same records; the initial in-process catalog keeps the bootstrap semantics testable before introducing another database relation.

The bootstrap profiles are `minimal`, `memory-read`, `memory` and `research`. The catalog includes read-only `calculator`, `time_now`, memory inspection/search, `plan_get`, proposal-only memory/plan mutation definitions, `mock_web_search`, and a declared-but-not-yet-wired `web_fetch` boundary. Proposal tools require confirmation and remain subject to native memory or plan policy; an enabled catalog entry never grants execution authority by itself.

`common_register_native_tool_adapters` is the first bridge from catalog to execution. It only registers implemented read-only executors and receives its memory store, memory scope and bound plan id from runtime-owned bindings. The current first set is `calculator`, UTC `time_now`, `memory_search`, scope-checked `memory_get`, and `plan_get`; proposal tools and web tools intentionally remain unavailable even if their profile declares them.

Reflection is a sideband interface. Its JSON parser accepts only a short decision, readiness flag, confidence, and revision guidance. It neither requires nor stores chain-of-thought, and the agent runtime never puts reflection output into normal conversation history.

Enable the generic PoC:

```powershell
cmake -S . -B build-plan -DLLAMA_MEMORY=ON -DLLAMA_PLAN=ON -DLLAMA_AGENT_REFLECTION=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build-plan --config Release
ctest --test-dir build-plan -C Release --output-on-failure
```

The runtime is intentionally mockable: a planner creates a turn/session/project/global plan, an executor produces a draft, and a reflector may accept, request a single revision, or propose policy-validated updates. `global` is useful for a single local instance's reusable test or operational plan; multi-user deployments should apply an explicit namespace/tenant policy before enabling it. Configure `max_iterations` and `max_reflection_rounds` (defaults: 2 and 1) to keep the loop bounded.

Registered tools are explicit opt-in runtime dependencies. An active plan step can carry a structured tool name and object-shaped JSON arguments; the runtime executes at most one read-only registered tool batch, validates it through the registry, and records the capped result as a plan observation. It does not expose shell execution, file writes, CozoScript, or unrestricted native calls. A network tool should be a constrained capability such as `web_fetch` or `web_search`, with URL, timeout, response-size, and allowlist policy in its registered handler rather than a general-purpose `curl` escape hatch.

Known limitations: this slice has no model-backed constrained planner wired into an inference CLI, no plan CLI command, and only the first one-tool bounded action shape. Cozo stores the full plan state and append-only event relation separately; a future migration can normalize individual steps, dependencies, observations, and assumptions into additional Cozo relations.
