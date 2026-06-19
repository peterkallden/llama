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

The first backend is `common_plan_in_memory_store`. Cozo plan persistence is intentionally not enabled in this vertical slice; it must use separate Cozo relations and APIs from memory storage.

Reflection is a sideband interface. Its JSON parser accepts only a short decision, readiness flag, confidence, and revision guidance. It neither requires nor stores chain-of-thought, and the agent runtime never puts reflection output into normal conversation history.

Enable the generic PoC:

```powershell
cmake -S . -B build-plan -DLLAMA_MEMORY=ON -DLLAMA_PLAN=ON -DLLAMA_AGENT_REFLECTION=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build-plan --config Release
ctest --test-dir build-plan -C Release --output-on-failure
```

The runtime is intentionally mockable: a planner creates a turn/session/project plan, an executor produces a draft, and a reflector may accept, request a single revision, or propose policy-validated updates. Configure `max_iterations` and `max_reflection_rounds` (defaults: 2 and 1) to keep the loop bounded.

Known limitations: this slice has no model-backed constrained planner yet, no CLI command, no registered-tool execution, and no Cozo plan backend. Those belong after the deterministic store and runtime tests are expanded and passing.
