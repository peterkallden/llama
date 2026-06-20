#include "agent/tool-adapters.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"

#include <cassert>

int main() {
    std::string error, result;
    common_memory_in_memory_store memories;
    assert(memories.open("", error));
    common_memory_record memory;
    memory.id = "memory-1";
    memory.kind = common_memory_kind::fact;
    memory.content = "The plan store uses optimistic version checks.";
    memory.scope = common_memory_scope::session;
    memory.session_id = "session-1";
    assert(memories.put(memory, error));

    common_plan_in_memory_store plans;
    assert(plans.open("", error));
    common_plan_state plan;
    plan.id = "plan-1";
    plan.goal = "Verify native adapters";
    assert(plans.create(plan, error));

    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    assert(catalog.bootstrap("memory-read", bootstrap, error));
    common_tool_registry registry;
    common_native_tool_bindings bindings;
    bindings.memory_store = &memories;
    bindings.plan_store = &plans;
    bindings.memory_query.scope = common_memory_scope::session;
    bindings.memory_query.session_id = "session-1";
    bindings.plan_id = "plan-1";
    common_tool_adapter_result adapters;
    assert(common_register_native_tool_adapters(catalog, "memory-read", bindings, registry, adapters, error));
    assert(adapters.registered.size() == 5);
    assert(registry.execute({"calculator", R"({"expression":"(18 + 2) * 3"})"}, result, error));
    assert(result == R"({"value":60.0})" || result == R"({"value":60})");
    assert(registry.execute({"memory_search", R"({"query":"optimistic checks"})"}, result, error));
    assert(result.find("memory-1") != std::string::npos);
    assert(registry.execute({"memory_get", R"({"id":"memory-1"})"}, result, error));
    assert(result.find("optimistic version checks") != std::string::npos);
    assert(registry.execute({"plan_get", "{}"}, result, error));
    assert(result.find("plan-1") != std::string::npos);
    assert(!registry.execute({"memory_remember", "{}"}, result, error));
    return 0;
}
