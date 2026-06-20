#pragma once

#include "agent/tool-catalog.h"
#include "agent/tool-registry.h"
#include "memory/memory-store.h"
#include "plan/plan-store.h"

#include <functional>
#include <string>
#include <vector>

// Runtime-owned bindings. Neither the profile nor tool-call arguments choose
// a store, scope, plan id, or native implementation.
struct common_native_tool_bindings {
    common_memory_store * memory_store = nullptr;
    common_plan_store * plan_store = nullptr;
    common_memory_query memory_query;
    std::string plan_id;
    // Optional runtime-owned semantic query embedding provider. Tool arguments
    // only supply text; model code never receives this callback or model path.
    std::function<bool(const std::string &, std::vector<float> &, std::string &)> embed_memory_query;
    // Executes a native memory proposal policy. It receives validated tool JSON
    // and returns a structured policy decision; it is never model-selected code.
    std::function<bool(const std::string &, std::string &, std::string &)> memory_remember_proposal;
};

struct common_tool_adapter_result {
    std::vector<std::string> registered;
    std::vector<std::string> unavailable;
};

// Registers only implemented, read-only native executors from a catalog
// profile. Proposal definitions intentionally remain unavailable until their
// policy/audit write paths are installed.
bool common_register_native_tool_adapters(
    const common_tool_catalog & catalog,
    const std::string & profile_id,
    const common_native_tool_bindings & bindings,
    common_tool_registry & registry,
    common_tool_adapter_result & result,
    std::string & error);
