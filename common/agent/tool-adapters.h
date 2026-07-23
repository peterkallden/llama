#pragma once

#include "agent/tool-catalog.h"
#include "agent/tool-registry.h"
#include "agent/sandbox-contract.h"
#include "memory/memory-store.h"
#include "plan/plan-store.h"
#include "resource/resource-contract.h"
#include "agent/data-store.h"

#include <functional>
#include <string>
#include <vector>

// Runtime-owned bindings. Neither the profile nor tool-call arguments choose
// a store, scope, plan id, or native implementation.
struct common_native_tool_bindings {
    common_memory_store * memory_store = nullptr;
    common_plan_store * plan_store = nullptr;
    common_memory_query memory_query;
    const std::string * plan_id = nullptr;
    // Canonical repository root supplied by the runtime, never by the model.
    // An empty value leaves repository tools unavailable.
    std::string repository_root;
    // Optional runtime-owned network tool overrides for deterministic tests or
    // alternative providers. Native defaults are used when absent.
    std::function<common_tool_execution_result(const std::string &)> web_search;
    std::function<common_tool_execution_result(const std::string &)> web_fetch;
    // Optional host-owned resource runtime for large tool payloads that should
    // be referenced without forcing the full content inline into model context.
    agent_resource_runtime resource_runtime;
    // Optional host-owned structured data backend. The backend is selected by
    // host configuration; tools only submit bounded semantic requests.
    common_agent_data_store * data_store = nullptr;
    // Optional host-owned semantic diagnostics provider. A runtime may back
    // this with clangd, an index, or another project-aware implementation.
    // The native adapters keep a bounded text fallback when it is absent.
    std::function<common_tool_execution_result(const std::string &)> diagnostics_symbol;
    std::function<common_tool_execution_result(const std::string &)> diagnostics_references;
    std::function<common_tool_execution_result(const std::string &)> diagnostics_call_hierarchy;
    // Optional runtime-owned semantic query embedding provider. Tool arguments
    // only supply text; model code never receives this callback or model path.
    std::function<bool(const std::string &, std::vector<float> &, std::string &)> embed_memory_query;
    // Executes a native memory proposal policy. It receives validated tool JSON
    // and returns a structured policy decision; it is never model-selected code.
    std::function<common_tool_execution_result(const std::string &)> memory_remember_proposal;
    // Host-owned semantic execution seam. The adapter creates a sandbox
    // request; the host helper owns workspace preparation and backend choice.
    std::function<common_tool_execution_result(common_agent_sandbox_request)> sandbox_execute;
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
