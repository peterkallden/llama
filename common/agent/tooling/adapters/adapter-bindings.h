#pragma once

#include "agent/data-store.h"
#include "agent/sandbox/sandbox-contract.h"
#include "agent/tooling/contracts/tool-runtime-contract.h"
#include "memory/memory-store.h"
#include "plan/plan-store.h"
#include "resource/resource-contract.h"

#include <functional>
#include <string>
#include <vector>

class agent_resource_processing_provider;

// Runtime-owned bindings shared by adapter families. This is deliberately
// separate from the registration entry point so a family header does not
// pull in the complete adapter router.
struct common_native_tool_bindings {
    common_memory_store * memory_store = nullptr;
    common_plan_store * plan_store = nullptr;
    common_memory_query memory_query;
    const std::string * plan_id = nullptr;
    std::string repository_root;
    std::function<common_tool_execution_result(const std::string &)> web_search;
    std::function<common_tool_execution_result(const std::string &)> web_fetch;
    agent_resource_runtime resource_runtime;
    agent_resource_processing_provider * resource_processing_service = nullptr;
    agent_resource_processing_provider_factory resource_processing_provider_factory;
    common_agent_data_store * data_store = nullptr;
    std::function<common_tool_execution_result(const std::string &)> document_tables;
    std::function<common_tool_execution_result(const std::string &)> document_table;
    std::function<common_tool_execution_result(const std::string &, const std::string &)> dataset_from_resource;
    std::function<common_tool_execution_result(const std::string &)> diagnostics_symbol;
    std::function<common_tool_execution_result(const std::string &)> diagnostics_references;
    std::function<common_tool_execution_result(const std::string &)> diagnostics_call_hierarchy;
    std::function<common_tool_execution_result(const std::string &)> diagnostics_native_crash;
    std::function<bool(const std::string &, std::vector<float> &, std::string &)> embed_memory_query;
    std::function<common_tool_execution_result(const std::string &)> memory_remember_proposal;
    std::function<common_tool_execution_result(common_agent_sandbox_request)> sandbox_execute;
};
