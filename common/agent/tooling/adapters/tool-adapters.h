#pragma once

#include "agent/tooling/adapters/adapter-bindings.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/registry/tool-registry.h"

#include <string>
#include <vector>

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
