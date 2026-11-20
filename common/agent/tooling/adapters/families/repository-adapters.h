#pragma once

#include "agent/tooling/adapters/adapter-bindings.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/registry/tool-registry.h"

bool common_try_register_repository_tool_adapter(
        const common_tool_definition & definition,
        const common_native_tool_bindings & bindings,
        common_tool_registry & registry,
        bool & installed,
        std::string & error);

// These reads are restricted to the host-bound repository and never fetch.
nlohmann::ordered_json common_repository_observation(const std::string & root, const std::string & path = {});
nlohmann::ordered_json common_repository_source_check(const std::string & root, const common_source_reference & source);
