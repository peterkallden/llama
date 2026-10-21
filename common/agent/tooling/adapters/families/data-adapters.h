#pragma once

#include "agent/tooling/adapters/adapter-bindings.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/registry/tool-registry.h"

// Resolve and authorize a host-registered dataset using the same scope rules
// as the native dataset tools. Other tool families should use this seam rather
// than reimplementing dataset reference checks.
bool common_authorize_dataset_reference(
        const common_native_tool_bindings & bindings,
        const std::string & dataset_uri,
        common_agent_dataset_descriptor & descriptor,
        common_tool_execution_result & failure);

// Registers dataset, data-manipulation and statistics adapters. Returns true
// when the executor belongs to this family, including unavailable bindings.
bool common_try_register_data_tool_adapter(
        const common_tool_definition & definition,
        const common_native_tool_bindings & bindings,
        common_tool_registry & registry,
        bool & installed,
        std::string & error);
