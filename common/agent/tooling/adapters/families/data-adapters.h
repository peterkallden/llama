#pragma once

#include "agent/tooling/adapters/tool-adapters.h"

// Registers dataset, data-manipulation and statistics adapters. Returns true
// when the executor belongs to this family, including unavailable bindings.
bool common_try_register_data_tool_adapter(
        const common_tool_definition & definition,
        const common_native_tool_bindings & bindings,
        common_tool_registry & registry,
        bool & installed,
        std::string & error);
