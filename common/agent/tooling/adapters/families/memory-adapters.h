#pragma once

#include "agent/tooling/adapters/adapter-bindings.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/registry/tool-registry.h"

// Registers one memory-family adapter. Returns true when the definition is a
// memory executor, including when it is unavailable because its host binding
// is absent. The caller remains responsible for recording the result.
bool common_try_register_memory_tool_adapter(
        const common_tool_definition & definition,
        const common_native_tool_bindings & bindings,
        common_tool_registry & registry,
        bool & installed,
        std::string & error);
