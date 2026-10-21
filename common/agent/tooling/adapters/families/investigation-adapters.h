#pragma once

#include "agent/tooling/adapters/adapter-bindings.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/registry/tool-registry.h"

bool common_try_register_investigation_tool_adapter(
        const common_tool_definition &, const common_native_tool_bindings &,
        common_tool_registry &, bool & installed, std::string & error);
