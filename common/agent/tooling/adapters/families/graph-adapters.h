#pragma once

#include "agent/tooling/adapters/tool-adapters.h"

bool common_try_register_graph_tool_adapter(const common_tool_definition &,
    const common_native_tool_bindings &, common_tool_registry &, bool &, std::string &);

std::string common_context_graph_prompt_projection(common_agent_data_store &, const common_agent_scope &,
    const std::string & query, size_t budget, bool include_edges);
