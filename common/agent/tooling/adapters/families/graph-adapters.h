#pragma once

#include "agent/tooling/adapters/tool-adapters.h"

bool common_try_register_graph_tool_adapter(const common_tool_definition &,
    const common_native_tool_bindings &, common_tool_registry &, bool &, std::string &);

std::string common_context_graph_prompt_projection(common_agent_data_store &, const common_agent_scope &,
    const std::string & query, size_t budget, bool include_edges);

// Called by the resolved tool view, with its model-visible source read tools.
void common_context_graph_check_sources(nlohmann::ordered_json & result,
    const common_native_tool_bindings & bindings, const std::vector<std::string> & source_tools);

nlohmann::ordered_json common_source_revision_comparison(const common_source_reference & source,
    const std::string & current_revision, int64_t modified_at = 0);
