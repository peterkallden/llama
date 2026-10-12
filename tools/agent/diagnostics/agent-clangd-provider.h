#pragma once

#include "agent-clangd-session.h"
#include "agent/tooling/contracts/tool-runtime-contract.h"

#include <memory>

class agent_clangd_diagnostics_provider {
public:
    explicit agent_clangd_diagnostics_provider(agent_clangd_session_config config);

    common_tool_execution_result symbol(const std::string & arguments_json);
    common_tool_execution_result references(const std::string & arguments_json);
    common_tool_execution_result call_hierarchy(const std::string & arguments_json);

private:
    common_tool_execution_result workspace_symbol(const agent_clangd_json & arguments);
    bool resolve_location(const agent_clangd_json & arguments, agent_clangd_json & location, std::string & error);
    common_tool_execution_result location_query(const agent_clangd_json & arguments, const std::string & method);

    std::unique_ptr<agent_clangd_session> session_;
    std::string repository_root_;
};
