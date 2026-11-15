#pragma once
#include "tools/agent/runtime/agent-runtime-session-host.h"
#include "agent/data-store.h"
#include <map>

struct agent_research_host_resources {
    common_agent_data_store * data_store = nullptr;
    agent_resource_store * resource_store = nullptr;
};
bool agent_prepare_research_workspace(agent_research_host_resources &, common_agent_runtime_session_host_turn_request &, std::string &);
bool agent_persist_research_workspace(agent_research_host_resources &, const common_agent_runtime_session_host_turn_request &, common_agent_runtime_session_host_turn_result &, std::string &, common_agent_research_draft_ref * = nullptr, std::map<std::string, agent_resource_descriptor> * = nullptr);
