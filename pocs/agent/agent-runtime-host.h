#pragma once

#include "agent-runtime-host-contracts.h"

bool run_agent_runtime_host_session(
    common_agent_runtime_host_inputs & inputs,
    common_agent_runtime_session & session,
    common_agent_result & result,
    std::string & error);

bool complete_agent_runtime_host_turn(
    common_agent_runtime_host_inputs & inputs,
    common_agent_runtime_session & session,
    const common_agent_result & result,
    std::string & error);

bool run_agent_runtime_host_turn(
    common_agent_runtime_host_inputs & inputs,
    common_agent_runtime_session & session,
    common_agent_result & result,
    std::string & error);

bool run_agent_runtime_host(
    common_agent_runtime_host_execution & execution,
    common_agent_result & result,
    std::string & error);
