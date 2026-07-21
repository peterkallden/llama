#pragma once

#include "agent-daemon-adapter.h"
#include "../mcp/agent-mcp-auth.h"

#include <memory>
#include <string>

bool run_agent_daemon_unix_socket_adapter(
    const daemon_options & options,
    const std::shared_ptr<common_agent_daemon_config_store> & config_store,
    common_agent_daemon_dispatcher & dispatcher,
    const std::shared_ptr<const agent_mcp_authenticator> & authenticator,
    std::string & error);
