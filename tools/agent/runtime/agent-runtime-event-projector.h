#pragma once

#include "../daemon/agent-daemon-events.h"
#include "agent/contracts/agent-events.h"

common_agent_event_sink make_agent_runtime_event_projector(
        common_agent_daemon_event_sink daemon_sink,
        common_agent_event_context context);
