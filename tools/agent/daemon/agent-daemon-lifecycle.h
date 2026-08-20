#pragma once

enum class common_agent_daemon_state {
    starting,
    ready,
    draining,
    stopping,
    stopped,
    failed,
};

inline const char * common_agent_daemon_state_name(common_agent_daemon_state state) {
    switch (state) {
        case common_agent_daemon_state::starting: return "starting";
        case common_agent_daemon_state::ready:    return "ready";
        case common_agent_daemon_state::draining: return "draining";
        case common_agent_daemon_state::stopping: return "stopping";
        case common_agent_daemon_state::stopped:  return "stopped";
        case common_agent_daemon_state::failed:   return "failed";
    }
    return "failed";
}

enum class common_agent_daemon_shutdown_mode {
    drain,
    cancel,
};
