#pragma once

#include "agent-runtime-session-host.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

struct common_agent_runtime_session_key {
    std::string namespace_id;
    std::string session_id;

    bool operator<(const common_agent_runtime_session_key & other) const {
        if (namespace_id != other.namespace_id) return namespace_id < other.namespace_id;
        return session_id < other.session_id;
    }
};

struct common_agent_runtime_session_descriptor {
    common_agent_runtime_session_key key;
    std::string project_id;
    common_memory_scope memory_scope = common_memory_scope::session;
    common_plan_scope plan_scope = common_plan_scope::turn;
    std::string policy_pack_id;
};

using common_agent_runtime_session_manager_turn_request = common_agent_runtime_session_host_turn_request;
using common_agent_runtime_session_manager_turn_result = common_agent_runtime_session_host_turn_result;
using common_agent_runtime_session_manager_config = common_agent_runtime_session_host_config;
using common_agent_runtime_session_manager_build_config = common_agent_runtime_session_host_build_config;

inline common_agent_runtime_session_manager_config make_agent_runtime_session_manager_config(
        common_agent_runtime_session_manager_build_config config) {
    return make_agent_runtime_session_host_config(std::move(config));
}

class common_agent_runtime_session_manager {
public:
    explicit common_agent_runtime_session_manager(common_agent_runtime_session_manager_config config);

    bool run_turn(
        const common_agent_runtime_session_manager_turn_request & request,
        common_agent_runtime_session_manager_turn_result & result,
        std::string & error);

    bool reset_session(
        const common_agent_runtime_session_key & key,
        std::string & error);

    bool close_session(
        const common_agent_runtime_session_key & key,
        std::string & error);

    std::vector<common_agent_runtime_session_descriptor> list_sessions() const;

    void reset_all();

private:
    common_agent_runtime_session_key make_session_key(
        const common_agent_runtime_session_manager_turn_request & request) const;

    common_agent_runtime_session_host & ensure_session_host(
        const common_agent_runtime_session_key & key);

    common_agent_runtime_session_manager_config config;
    std::map<common_agent_runtime_session_key, std::unique_ptr<common_agent_runtime_session_host>> hosts;
};
