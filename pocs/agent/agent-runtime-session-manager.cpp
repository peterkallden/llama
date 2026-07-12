#include "agent-runtime-session-manager.h"

common_agent_runtime_session_manager::common_agent_runtime_session_manager(
        common_agent_runtime_session_manager_config config)
    : config(std::move(config)) {}

common_agent_runtime_session_key common_agent_runtime_session_manager::make_session_key(
        const common_agent_runtime_session_manager_turn_request & request) const {
    return {
        request.namespace_id,
        request.session_id,
    };
}

common_agent_runtime_session_host & common_agent_runtime_session_manager::ensure_session_host(
        const common_agent_runtime_session_key & key) {
    auto it = hosts.find(key);
    if (it == hosts.end()) {
        it = hosts.emplace(
            key,
            std::make_unique<common_agent_runtime_session_host>(config)).first;
    }
    return *it->second;
}

bool common_agent_runtime_session_manager::run_turn(
        const common_agent_runtime_session_manager_turn_request & request,
        common_agent_runtime_session_manager_turn_result & result,
        std::string & error) {
    auto & host = ensure_session_host(make_session_key(request));
    return host.run_turn(request, result, error);
}

bool common_agent_runtime_session_manager::reset_session(
        const common_agent_runtime_session_key & key,
        std::string & error) {
    auto it = hosts.find(key);
    if (it == hosts.end()) {
        error = "session is not active";
        return false;
    }

    it->second->reset();
    error.clear();
    return true;
}

bool common_agent_runtime_session_manager::close_session(
        const common_agent_runtime_session_key & key,
        std::string & error) {
    auto it = hosts.find(key);
    if (it == hosts.end()) {
        error = "session is not active";
        return false;
    }

    hosts.erase(it);
    error.clear();
    return true;
}

std::vector<common_agent_runtime_session_descriptor> common_agent_runtime_session_manager::list_sessions() const {
    std::vector<common_agent_runtime_session_descriptor> sessions;
    sessions.reserve(hosts.size());
    for (const auto & entry : hosts) {
        const auto descriptor = entry.second->describe_session();
        sessions.push_back({
            entry.first,
            descriptor.project_id,
            descriptor.memory_scope,
            descriptor.plan_scope,
            descriptor.policy_pack_id,
        });
    }
    return sessions;
}

void common_agent_runtime_session_manager::reset_all() {
    hosts.clear();
}
