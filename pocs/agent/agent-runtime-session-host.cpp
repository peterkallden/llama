#include "agent-runtime-session-host.h"

#include "agent-runtime-resident.h"
#include "agent-tool-provider.h"

namespace {

bool validate_session_host_turn_request(
        const common_agent_runtime_session_host_turn_request & request,
        std::string & error) {
    if (request.prompt.empty()) {
        error = "session host turn request requires a prompt";
        return false;
    }
    if (request.session_id.empty()) {
        error = "session host turn request requires a session_id";
        return false;
    }
    if (request.namespace_id.empty()) {
        error = "session host turn request requires a namespace_id";
        return false;
    }

    const bool project_scope_requested =
        request.memory_scope == common_memory_scope::project ||
        request.plan_scope == common_plan_scope::project;
    if (project_scope_requested && request.project_id.empty()) {
        error = "project-scoped session host turn request requires a project_id";
        return false;
    }

    const bool turn_scope_requested =
        request.memory_scope == common_memory_scope::turn ||
        request.plan_scope == common_plan_scope::turn;
    if (turn_scope_requested && request.turn_id.empty()) {
        error = "turn-scoped session host turn request requires a turn_id";
        return false;
    }

    error.clear();
    return true;
}

} // namespace

common_agent_runtime_session_host_config make_agent_runtime_session_host_config(
        common_agent_runtime_session_host_build_config config) {
    return {
        config.memory_store,
        config.plan_store,
        std::move(config.resident_request),
        std::move(config.policy),
        std::move(config.runtime_config),
        std::move(config.orchestration_config),
        config.memory_scope,
        config.memory_enabled,
        std::move(config.installed_blueprint_candidates),
        std::move(config.tooling),
    };
}

common_agent_runtime_session_host::common_agent_runtime_session_host(
        common_agent_runtime_session_host_config config)
    : config(std::move(config)) {}

common_agent_runtime_session_host::~common_agent_runtime_session_host() = default;

common_agent_runtime_turn_request common_agent_runtime_session_host::make_base_turn_request(
        const common_agent_runtime_session_host_turn_request & request) const {
    auto resident_request = config.resident_request;
    resident_request.prompt = request.prompt;
    resident_request.session_id = request.session_id;
    resident_request.namespace_id = request.namespace_id;
    resident_request.project_id = request.project_id;
    resident_request.memory_scope = request.memory_scope;
    resident_request.plan_scope = request.plan_scope;
    if (request.n_predict > 0) {
        resident_request.n_predict = request.n_predict;
    }

    auto turn_request = make_agent_runtime_resident_base_turn_request(resident_request);
    turn_request.policy = config.policy;
    turn_request.policy.agent_inference_backend = resident_request.inference_backend;
    turn_request.runtime_config = config.runtime_config;
    turn_request.orchestration_config = config.orchestration_config;
    turn_request.orchestration_config.prompt = request.prompt;
    turn_request.memory_scope = request.memory_scope;
    turn_request.memory_enabled = config.memory_enabled;
    if (turn_request.generation_options.n_predict == 0) {
        turn_request.generation_options.n_predict = resident_request.n_predict;
    }
    return turn_request;
}

bool common_agent_runtime_session_host::ensure_runtime(
        const common_agent_runtime_session_host_turn_request & request,
        bool & reused,
        std::string & error) {
    const bool needs_new_runtime =
        !runtime ||
        active_session_id != request.session_id ||
        active_namespace_id != request.namespace_id ||
        active_project_id != request.project_id ||
        active_memory_scope != request.memory_scope ||
        active_plan_scope != request.plan_scope;

    if (!needs_new_runtime) {
        reused = true;
        error.clear();
        return true;
    }

    reused = false;
    runtime = std::make_unique<common_agent_runtime_resident_runtime>(
        make_agent_runtime_resident_runtime_config(
            config.memory_store,
            &config.plan_store,
            make_base_turn_request(request),
            {},
            config.installed_blueprint_candidates,
            config.tooling));
    active_session_id = request.session_id;
    active_namespace_id = request.namespace_id;
    active_project_id = request.project_id;
    active_memory_scope = request.memory_scope;
    active_plan_scope = request.plan_scope;
    error.clear();
    return true;
}

bool common_agent_runtime_session_host::run_turn(
        const common_agent_runtime_session_host_turn_request & request,
        common_agent_runtime_session_host_turn_result & result,
        std::string & error) {
    result = {};

    if (!validate_session_host_turn_request(request, error)) {
        result.error = error;
        return false;
    }

    bool runtime_reused = false;
    if (!ensure_runtime(request, runtime_reused, error)) {
        result.error = error;
        return false;
    }

    const std::string turn_id = request.turn_id.empty()
        ? "daemon-turn-" + std::to_string(++generated_turn_counter)
        : request.turn_id;

    common_agent_result agent_result;
    bool ok = false;
    switch (request.mode) {
        case common_agent_runtime_host_mode::chat:
            ok = runtime->run_chat_prompt(request.prompt, turn_id, request.n_predict, agent_result, error);
            break;
        case common_agent_runtime_host_mode::mini:
            ok = runtime->run_mini_prompt(request.prompt, turn_id, request.n_predict, agent_result, error);
            break;
    }

    result.ok = ok;
    result.runtime_reused = runtime_reused;
    result.limit_reached = agent_result.limit_reached;
    result.reflected = agent_result.reflected;
    result.revised = agent_result.revised;
    result.response = agent_result.response;
    result.plan_id = agent_result.plan_id ? *agent_result.plan_id : "";
    result.total_decoded_tokens = agent_result.total_decoded_tokens;
    result.event_count = agent_result.events.size();
    result.memory_learning_related_count = agent_result.memory_learning_related_count;
    result.memory_learning_summary = agent_result.memory_learning_summary;
    result.error = ok ? std::string() : error;
    return ok;
}

const common_agent_runtime_session * common_agent_runtime_session_host::session() const {
    return runtime ? &runtime->runtime_host().session() : nullptr;
}

common_agent_runtime_session * common_agent_runtime_session_host::session() {
    return runtime ? &runtime->runtime_host().session() : nullptr;
}

void common_agent_runtime_session_host::reset() {
    runtime.reset();
    active_session_id.clear();
    active_namespace_id.clear();
    active_project_id.clear();
    active_memory_scope = common_memory_scope::session;
    active_plan_scope = common_plan_scope::turn;
    generated_turn_counter = 0;
}
