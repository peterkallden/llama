#include "agent-runtime-host.h"
#include "agent-runtime-session-host.h"

namespace {

const std::vector<common_blueprint_candidate> k_empty_blueprints;
const std::vector<common_memory_hit> k_empty_memories;
const std::string k_empty_fallback_reason;

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

common_agent_runtime_host_inputs make_agent_runtime_host_chat_inputs(
        common_agent_runtime_host_build_context & context) {
    auto turn_request = context.turn_request;
    turn_request.request.enable_memory = turn_request.memory_enabled;
    turn_request.request.memories = context.memories;
    if (turn_request.request.prompt.empty()) {
        turn_request.request.prompt = turn_request.orchestration_config.prompt;
    }
    common_agent_scope_apply(turn_request.scope, turn_request.request);
    turn_request.generation_options.n_predict = turn_request.inference_options.n_predict;

    common_agent_runtime_host_inputs inputs{
        common_agent_runtime_host_mode::chat,
        context.memory_store,
        nullptr,
        std::move(turn_request),
        nullptr,
        nullptr,
        &context.memories,
        context.tools,
        context.profile_tools_active,
        context.tool_registry,
        context.tool_handler,
        false,
        {},
    };
    return inputs;
}

common_agent_runtime_host_inputs make_agent_runtime_host_mini_inputs(
        common_agent_runtime_host_build_context & context,
        const common_agent_orchestration_config & orchestration_config) {
    auto turn_request = context.turn_request;
    turn_request.orchestration_config = orchestration_config;
    turn_request.request.enable_memory = turn_request.memory_enabled;
    turn_request.request.memories = context.memories;
    if (turn_request.request.prompt.empty()) {
        turn_request.request.prompt = turn_request.orchestration_config.prompt;
    }
    common_agent_scope_apply(turn_request.scope, turn_request.request);

    common_agent_runtime_host_inputs inputs{
        common_agent_runtime_host_mode::mini,
        context.memory_store,
        context.plan_store,
        std::move(turn_request),
        context.current_plan_id,
        context.installed_blueprint_candidates,
        &context.memories,
        context.tools,
        context.profile_tools_active,
        context.tool_registry,
        context.tool_handler,
        false,
        {},
    };
    return inputs;
}

common_agent_runtime_host_execution make_agent_runtime_host_execution(
        common_agent_runtime_host_inputs & inputs,
        common_agent_inference & inference) {
    common_agent_runtime_host_execution execution{
        inputs.mode,
        inputs.memory_store,
        inputs.plan_store,
        inference,
        {},
        inputs.current_plan_id,
        inputs.installed_blueprint_candidates,
        inputs.memories,
        inputs.tools,
        inputs.profile_tools_active,
        inputs.tool_registry,
        inputs.tool_handler,
    };
    execution.turn_request = inputs.turn_request;
    return execution;
}

bool run_agent_runtime_host_session(
        common_agent_runtime_host_inputs & inputs,
        common_agent_runtime_session & session,
        common_agent_result & result,
        std::string & error) {
    agent_inference_backend inference_backend_kind = agent_inference_backend::cli;
    if (!parse_agent_inference_backend(inputs.turn_request.policy.agent_inference_backend, inference_backend_kind)) {
        error = "unsupported --agent-inference-backend: " + inputs.turn_request.policy.agent_inference_backend;
        return false;
    }

    const std::string & fallback_reason = inputs.turn_request.fallback_reason.empty() ? k_empty_fallback_reason : inputs.turn_request.fallback_reason;
    if (!initialize_agent_runtime_session(
            inputs.turn_request.inference_options,
            inference_backend_kind,
            inputs.turn_request.memory_enabled,
            fallback_reason,
            session,
            error)) {
        return false;
    }

    auto execution = make_agent_runtime_host_execution(inputs, *session.inference_session.inference);
    return run_agent_runtime_host(execution, result, error);
}

bool complete_agent_runtime_host_turn(
        common_agent_runtime_host_inputs & inputs,
        common_agent_runtime_session & session,
        const common_agent_result & result,
        std::string & error) {
    if (inputs.post_run && !inputs.post_run(result, error)) {
        if (inputs.reset_session_on_completion) {
            session.reset();
        }
        return false;
    }

    if (inputs.reset_session_on_completion) {
        session.reset();
    }

    error.clear();
    return true;
}

bool run_agent_runtime_host_turn(
        common_agent_runtime_host_inputs & inputs,
        common_agent_runtime_session & session,
        common_agent_result & result,
        std::string & error) {
    if (!run_agent_runtime_host_session(inputs, session, result, error)) {
        if (inputs.reset_session_on_completion) {
            session.reset();
        }
        return false;
    }

    return complete_agent_runtime_host_turn(inputs, session, result, error);
}

bool run_agent_runtime_host(
        common_agent_runtime_host_execution & execution,
        common_agent_result & result,
        std::string & error) {
    switch (execution.mode) {
        case common_agent_runtime_host_mode::mini: {
            if (execution.plan_store == nullptr) {
                error = "mini runtime host requires a plan store";
                return false;
            }
            if (execution.current_plan_id == nullptr) {
                error = "mini runtime host requires a current plan id";
                return false;
            }
            const auto & blueprints = execution.installed_blueprint_candidates ? *execution.installed_blueprint_candidates : k_empty_blueprints;
            const auto & memories = execution.memories ? *execution.memories : k_empty_memories;
            common_agent_runtime_driver_execution driver_execution{
                execution.memory_store,
                *execution.plan_store,
                execution.inference,
                execution.turn_request.policy,
                execution.turn_request.runtime_config,
                execution.turn_request.orchestration_config,
                *execution.current_plan_id,
                execution.turn_request.scope,
                blueprints,
                memories,
                execution.turn_request.memory_scope,
                execution.turn_request.memory_enabled,
                execution.tools,
                execution.profile_tools_active,
                execution.tool_registry,
            };
            return run_agent_runtime_driver(driver_execution, result, error);
        }

        case common_agent_runtime_host_mode::chat: {
            common_agent_chat_runtime_execution chat_execution{
                execution.inference,
                execution.turn_request.request,
                execution.turn_request.generation_options,
                {execution.turn_request.policy.max_tool_rounds},
                execution.tools,
                execution.profile_tools_active,
                execution.tool_registry,
                execution.tool_handler,
            };
            return run_agent_chat_runtime(chat_execution, result, error);
        }
    }

    error = "unsupported runtime host mode";
    return false;
}

bool common_agent_runtime_resident_host::run_turn(
        common_agent_runtime_host_inputs & inputs,
        common_agent_result & result,
        std::string & error) {
    const bool original_reset = inputs.reset_session_on_completion;
    inputs.reset_session_on_completion = false;
    const bool ok = run_agent_runtime_host_turn(inputs, runtime_session, result, error);
    inputs.reset_session_on_completion = original_reset;
    return ok;
}

void common_agent_runtime_resident_host::reset() {
    runtime_session.reset();
}

common_agent_runtime_turn_request make_agent_runtime_resident_base_turn_request(
        const common_agent_runtime_resident_request_config & config) {
    common_agent_runtime_turn_request turn_request;
    turn_request.request.prompt = config.prompt;
    turn_request.request.messages = {{"user", config.prompt}};
    turn_request.request.session_id = config.session_id;
    turn_request.request.namespace_id = config.namespace_id;
    turn_request.request.project_id = config.project_id;
    turn_request.scope.namespace_id = config.namespace_id;
    turn_request.scope.session_id = config.session_id;
    turn_request.scope.project_id = config.project_id;
    turn_request.scope.memory_scope = config.memory_scope;
    turn_request.scope.plan_scope = config.plan_scope;
    turn_request.inference_options.model = config.model;
    turn_request.inference_options.n_predict = config.n_predict;
    turn_request.inference_options.n_gpu_layers = config.n_gpu_layers;
    turn_request.inference_options.fit_params = config.fit_params;
    turn_request.policy.agent_inference_backend = config.inference_backend;
    turn_request.orchestration_config.prompt = config.prompt;
    turn_request.generation_options.n_predict = config.n_predict;
    return turn_request;
}

common_agent_runtime_turn_request make_agent_runtime_resident_turn_request(
        const common_agent_runtime_turn_request & base_turn_request,
        const std::string & prompt,
        const std::string & turn_id) {
    auto turn_request = base_turn_request;
    turn_request.request.prompt = prompt;
    turn_request.request.messages = {{"user", prompt}};
    turn_request.request.turn_id = turn_id;
    turn_request.scope.turn_id = turn_id;
    turn_request.orchestration_config.prompt = prompt;
    return turn_request;
}

common_agent_runtime_resident_runtime_config make_agent_runtime_resident_runtime_config(
        common_memory_store & memory_store,
        common_plan_store * plan_store,
        common_agent_runtime_turn_request base_turn_request,
        std::string current_plan_id,
        std::vector<common_blueprint_candidate> installed_blueprint_candidates,
        std::vector<common_chat_tool> tools,
        bool profile_tools_active,
        const common_tool_registry * tool_registry) {
    return {
        memory_store,
        plan_store,
        std::move(base_turn_request),
        std::move(current_plan_id),
        std::move(installed_blueprint_candidates),
        std::move(tools),
        profile_tools_active,
        tool_registry,
    };
}

common_agent_runtime_resident_runtime::common_agent_runtime_resident_runtime(
        common_agent_runtime_resident_runtime_config config)
    : memory_store(config.memory_store),
      plan_store(config.plan_store),
      base_turn_request(std::move(config.base_turn_request)),
      resident_current_plan_id(std::move(config.current_plan_id)),
      installed_blueprint_candidates(std::move(config.installed_blueprint_candidates)),
      tools(std::move(config.tools)),
      profile_tools_active(config.profile_tools_active),
      tool_registry(config.tool_registry) {}

bool common_agent_runtime_resident_runtime::run_chat_prompt(
        const std::string & prompt,
        const std::string & turn_id,
        common_agent_result & result,
        std::string & error) {
    const std::vector<common_memory_hit> memories;
    common_agent_runtime_host_build_context build_context{
        memory_store,
        nullptr,
        make_agent_runtime_resident_turn_request(base_turn_request, prompt, turn_id),
        nullptr,
        nullptr,
        memories,
        tools,
        profile_tools_active,
        tool_registry,
        {},
    };
    auto inputs = make_agent_runtime_host_chat_inputs(build_context);
    return host.run_turn(inputs, result, error);
}

bool common_agent_runtime_resident_runtime::run_mini_prompt(
        const std::string & prompt,
        const std::string & turn_id,
        common_agent_result & result,
        std::string & error) {
    if (plan_store == nullptr) {
        error = "resident mini runtime requires a plan store";
        return false;
    }

    const std::vector<common_memory_hit> memories;
    common_agent_runtime_host_build_context build_context{
        memory_store,
        plan_store,
        make_agent_runtime_resident_turn_request(base_turn_request, prompt, turn_id),
        &resident_current_plan_id,
        &installed_blueprint_candidates,
        memories,
        tools,
        profile_tools_active,
        tool_registry,
        {},
    };
    auto inputs = make_agent_runtime_host_mini_inputs(build_context, build_context.turn_request.orchestration_config);
    const bool ok = host.run_turn(inputs, result, error);
    if (ok && result.plan_id) {
        resident_current_plan_id = *result.plan_id;
    }
    return ok;
}

void common_agent_runtime_resident_runtime::reset() {
    host.reset();
    resident_current_plan_id.clear();
}

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
        std::move(config.tools),
        config.profile_tools_active,
        config.tool_registry,
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
    const int requested_n_predict = request.n_predict > 0 ? request.n_predict : config.resident_request.n_predict;
    const bool needs_new_runtime =
        !runtime ||
        active_session_id != request.session_id ||
        active_namespace_id != request.namespace_id ||
        active_project_id != request.project_id ||
        active_memory_scope != request.memory_scope ||
        active_plan_scope != request.plan_scope ||
        active_n_predict != requested_n_predict;

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
            config.tools,
            config.profile_tools_active,
            config.tool_registry));
    active_session_id = request.session_id;
    active_namespace_id = request.namespace_id;
    active_project_id = request.project_id;
    active_memory_scope = request.memory_scope;
    active_plan_scope = request.plan_scope;
    active_n_predict = requested_n_predict;
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
            ok = runtime->run_chat_prompt(request.prompt, turn_id, agent_result, error);
            break;
        case common_agent_runtime_host_mode::mini:
            ok = runtime->run_mini_prompt(request.prompt, turn_id, agent_result, error);
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
    active_n_predict = 0;
    generated_turn_counter = 0;
}
