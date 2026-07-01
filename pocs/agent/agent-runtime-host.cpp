#include "agent-runtime-host.h"

namespace {

const std::vector<common_blueprint_candidate> k_empty_blueprints;
const std::vector<common_memory_hit> k_empty_memories;
const std::string k_empty_fallback_reason;

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
