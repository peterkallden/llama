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
        nullptr,
        context.installed_blueprint_candidates,
        &context.memories,
        context.tools,
        context.profile_tools_active,
        context.tool_registry,
        context.tool_handler,
        false,
        {},
    };
    inputs.scope = &inputs.turn_request.scope;
    return inputs;
}

common_agent_runtime_host_execution make_agent_runtime_host_execution(
        common_agent_runtime_host_inputs & inputs,
        common_agent_inference & inference) {
    return {
        inputs.mode,
        inputs.memory_store,
        inputs.plan_store,
        inference,
        inputs.turn_request,
        inputs.current_plan_id,
        inputs.scope,
        inputs.installed_blueprint_candidates,
        inputs.memories,
        inputs.tools,
        inputs.profile_tools_active,
        inputs.tool_registry,
        inputs.tool_handler,
    };
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
            if (execution.scope == nullptr) {
                error = "mini runtime host requires an agent scope";
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
                *execution.scope,
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

common_agent_runtime_turn_request make_agent_runtime_resident_chat_turn_request(
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

common_agent_runtime_resident_chat_host::common_agent_runtime_resident_chat_host(
        common_agent_runtime_resident_chat_host_config config)
    : memory_store(config.memory_store), base_turn_request(std::move(config.base_turn_request)) {}

bool common_agent_runtime_resident_chat_host::run_prompt(
        const std::string & prompt,
        const std::string & turn_id,
        common_agent_result & result,
        std::string & error) {
    const std::vector<common_memory_hit> memories;
    const std::vector<common_chat_tool> tools;
    common_agent_runtime_host_build_context build_context{
        memory_store,
        nullptr,
        make_agent_runtime_resident_chat_turn_request(base_turn_request, prompt, turn_id),
        nullptr,
        nullptr,
        memories,
        tools,
        false,
        nullptr,
        {},
    };
    auto inputs = make_agent_runtime_host_chat_inputs(build_context);
    return host.run_turn(inputs, result, error);
}

void common_agent_runtime_resident_chat_host::reset() {
    host.reset();
}
