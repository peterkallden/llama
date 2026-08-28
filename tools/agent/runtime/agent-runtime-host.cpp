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
    turn_request.request.available_datasets = context.tooling.available_datasets;
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
        context.tooling,
        false,
        {},
    };
    return inputs;
}

common_agent_runtime_host_inputs make_agent_runtime_host_agent_inputs(
        common_agent_runtime_host_build_context & context,
        const common_agent_orchestration_config & orchestration_config) {
    auto turn_request = context.turn_request;
    turn_request.orchestration_config = orchestration_config;
    turn_request.request.enable_memory = turn_request.memory_enabled;
    turn_request.request.memories = context.memories;
    turn_request.request.available_datasets = context.tooling.available_datasets;
    if (turn_request.request.prompt.empty()) {
        turn_request.request.prompt = turn_request.orchestration_config.prompt;
    }
    common_agent_scope_apply(turn_request.scope, turn_request.request);

    common_agent_runtime_host_inputs inputs{
        common_agent_runtime_host_mode::agent,
        context.memory_store,
        context.plan_store,
        std::move(turn_request),
        context.current_plan_id,
        context.installed_blueprint_candidates,
        &context.memories,
        context.tooling,
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
        inputs.tooling,
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

    auto * inference_session = session.active_inference_session();
    if (inference_session == nullptr || !inference_session->inference) {
        error = "runtime host failed to initialize an inference context";
        return false;
    }

    auto execution = make_agent_runtime_host_execution(inputs, *inference_session->inference);
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
        case common_agent_runtime_host_mode::agent: {
            if (execution.plan_store == nullptr) {
                error = "agent runtime host requires a plan store";
                return false;
            }
            if (execution.current_plan_id == nullptr) {
                error = "agent runtime host requires a current plan id";
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
                execution.turn_request.request.policy_pack,
                memories,
                execution.turn_request.memory_scope,
                execution.turn_request.memory_enabled,
                execution.tooling,
                {},
                {},
                false,
                execution.turn_request.request.input_resources,
            };
            driver_execution.execution_control = execution.turn_request.execution_control;
            driver_execution.research_should_stop = execution.turn_request.request.research_should_stop;
            driver_execution.research_stop_reason = execution.turn_request.request.research_stop_reason;
            driver_execution.explicit_memory_candidate = execution.turn_request.request.explicit_memory_candidate;
            driver_execution.explicit_memory_confirmed = execution.turn_request.request.explicit_memory_confirmed;
            return run_agent_runtime_driver(driver_execution, result, error);
        }

        case common_agent_runtime_host_mode::chat: {
            common_agent_chat_runtime_execution chat_execution{
                execution.inference,
                execution.turn_request.request,
                execution.turn_request.generation_options,
                {execution.turn_request.policy.max_tool_rounds,
                    execution.turn_request.runtime_config.max_continuations},
                execution.tooling,
                execution.turn_request.execution_control,
            };
            return run_agent_chat_runtime(chat_execution, result, error);
        }
    }

    error = "unsupported runtime host mode";
    return false;
}
