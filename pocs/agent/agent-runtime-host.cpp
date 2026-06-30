#include "agent-runtime-host.h"

namespace {

const std::vector<common_blueprint_candidate> k_empty_blueprints;
const std::vector<common_memory_hit> k_empty_memories;
const std::string k_empty_fallback_reason;

} // namespace

common_agent_runtime_host_inputs make_agent_runtime_host_chat_inputs(
        common_agent_runtime_host_build_context & context) {
    common_agent_request request = std::move(context.request);
    request.prompt = context.options.prompt;
    request.enable_memory = context.memory_enabled;
    request.memories = context.memories;
    common_agent_scope_apply(context.scope, request);

    common_agent_generation_options generation_options = context.generation_options;
    generation_options.n_predict = context.options.n_predict;

    return {
        common_agent_runtime_host_mode::chat,
        context.memory_store,
        nullptr,
        make_agent_inference_options(context.options),
        make_agent_runtime_policy(context.options),
        make_agent_runtime_config(context.options),
        {},
        nullptr,
        nullptr,
        nullptr,
        &context.memories,
        context.memory_scope,
        context.memory_enabled,
        &context.fallback_reason,
        std::move(request),
        generation_options,
        context.tools,
        context.profile_tools_active,
        context.tool_registry,
        context.tool_handler,
    };
}

common_agent_runtime_host_inputs make_agent_runtime_host_mini_inputs(
        common_agent_runtime_host_build_context & context,
        const common_agent_orchestration_config & orchestration_config) {
    common_agent_request request = std::move(context.request);
    request.prompt = context.options.prompt;
    request.enable_memory = context.memory_enabled;
    request.memories = context.memories;
    common_agent_scope_apply(context.scope, request);

    return {
        common_agent_runtime_host_mode::mini,
        context.memory_store,
        context.plan_store,
        make_agent_inference_options(context.options),
        make_agent_runtime_policy(context.options),
        make_agent_runtime_config(context.options),
        orchestration_config,
        context.current_plan_id,
        &context.scope,
        context.installed_blueprint_candidates,
        &context.memories,
        context.memory_scope,
        context.memory_enabled,
        &context.fallback_reason,
        std::move(request),
        context.generation_options,
        context.tools,
        context.profile_tools_active,
        context.tool_registry,
        context.tool_handler,
    };
}

common_agent_runtime_host_execution make_agent_runtime_host_execution(
        common_agent_runtime_host_inputs & inputs,
        common_agent_inference & inference) {
    return {
        inputs.mode,
        inputs.memory_store,
        inputs.plan_store,
        inference,
        inputs.policy,
        inputs.runtime_config,
        inputs.orchestration_config,
        inputs.current_plan_id,
        inputs.scope,
        inputs.installed_blueprint_candidates,
        inputs.memories,
        inputs.memory_scope,
        inputs.memory_enabled,
        inputs.request,
        inputs.generation_options,
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
    if (!parse_agent_inference_backend(inputs.policy.agent_inference_backend, inference_backend_kind)) {
        error = "unsupported --agent-inference-backend: " + inputs.policy.agent_inference_backend;
        return false;
    }

    const std::string & fallback_reason = inputs.fallback_reason ? *inputs.fallback_reason : k_empty_fallback_reason;
    if (!initialize_agent_runtime_session(
            inputs.inference_options,
            inference_backend_kind,
            inputs.memory_enabled,
            fallback_reason,
            session,
            error)) {
        return false;
    }

    auto execution = make_agent_runtime_host_execution(inputs, *session.inference_session.inference);
    return run_agent_runtime_host(execution, result, error);
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
                execution.policy,
                execution.runtime_config,
                execution.orchestration_config,
                *execution.current_plan_id,
                *execution.scope,
                blueprints,
                memories,
                execution.memory_scope,
                execution.memory_enabled,
                execution.tools,
                execution.profile_tools_active,
                execution.tool_registry,
            };
            return run_agent_runtime_driver(driver_execution, result, error);
        }

        case common_agent_runtime_host_mode::chat: {
            common_agent_chat_runtime_execution chat_execution{
                execution.inference,
                execution.request,
                execution.generation_options,
                {execution.policy.max_tool_rounds},
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
