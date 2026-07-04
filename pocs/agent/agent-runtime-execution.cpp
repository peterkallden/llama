#include "agent-runtime-execution.h"

#include "agent-plan-orchestration.h"
#include "agent-runtime-assembly.h"

#include <cstdio>

common_agent_runtime_policy make_agent_runtime_policy(const args & options) {
    common_agent_runtime_policy policy;
    policy.agent_inference_backend = options.agent_inference_backend;
    policy.tool_profile = options.tool_profile;
    policy.memory_learn = options.memory_learn;
    policy.memory_learn_show_candidate = options.memory_learn_show_candidate;
    policy.plan_show_summary = options.plan_show_summary;
    policy.agent_trace = options.agent_trace;
    policy.enable_reflection = options.reflection_mode == "always";
    policy.max_iterations = policy.enable_reflection ? 2 : 1;
    policy.max_reflection_rounds = policy.enable_reflection ? 1 : 0;
    policy.max_tool_rounds = options.max_tool_rounds;
    policy.allow_policy_gated_tool_proposals =
        options.tool_profile == "memory" || options.tool_profile == "research";
    return policy;
}

common_agent_runtime_driver_execution make_agent_runtime_driver_execution(
    common_agent_runtime_driver_inputs & inputs,
    common_agent_inference & inference) {
    return {
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
        inputs.tools,
        inputs.profile_tools_active,
        inputs.tool_view,
    };
}

common_agent_request make_agent_runtime_driver_request(
    const common_agent_runtime_driver_execution & execution) {
    common_agent_request request;
    request.memories = execution.memories;
    request.enable_memory = execution.memory_enabled;
    request.enable_planning = true;
    request.enable_reflection = execution.policy.enable_reflection;
    request.memory_scope = execution.memory_scope;
    request.plan_scope = execution.scope.plan_scope;
    request.prompt = execution.orchestration_config.prompt;
    if (!execution.current_plan_id.empty()) {
        request.plan_id = execution.current_plan_id;
    }
    common_agent_scope_apply(execution.scope, request);
    request.max_iterations = execution.policy.max_iterations;
    request.max_reflection_rounds = execution.policy.max_reflection_rounds;
    request.max_tool_batches = execution.profile_tools_active ? execution.policy.max_tool_rounds : 0;
    request.allow_policy_gated_tool_proposals = execution.policy.allow_policy_gated_tool_proposals;
    return request;
}

bool run_agent_runtime_driver_session(
    common_agent_runtime_driver_inputs & inputs,
    common_agent_runtime_session & session,
    common_agent_result & result,
    std::string & error) {
    agent_inference_backend inference_backend_kind = agent_inference_backend::cli;
    if (!parse_agent_inference_backend(inputs.policy.agent_inference_backend, inference_backend_kind)) {
        error = "unsupported --agent-inference-backend: " + inputs.policy.agent_inference_backend;
        return false;
    }

    if (!initialize_agent_runtime_session(
            inputs.inference_options,
            inference_backend_kind,
            inputs.memory_enabled,
            inputs.fallback_reason,
            session,
            error)) {
        return false;
    }

    auto execution = make_agent_runtime_driver_execution(inputs, *session.inference_session.inference);
    return run_agent_runtime_driver(execution, result, error);
}

bool run_agent_runtime_driver(
    common_agent_runtime_driver_execution & execution,
    common_agent_result & result,
    std::string & error) {
    if (execution.profile_tools_active && execution.tool_view == nullptr) {
        error = "profile tool execution requires a resolved tool view";
        return false;
    }

    if (!maybe_auto_select_plan(
            execution.inference,
            execution.runtime_config.generation_config,
            execution.orchestration_config,
            execution.current_plan_id,
            execution.scope,
            execution.plan_store,
            error)) {
        return false;
    }

    if (!maybe_auto_select_blueprint(
            execution.inference,
            execution.runtime_config.generation_config,
            execution.orchestration_config,
            execution.current_plan_id,
            execution.scope,
            execution.plan_store,
            execution.installed_blueprint_candidates,
            execution.profile_tools_active,
            execution.tool_view,
            error)) {
        return false;
    }

    auto assembly = make_agent_runtime_assembly(
        execution.memory_store,
        execution.plan_store,
        execution.inference,
        execution.runtime_config,
        execution.tools,
        execution.tool_view);

    const common_agent_request request = make_agent_runtime_driver_request(execution);
    result = assembly.runtime->run(request);
    if (!result.error.empty()) {
        error = "agent runtime failed: " + result.error;
        return false;
    }

    if (execution.policy.memory_learn == "post-turn") {
        const auto * candidate = result.learned_memory_candidate ? &*result.learned_memory_candidate : nullptr;
        fprintf(stderr, "audit: memory_learn summary=%s plan=%s candidate=%s confidence=%.2f reuse=%.2f related=%zu\n",
            result.memory_learning_summary.c_str(), result.plan_id ? result.plan_id->c_str() : "",
            candidate ? common_memory_kind_name(candidate->kind) : "none", candidate ? candidate->confidence : 0.0f,
            candidate ? candidate->expected_reuse : 0.0f, result.memory_learning_related_count);
        if (execution.policy.memory_learn_show_candidate && candidate) {
            fprintf(stderr, "memory_learn candidate: kind=%s content=%s rationale=%s\n",
                common_memory_kind_name(candidate->kind), candidate->content.c_str(), candidate->rationale.c_str());
        }
    }

    if (execution.policy.plan_show_summary && result.plan_id) {
        std::string plan_error;
        const auto plan = execution.plan_store.get(*result.plan_id, plan_error);
        if (plan) {
            fprintf(stderr, "plan: id=%s version=%llu steps=%zu observations=%zu reflected=%s revised=%s\n",
                plan->id.c_str(), (unsigned long long) plan->version, plan->steps.size(), plan->observations.size(),
                result.reflected ? "yes" : "no", result.revised ? "yes" : "no");
        }
    }

    if (execution.policy.agent_trace) {
        for (const auto & event : result.events) {
            fprintf(stderr, "agent: event=%d plan=%s detail=%s\n", (int) event.type,
                event.plan_id ? event.plan_id->c_str() : "", event.detail.c_str());
        }
    }

    error.clear();
    return true;
}
