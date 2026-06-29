#include "agent-runtime-execution.h"

#include "agent-plan-orchestration.h"
#include "agent-runtime-assembly.h"

#include <cstdio>

common_agent_request make_agent_cli_runtime_request(
    const common_agent_cli_runtime_execution & execution) {
    common_agent_request request;
    request.prompt = execution.options.prompt;
    request.memories = execution.memories;
    request.enable_memory = execution.memory_enabled;
    request.enable_planning = true;
    request.enable_reflection = execution.options.reflection_mode == "always";
    request.memory_scope = execution.memory_scope;
    request.plan_scope = execution.scope.plan_scope;
    if (!execution.options.plan_id.empty()) {
        request.plan_id = execution.options.plan_id;
    }
    common_agent_scope_apply(execution.scope, request);
    request.max_iterations = execution.options.reflection_mode == "always" ? 2 : 1;
    request.max_reflection_rounds = execution.options.reflection_mode == "always" ? 1 : 0;
    request.max_tool_batches = execution.profile_tools_active ? execution.options.max_tool_rounds : 0;
    request.allow_policy_gated_tool_proposals =
        execution.options.tool_profile == "memory" || execution.options.tool_profile == "research";
    return request;
}

bool run_agent_cli_mini_runtime(
    common_agent_cli_runtime_execution & execution,
    common_agent_result & result,
    std::string & error) {
    if (!maybe_auto_select_plan(
            execution.inference,
            execution.options,
            execution.options,
            execution.scope,
            execution.plan_store,
            error)) {
        return false;
    }

    if (!maybe_auto_select_blueprint(
            execution.inference,
            execution.options,
            execution.options,
            execution.scope,
            execution.plan_store,
            execution.installed_blueprint_candidates,
            execution.profile_tools_active,
            execution.tool_registry,
            error)) {
        return false;
    }

    auto assembly = make_agent_runtime_assembly(
        execution.memory_store,
        execution.plan_store,
        execution.inference,
        execution.options,
        execution.tools,
        execution.tool_registry);

    const common_agent_request request = make_agent_cli_runtime_request(execution);
    result = assembly.runtime->run(request);
    if (!result.error.empty()) {
        error = "agent runtime failed: " + result.error;
        return false;
    }

    if (execution.options.memory_learn == "post-turn") {
        const auto * candidate = result.learned_memory_candidate ? &*result.learned_memory_candidate : nullptr;
        fprintf(stderr, "audit: memory_learn summary=%s plan=%s candidate=%s confidence=%.2f reuse=%.2f related=%zu\n",
            result.memory_learning_summary.c_str(), result.plan_id ? result.plan_id->c_str() : "",
            candidate ? common_memory_kind_name(candidate->kind) : "none", candidate ? candidate->confidence : 0.0f,
            candidate ? candidate->expected_reuse : 0.0f, result.memory_learning_related_count);
        if (execution.options.memory_learn_show_candidate && candidate) {
            fprintf(stderr, "memory_learn candidate: kind=%s content=%s rationale=%s\n",
                common_memory_kind_name(candidate->kind), candidate->content.c_str(), candidate->rationale.c_str());
        }
    }

    if (execution.options.plan_show_summary && result.plan_id) {
        std::string plan_error;
        const auto plan = execution.plan_store.get(*result.plan_id, plan_error);
        if (plan) {
            fprintf(stderr, "plan: id=%s version=%llu steps=%zu observations=%zu reflected=%s revised=%s\n",
                plan->id.c_str(), (unsigned long long) plan->version, plan->steps.size(), plan->observations.size(),
                result.reflected ? "yes" : "no", result.revised ? "yes" : "no");
        }
    }

    if (execution.options.agent_trace) {
        for (const auto & event : result.events) {
            fprintf(stderr, "agent: event=%d plan=%s detail=%s\n", (int) event.type,
                event.plan_id ? event.plan_id->c_str() : "", event.detail.c_str());
        }
    }

    error.clear();
    return true;
}
