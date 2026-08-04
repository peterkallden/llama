#include "agent-runtime-execution.h"

#include "../runtime/agent-plan-orchestration.h"
#include "../runtime/agent-runtime-assembly.h"

#include <algorithm>
#include <cstdio>
#include <sstream>

namespace {

void append_unique_strings(std::vector<std::string> & target, const std::vector<std::string> & values) {
    for (const auto & value : values) {
        if (std::find(target.begin(), target.end(), value) == target.end()) target.push_back(value);
    }
}

void append_runtime_result(common_agent_result & aggregate, const common_agent_result & slice) {
    if (!slice.response.empty()) {
        if (!aggregate.response.empty()) aggregate.response += "\n";
        aggregate.response += slice.response;
    }
    aggregate.total_decoded_tokens += slice.total_decoded_tokens;
    aggregate.response_decoded_tokens += slice.response_decoded_tokens;
    aggregate.reasoning_decoded_tokens += slice.reasoning_decoded_tokens;
    aggregate.response_generation_status = slice.response_generation_status;
    aggregate.response_stop_reason = slice.response_stop_reason;
    aggregate.plan_id = slice.plan_id;
    aggregate.plan_version = slice.plan_version;
    aggregate.reflected = aggregate.reflected || slice.reflected;
    aggregate.revised = aggregate.revised || slice.revised;
    aggregate.limit_reached = slice.limit_reached;
    aggregate.memory_learning_related_count = slice.memory_learning_related_count;
    aggregate.memory_learning_summary = slice.memory_learning_summary;
    aggregate.learned_memory_candidate = slice.learned_memory_candidate;
    aggregate.learning_signals.insert(
        aggregate.learning_signals.end(), slice.learning_signals.begin(), slice.learning_signals.end());
    append_unique_strings(aggregate.memory_ids, slice.memory_ids);
    aggregate.generation_records.insert(
        aggregate.generation_records.end(), slice.generation_records.begin(), slice.generation_records.end());
    aggregate.events.insert(aggregate.events.end(), slice.events.begin(), slice.events.end());
    aggregate.trace.insert(aggregate.trace.end(), slice.trace.begin(), slice.trace.end());
    aggregate.research_result = slice.research_result;
    aggregate.research_verification = slice.research_verification;
    aggregate.continuation_checkpoint = slice.continuation_checkpoint;
}

std::string make_continuation_prompt(
        const common_agent_result & slice,
        const common_plan_state & plan) {
    std::ostringstream prompt;
    prompt << "Continue the same bounded agent task from the existing plan.\n"
           << "Do not create a new plan, repeat completed work, or treat this as a new user request.\n"
           << "The previous generation reached its completion limit.\n"
           << "plan_id=" << plan.id << "\n"
           << "plan_version=" << plan.version << "\n";
    if (plan.active_step_id) prompt << "active_step_id=" << *plan.active_step_id << "\n";
    if (plan.next_action) prompt << "next_action=" << *plan.next_action << "\n";
    if (!slice.response.empty()) {
        constexpr size_t max_fragment_chars = 4096;
        const size_t offset = slice.response.size() > max_fragment_chars
            ? slice.response.size() - max_fragment_chars : 0;
        prompt << "Continue after this bounded previous output fragment:\n"
               << slice.response.substr(offset) << "\n";
    }
    prompt << "Produce only the next bounded result or the final answer.\n";
    return prompt.str();
}

bool make_continuation_checkpoint(
        const common_agent_runtime_driver_execution & execution,
        const common_agent_result & result,
        size_t sequence,
        common_agent_continuation_checkpoint & checkpoint,
        std::string & error) {
    if (result.response_stop_reason != common_agent_generation_stop_reason::limit ||
            !result.plan_id || execution.scope.turn_id.empty()) {
        return false;
    }
    const auto plan = execution.plan_store.get(*result.plan_id, error);
    if (!plan) {
        if (error.empty()) error = "continuation checkpoint requires an existing plan";
        return false;
    }
    checkpoint = {};
    checkpoint.checkpoint_id = "checkpoint:" + execution.scope.turn_id + ":" +
        plan->id + ":" + std::to_string(plan->version);
    checkpoint.request_id = "turn:" + execution.scope.turn_id;
    checkpoint.turn_id = execution.scope.turn_id;
    checkpoint.plan_id = plan->id;
    checkpoint.plan_version = plan->version;
    checkpoint.active_step_id = plan->active_step_id.value_or(std::string{});
    checkpoint.next_action = plan->next_action.value_or(std::string{});
    checkpoint.sequence = sequence;
    checkpoint.reason = common_agent_continuation_reason::completion_limit;
    for (const auto & step : plan->steps) {
        if (step.status == common_plan_step_status::completed) checkpoint.completed_step_ids.push_back(step.id);
    }
    for (const auto & observation : plan->observations) {
        checkpoint.resource_refs.insert(
            checkpoint.resource_refs.end(), observation.resource_refs.begin(), observation.resource_refs.end());
    }
    if (!common_agent_continuation_checkpoint_valid(checkpoint, error)) return false;
    return true;
}

} // namespace

static void apply_explicit_deliberation_policy(
        const common_agent_deliberation_policy & policy,
        common_agent_request & request) {
    // Reflective remains the compatibility baseline for existing callers. The
    // deeper modes explicitly opt into the stronger runtime guarantees.
    if (policy.mode == common_agent_thinking_mode::reflective) {
        return;
    }

    request.enable_planning = true;
    request.enable_reflection = true;
    request.max_reflection_rounds = std::max<size_t>(
        request.max_reflection_rounds,
        static_cast<size_t>(std::max(0, policy.max_reflection_rounds)));
    request.max_iterations = std::max<size_t>(
        request.max_iterations,
        static_cast<size_t>(1 + std::max(0, policy.max_plan_revisions)));
    request.max_tool_batches = std::max<size_t>(
        request.max_tool_batches,
        static_cast<size_t>(std::max(0, policy.max_tool_rounds)));
}

common_agent_runtime_policy make_agent_runtime_policy(common_agent_runtime_policy_build_config options) {
    common_agent_runtime_policy policy;
    policy.deliberation_policy = make_common_agent_deliberation_policy(
        common_agent_thinking_mode::reflective);
    policy.agent_inference_backend = std::move(options.agent_inference_backend);
    policy.tool_profile = std::move(options.tool_profile);
    policy.memory_learn = std::move(options.memory_learn);
    policy.memory_learn_show_candidate = options.memory_learn_show_candidate;
    policy.plan_show_summary = options.plan_show_summary;
    policy.agent_trace = options.agent_trace;
    policy.enable_reflection = true;
    policy.max_iterations = 2;
    policy.max_reflection_rounds = 1;
    policy.max_tool_rounds = options.max_tool_rounds;
    common_tool_profile_snapshot profile_snapshot;
    std::string profile_error;
    if (resolve_common_tool_profile_snapshot(
            policy.tool_profile,
            options.tool_capabilities,
            options.tool_profiles,
            profile_snapshot,
            profile_error)) {
        policy.allow_policy_gated_tool_proposals =
            profile_snapshot.allow_policy_gated_writes.value_or(false);
    }
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
        inputs.policy_pack,
        inputs.memories,
        inputs.memory_scope,
        inputs.memory_enabled,
        inputs.tooling,
        inputs.input_resources,
        inputs.research_should_stop,
        inputs.research_stop_reason,
        inputs.explicit_memory_candidate,
        inputs.explicit_memory_confirmed,
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
    request.input_resources = execution.input_resources;
    if (!execution.current_plan_id.empty()) {
        request.plan_id = execution.current_plan_id;
    }
    request.policy_pack = execution.policy_pack;
    common_agent_scope_apply(execution.scope, request);
    request.max_iterations = execution.policy.max_iterations;
    request.max_reflection_rounds = execution.policy.max_reflection_rounds;
    request.max_tool_batches = execution.tooling.profile_tools_active ? execution.policy.max_tool_rounds : 0;
    request.allow_policy_gated_tool_proposals = execution.policy.allow_policy_gated_tool_proposals;
    request.deliberation_policy = execution.policy.deliberation_policy;
    request.research_should_stop = execution.research_should_stop;
    request.research_stop_reason = execution.research_stop_reason;
    request.explicit_memory_candidate = execution.explicit_memory_candidate;
    request.explicit_memory_confirmed = execution.explicit_memory_confirmed;
    apply_explicit_deliberation_policy(request.deliberation_policy, request);
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

    auto * inference_session = session.active_inference_session();
    if (inference_session == nullptr || !inference_session->inference) {
        error = "runtime session failed to initialize an inference context";
        return false;
    }

    auto execution = make_agent_runtime_driver_execution(inputs, *inference_session->inference);
    return run_agent_runtime_driver(execution, result, error);
}

bool run_agent_runtime_driver(
        common_agent_runtime_driver_execution & execution,
        common_agent_result & result,
        std::string & error) {
    execution.pre_turn_events.clear();
    execution.pre_turn_trace.clear();
    if (execution.tooling.profile_tools_active && execution.tooling.tool_view == nullptr) {
        error = "profile tool execution requires a resolved tool view";
        return false;
    }

    const common_agent_orchestration_runtime_context orchestration_context{
        execution.inference,
        execution.runtime_config.generation_config,
        execution.orchestration_config,
        execution.current_plan_id,
        execution.scope,
        execution.plan_store,
        execution.installed_blueprint_candidates,
        &execution.policy_pack,
        &execution.tooling,
        execution.pre_turn_events,
        execution.pre_turn_trace,
    };

    if (!maybe_auto_select_plan(orchestration_context, error)) {
        return false;
    }

    if (!maybe_auto_select_blueprint(orchestration_context, error)) {
        return false;
    }

    const std::string original_prompt = execution.orchestration_config.prompt;
    common_agent_result aggregate;
    size_t continuation_count = 0;
    while (true) {
        auto assembly = make_agent_runtime_assembly(
            execution.memory_store,
            execution.plan_store,
            execution.inference,
            execution.runtime_config,
            execution.tooling.tools,
            execution.tooling.tool_view);

        const common_agent_request request = make_agent_runtime_driver_request(execution);
        const auto slice = assembly.runtime->run(request);
        if (!slice.error.empty()) {
            error = "agent runtime failed: " + slice.error;
            return false;
        }
        append_runtime_result(aggregate, slice);

        if (slice.response_stop_reason != common_agent_generation_stop_reason::limit ||
                !slice.plan_id ||
                continuation_count >= execution.runtime_config.max_continuations) {
            result = std::move(aggregate);
            break;
        }

        std::string plan_error;
        const auto plan = execution.plan_store.get(*slice.plan_id, plan_error);
        if (!plan) {
            result = std::move(aggregate);
            result.error = plan_error.empty()
                ? "continuation could not resolve the current plan"
                : plan_error;
            error = result.error;
            return false;
        }
        execution.current_plan_id = plan->id;
        execution.orchestration_config.prompt = make_continuation_prompt(slice, *plan);
        ++continuation_count;
    }

    execution.orchestration_config.prompt = original_prompt;
    if (result.response_stop_reason == common_agent_generation_stop_reason::limit &&
            !result.continuation_checkpoint) {
        common_agent_continuation_checkpoint checkpoint;
        std::string checkpoint_error;
        if (make_continuation_checkpoint(
                execution, result, continuation_count + 1, checkpoint, checkpoint_error)) {
            result.continuation_checkpoint = std::move(checkpoint);
        } else if (!checkpoint_error.empty()) {
            result.error = checkpoint_error;
            error = result.error;
            return false;
        }
    }
    result.events.insert(result.events.begin(), execution.pre_turn_events.begin(), execution.pre_turn_events.end());
    result.trace.insert(result.trace.begin(), execution.pre_turn_trace.begin(), execution.pre_turn_trace.end());

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
        for (const auto & entry : result.trace) {
            fprintf(stderr, "agent: trace stage=%s kind=%s plan=%s step=%s tool=%s observation=%s detail=%s\n",
                common_runtime_trace_stage_name(entry.stage),
                common_runtime_trace_kind_name(entry.kind),
                entry.plan_id.c_str(),
                entry.step_id.c_str(),
                entry.tool_name.c_str(),
                entry.observation_id.c_str(),
                entry.detail.c_str());
        }
        for (const auto & event : result.events) {
            fprintf(stderr, "agent: event=%d plan=%s detail=%s\n", (int) event.type,
                event.plan_id ? event.plan_id->c_str() : "", event.detail.c_str());
        }
    }

    error.clear();
    return true;
}
