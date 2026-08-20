#include "agent/agent-runtime-context.h"

bool evaluate_common_agent_reflection_phase(
        common_agent_runtime_turn_context & context,
        common_plan_state & plan,
        const std::string & draft,
        const std::set<std::string> & executed_step_ids,
        common_reflection_result & reflection) {
    reflection = context.reflector.evaluate_result(
        context.request, plan, draft, context.error);
    if (!context.error.empty()) return false;

    if (reflection.generation) {
        context.result.generation_records.push_back(
            common_agent_generation_record_from_result(
                common_agent_generation_stage::reflection,
                *reflection.generation));
    }

    const bool deeper_deliberation =
        context.request.deliberation_policy.mode != common_agent_thinking_mode::reflective;
    context.result.reflected = true;
    context.emit_event(
        common_agent_event_type::reflection_completed,
        "reflection completed");
    if (deeper_deliberation) {
        if (executed_step_ids.empty()) {
            context.emit_full_event({
                common_agent_event_type::step_reviewed,
                "deliberation step review completed", {}, plan.id,
                plan.active_step_id.value_or("")});
        } else {
            for (const auto & reviewed_step_id : executed_step_ids) {
                context.emit_full_event({
                    common_agent_event_type::step_reviewed,
                    "deliberation step review completed", {}, plan.id,
                    reviewed_step_id});
            }
        }
        context.emit_full_event({
            common_agent_event_type::answer_reviewed,
            "deliberation answer review completed", {}, plan.id});
    }
    context.emit_trace(
        common_runtime_trace_stage::reflection,
        common_runtime_trace_kind::completed,
        "reflection completed",
        plan.id,
        {});
    return true;
}

common_agent_reflection_escalation_result handle_common_agent_reflection_escalation(
        common_agent_runtime_turn_context & context,
        common_plan_state & plan,
        const common_reflection_result & reflection,
        const std::string & draft,
        std::vector<std::string> & guidance,
        size_t iteration,
        size_t & runtime_iteration_limit) {
    common_agent_reflection_escalation_result outcome;
    outcome.requested =
        reflection.next_action == common_agent_reflection_next_action::escalate_deliberate ||
        reflection.next_action == common_agent_reflection_next_action::escalate_research;
    if (!outcome.requested) return outcome;

    const auto requested_mode = reflection.next_action ==
        common_agent_reflection_next_action::escalate_research
        ? common_agent_thinking_mode::research
        : common_agent_thinking_mode::deliberate;
    context.emit_full_event({
        common_agent_event_type::thinking_escalation_requested,
        std::string("post_reflection target=") + common_agent_thinking_mode_name(requested_mode),
        {}, plan.id});
    context.emit_trace(
        common_runtime_trace_stage::reflection,
        common_runtime_trace_kind::decided,
        std::string("post-reflection escalation requested to ") +
            common_agent_thinking_mode_name(requested_mode),
        plan.id,
        {});

    const auto escalation = resolve_common_agent_reflection_escalation(
        context.request.deliberation_policy, reflection.next_action);
    if (!escalation.allowed || context.late_escalation_used) {
        outcome.denied = true;
        context.emit_full_event({
            common_agent_event_type::thinking_escalation_denied,
            escalation.summary + " phase=post_reflection reason=" +
                common_agent_escalation_reason_name(escalation.reason),
            {}, plan.id});
        context.emit_trace(
            common_runtime_trace_stage::reflection,
            common_runtime_trace_kind::failed,
            "post-reflection escalation denied",
            plan.id,
            {});
        return outcome;
    }

    context.request.deliberation_policy = context.policy_after_escalation(
        context.request.deliberation_policy, escalation.to_mode);
    context.request.max_reflection_rounds = std::max<size_t>(
        context.request.max_reflection_rounds,
        static_cast<size_t>(context.request.deliberation_policy.max_reflection_rounds));
    context.late_escalation_used = true;
    context.emit_full_event({
        common_agent_event_type::thinking_escalation_allowed,
        escalation.summary + " phase=post_reflection reason=" +
            common_agent_escalation_reason_name(escalation.reason),
        {}, plan.id});
    context.emit_full_event({
        common_agent_event_type::thinking_mode_resolved,
        std::string("thinking mode resolved to ") +
            common_agent_thinking_mode_name(context.request.deliberation_policy.mode) +
            " phase=post_reflection",
        {}, plan.id});

    if (context.request.deliberation_policy.mode == common_agent_thinking_mode::research) {
        if (!run_common_agent_research_phase(context)) {
            context.result.response = draft;
            outcome.stop_turn = true;
            return outcome;
        }
        if (!context.research_synthesis_context.empty()) {
            context.request.prompt += "\n\nHost-approved research synthesis context:\n";
            context.request.prompt += context.research_synthesis_context;
            if (context.request.prompt.size() > 8192) context.request.prompt.resize(8192);
        }
    }
    for (const auto & issue : reflection.issues) {
        if (!issue.description.empty()) guidance.push_back(issue.description);
    }
    guidance.insert(guidance.end(), reflection.revision_guidance.begin(), reflection.revision_guidance.end());
    context.result.revised = true;
    runtime_iteration_limit = std::max(runtime_iteration_limit, iteration + 2);
    outcome.continue_turn = true;
    return outcome;
}
