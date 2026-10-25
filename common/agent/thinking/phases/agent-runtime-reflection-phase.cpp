#include "agent/agent-runtime-context.h"
#include "plan/plan-context.h"
#include "plan/plan-workflow-state.h"
#include "plan/plan-in-memory.h"
#include <algorithm>
#include <map>

bool evaluate_common_agent_reflection_phase(
        common_agent_runtime_turn_context & context,
        common_plan_state & plan,
        const std::string & draft,
        const std::set<std::string> & executed_step_ids,
        common_reflection_result & reflection) {
    if (context.tools != nullptr && context.tools->has_execution_envelope() &&
            !context.tools->set_execution_phase(common_agent_execution_phase::reflection)) {
        context.error = "reflection phase is not permitted by the route execution envelope";
        return false;
    }
    if (!common_agent_runtime_transition_cognitive_mode(
            context, common_agent_cognitive_mode::reflect,
            "review execution result", "host", common_runtime_trace_stage::reflection)) {
        context.error = "invalid cognitive-mode transition into reflection";
        return false;
    }
    reflection = context.reflector.evaluate_result(
        context.request, plan, draft, context.error);
    if (reflection.generation) {
        context.result.generation_records.push_back(
            common_agent_generation_record_from_result(
                common_agent_generation_stage::reflection,
                *reflection.generation));
    }
    if (!context.error.empty()) return false;

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
    if (context.tools != nullptr && context.tools->has_execution_envelope()) {
        const auto phase = escalation.to_mode == common_agent_thinking_mode::research
            ? common_agent_execution_phase::research
            : common_agent_execution_phase::deliberate;
        if (!context.tools->set_execution_phase(phase)) {
            outcome.denied = true;
            context.emit_full_event({
                common_agent_event_type::thinking_escalation_denied,
                "route execution envelope denied phase escalation", {}, plan.id});
            return outcome;
        }
    }
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

bool prepare_common_agent_reflection_plan_repair(
        common_agent_runtime_turn_context & context,
        const common_plan_state & plan,
        common_planner & planner,
        common_reflection_result & reflection) {
    if (plan.status == common_plan_status::completed) {
        context.error = "accepted completed plan requires an explicit continuation";
        return false;
    }
    if (!context.tools || reflection.revision_guidance.empty()) {
        context.error = "plan repair requires authorized tools and specific revision guidance";
        return false;
    }
    const bool declared_graph = plan.workflow_definition && !plan.workflow_definition->transitions.empty();
    std::vector<common_plan_step> steps;
    if (declared_graph) {
        // Only the existing host continuation provider can select graph edges.
        const auto * observed = common_plan_workflow_observation(plan, {});
        const auto * source = observed ? common_plan_find_tool_observation_step(plan, *observed) : nullptr;
        if (!context.request.workflow_continuation || !observed || !source) {
            context.error = "this repair requires an available declared workflow continuation";
            return false;
        }
        if (!context.request.workflow_continuation(plan, source->id, *observed, steps, context.error)) return false;
    } else {
        auto request = context.request;
        request.plan_id = plan.id;
        request.route_binding = plan.route_binding;
        request.workflow_definition = plan.workflow_definition;
        request.blueprint_definition.reset();
        request.blueprint_plan_id.reset();
        request.require_tool_execution = true;
        request.prompt += "\n[Host repair request]\nPropose only the additional read operations needed to correct the evidence. "
            "Keep the selected route, source permissions and host argument bindings. "
            "Do not repeat writes or declare the previous response accepted.\n";
        for (const auto & guidance : reflection.revision_guidance) request.prompt += "Correction: " + guidance + "\n";
        request.prompt += common_plan_render_tool_observations(plan, {2048});
        auto proposal = planner.create_plan_result(request, context.error);
        if (proposal.generation) {
            context.result.generation_records.push_back(
                common_agent_generation_record_from_result(common_agent_generation_stage::planner, *proposal.generation));
            if (proposal.generation->status != common_agent_generation_status::completed) {
                context.result.response_generation_status = proposal.generation->status;
                context.result.response_stop_reason = proposal.generation->stop_reason;
            }
        }
        if (!context.error.empty()) return false;
        steps = proposal.plan.steps;
        if (!proposal.operations.empty()) {
            steps.clear();
            for (const auto & operation : proposal.operations) {
                if (operation.kind == common_plan_operation_kind::add_step && operation.step) steps.push_back(*operation.step);
                else { context.error = "repair planner may only propose additional steps"; return false; }
            }
        }
    }
    std::map<std::string, std::string> identities;
    for (const auto & step : steps) {
        if (common_plan_step_effective_mode(step) == common_plan_step_mode::final_response) continue;
        if (step.id.empty() || !identities.emplace(step.id,
                declared_graph ? step.id : "repair:" + std::to_string(plan.version) + ":" + std::to_string(identities.size())).second) {
            context.error = "repair contains missing or duplicate step identities";
            return false;
        }
    }
    if (identities.empty()) { context.error = "repair planner returned no executable work"; return false; }
    const auto remap_bindings = [&](auto && self, nlohmann::ordered_json & value) -> void {
        if (value.is_object()) {
            if (value.contains("$from_step") && value["$from_step"].is_string()) {
                const auto found = identities.find(value["$from_step"].get<std::string>());
                if (found != identities.end()) value["$from_step"] = found->second;
            }
            for (auto & field : value.items()) self(self, field.value());
        } else if (value.is_array()) for (auto & item : value) self(self, item);
    };
    common_plan_state candidate = plan;
    std::vector<common_plan_operation> additions;
    for (auto step : steps) {
        if (common_plan_step_effective_mode(step) == common_plan_step_mode::final_response) continue;
        if (!step.tool_call || !context.tools->is_available(step.tool_call->name) ||
                !context.tools->is_read_only(step.tool_call->name)) {
            context.error = "additional evidence repair must use available read-only tools";
            return false;
        }
        step.id = identities.at(step.id);
        step.status = common_plan_step_status::pending;
        if (!declared_graph && step.semantic_alias) step.semantic_alias = "repair:" + std::to_string(plan.version) + ":" + *step.semantic_alias;
        for (auto & dependency : step.depends_on) {
            const auto found = identities.find(dependency);
            if (found != identities.end()) dependency = found->second;
            else if (std::none_of(plan.steps.begin(), plan.steps.end(), [&](const auto & existing) {
                    return existing.id == dependency && existing.status == common_plan_step_status::completed;
                })) { context.error = "repair dependency is unavailable"; return false; }
        }
        auto arguments = nlohmann::ordered_json::parse(step.tool_call->arguments_json, nullptr, false);
        if (!arguments.is_object()) { context.error = "repair arguments must be an object"; return false; }
        remap_bindings(remap_bindings, arguments);
        step.tool_call->arguments_json = arguments.dump();
        candidate.steps.push_back(step);
        common_plan_operation operation;
        operation.kind = common_plan_operation_kind::add_step;
        operation.step = std::move(step);
        operation.reason_summary = "additional evidence requested by reflection";
        additions.push_back(std::move(operation));
    }
    if (!context.tools->validate_plan(candidate, context.error)) return false;
    // Reuse the actual plan policy for preflight, including dependency bounds.
    common_plan_in_memory_store preview;
    if (!preview.open("", context.error) || !preview.create(plan, context.error)) return false;
    auto projected = plan;
    for (auto operation : additions) {
        operation.plan_id = projected.id;
        operation.expected_version = projected.version;
        if (!preview.apply(operation, projected, context.error)) return false;
    }
    reflection.proposed_plan_operations = std::move(additions);
    return true;
}
