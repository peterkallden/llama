#include "agent/agent-runtime-context.h"
#include "agent/method-phases.h"
#include "plan/plan-context.h"
#include "plan/plan-workflow-state.h"
#include "plan/plan-in-memory.h"
#include "plan/plan-policy.h"
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

static bool prepare_reflection_plan_repair_attempt(
        common_agent_runtime_turn_context & context,
        const common_plan_state & plan,
        common_planner & planner,
        common_reflection_result & reflection,
        const std::string & previous_error,
        bool & validation_rejected) {
    validation_rejected=false;
    const bool method = plan.phase_review_required && !plan.workflow_definition;
    if (plan.status == common_plan_status::completed && !method) {
        context.error = "accepted completed plan requires an explicit continuation";
        return false;
    }
    if (!context.tools || (!method && reflection.revision_guidance.empty())) {
        context.error = "plan repair requires authorized tools and specific revision guidance";
        return false;
    }
    const bool declared_graph = plan.workflow_definition && !plan.workflow_definition->transitions.empty();
    common_agent_workflow_continuation steps;
    std::map<std::string,std::string> repair_targets;
    std::string selected_phase;
    if (declared_graph) {
        // Only the existing host continuation provider can select graph edges.
        const auto * observed = common_plan_workflow_observation(plan, {});
        const auto * source = observed ? common_plan_find_tool_observation_step(plan, *observed) : nullptr;
        if (!context.request.workflow_continuation || !observed || !source) {
            context.error = "this repair requires an available declared workflow continuation";
            return false;
        }
        if (!context.request.workflow_continuation(plan, source->id, *observed, steps, context.error)) return false;
        if (steps.terminal_choice) {
            context.error = "workflow repair requires an additional step, not a terminal choice";
            return false;
        }
    } else {
        auto request = context.request;
        request.plan_id = plan.id;
        request.route_binding = plan.route_binding;
        request.workflow_definition = plan.workflow_definition;
        request.blueprint_definition.reset();
        request.blueprint_plan_id.reset();
        request.plan_repair_targets.clear();
        request.require_tool_execution = !method;
        request.method_phase_choices.clear();
        if (method) {
            auto failed=nlohmann::ordered_json::array();
            size_t omitted=0;
            for (const auto & step:plan.steps) {
                if (step.status!=common_plan_step_status::failed || !step.tool_call) continue;
                if (request.plan_repair_targets.size()==8) {++omitted;continue;}
                const auto handle="r"+std::to_string(request.plan_repair_targets.size()+1);
                request.plan_repair_targets.emplace_back(handle,step.id);
                nlohmann::ordered_json target={{"repair",handle},{"tool",step.tool_call->name},
                    {"objective_excerpt",step.objective.substr(0,256)},
                    {"result_excerpt",step.result_summary.value_or("").substr(0,512)}};
                if (step.tool_call->arguments_json.size()<=1024) target["previous_arguments"]=step.tool_call->arguments_json;
                else target["previous_arguments_omitted"]=true;
                failed.push_back(std::move(target));
            }
            request.prompt += "\n[Host-bound failed operations; repair selects one of these handles]\n"+
                nlohmann::ordered_json{{"targets",failed},{"omitted_targets",omitted}}.dump();
            const size_t plan_limit=common_plan_policy_config{}.max_steps;
            request.method_remaining_new_steps=plan.steps.size()<plan_limit ? plan_limit-plan.steps.size() : 0;
            if (*request.method_remaining_new_steps==0 && request.plan_repair_targets.empty()) {
                context.error="method plan step capacity exhausted; unresolved phases remain in the saved plan";
                return false;
            }
        }
        if (!method) request.prompt += "\n[Host repair request]\nPropose only the additional read operations needed to correct the evidence. "
            "Keep the selected route, source permissions and host argument bindings. "
            "Do not repeat writes or declare the previous response accepted.\n";
        if (method) request.prompt += common_agent_render_method_progress(plan)+
            "\nThis is a method continuation, not a read-only answer repair. "
            "Plan only a bounded slice for one model-selected dependency-ready unresolved phase. "
            "Use normal family/tool selection within existing authority; a phase may use reasoning without tools. "
            "Retain local/external evidence and do not repeat completed writes. "
            "A failed verification remains failed until corrected and rerun with fresh source arguments. "
            "Never skip mandatory failed operations or invent a passed verification.\n"+
            common_plan_render_context(plan,{8192,true});
        if (method) {
            const auto ready=common_agent_method_ready_phases(plan);
            if (ready.empty()) { context.error="method has no dependency-ready unresolved phase";return false; }
            for (const auto & phase:ready)
                request.method_phase_choices.emplace_back(phase["handle"].get<std::string>(),phase["phase_id"].get<std::string>());
            request.prompt += "\n[Host-issued ready method phases]\n"+ready.dump()+
                "\nSelect exactly one phase using its handle, and plan the concrete work that resolves its remaining need. "
                "Use tool operations when new source evidence or execution is needed; reasoning is appropriate for existing sufficient evidence. "
                "Searching locates sources; read the relevant results before claiming their contents are inspected. "
                "Honor pagination and incomplete-coverage markers. Do not replay completed operations against unchanged inputs.\n";
        }
        for (const auto & guidance : reflection.revision_guidance) request.prompt += "Correction: " + guidance + "\n";
        request.prompt += common_plan_render_tool_observations(plan, {2048});
        if (!previous_error.empty()) request.prompt +=
            "\n[Host rejected the previous continuation; it was not executed]\n"+previous_error.substr(0,1024)+
            "\nGenerate one complete corrected slice. Do not replay the rejected proposal. "
            "Use new steps for new work and an issued repair handle only for an explicit replacement.\n";
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
        if (method && std::none_of(request.method_phase_choices.begin(),request.method_phase_choices.end(),
                [&](const auto & choice){return choice.second==proposal.selected_method_phase;})) {
            validation_rejected=true;
            context.error="planner must select one of the supplied ready method phases";return false;
        }
        selected_phase=proposal.selected_method_phase;
        validation_rejected=method;
        steps = proposal.plan.steps;
        if (!proposal.operations.empty()) {
            steps.clear();
            for (const auto & operation : proposal.operations) {
                if (operation.kind == common_plan_operation_kind::add_step && operation.step) steps.push_back(*operation.step);
                else if (method && operation.kind==common_plan_operation_kind::replace_step && operation.step && operation.step_id) {
                    if (!repair_targets.emplace(operation.step->id,*operation.step_id).second) {
                        validation_rejected=true;context.error="duplicate repair slice step";return false;
                    }
                    steps.push_back(*operation.step);
                }
                else { context.error = "repair planner may only propose additional steps"; return false; }
            }
        }
    }
    if (method) {
        validation_rejected=true;
        common_plan_state slice;slice.steps=steps;
        std::vector<common_plan_operation> operations;
        if (!common_agent_method_append_slice(plan,slice,operations,context.error,repair_targets)) return false;
        for (auto & operation:operations) {
            operation.reason_summary+="; method_phase="+selected_phase;
            operation.step->method_phase_id=selected_phase;
            operation.step->method_turn_id=context.request.turn_id;
        }
        if (!common_agent_method_validate_progress(plan,operations,selected_phase,context.request.turn_id,
                [&](const std::string & name) {return context.tools->is_read_only(name);},context.error)) return false;
        auto candidate=plan;
        for (const auto & operation:operations) {
            if (operation.step->tool_call && !context.tools->is_available(operation.step->tool_call->name)) {
                context.error="method continuation selected an unavailable tool";return false;
            }
            if (operation.kind==common_plan_operation_kind::add_step) candidate.steps.push_back(*operation.step);
            else for (auto & step:candidate.steps) if (step.id==operation.step->id) step=*operation.step;
        }
        if (!context.tools->validate_plan(candidate,context.error)) return false;
        common_plan_in_memory_store preview;
        if (!preview.open("",context.error) || !preview.create(plan,context.error)) {
            validation_rejected=false;return false;
        }
        auto projected=plan;
        for (auto operation:operations) {
            operation.plan_id=projected.id;operation.expected_version=projected.version;
            if (!preview.apply(operation,projected,context.error)) return false;
        }
        reflection.proposed_plan_operations=std::move(operations);
        validation_rejected=false;
        return true;
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

// Reuse the ordinary planner, tool view and preview store. A rejected slice
// never changes the live plan and gets at most one validation-feedback retry.
bool prepare_common_agent_reflection_plan_repair(
        common_agent_runtime_turn_context & context,
        const common_plan_state & plan,
        common_planner & planner,
        common_reflection_result & reflection) {
    const bool method=plan.phase_review_required && !plan.workflow_definition;
    if (!common_agent_runtime_transition_cognitive_mode(context,common_agent_cognitive_mode::frame,
            "prepare reflection plan repair","reflection",common_runtime_trace_stage::reflection)) return false;
    std::string previous_error;
    for (size_t attempt=0;attempt<(method?2u:1u);++attempt) {
        bool rejected=false;
        context.error.clear();
        if (prepare_reflection_plan_repair_attempt(context,plan,planner,reflection,previous_error,rejected)) return true;
        if (!rejected || !method) return false;
        previous_error=context.error;
        context.emit_trace(common_runtime_trace_stage::plan,common_runtime_trace_kind::failed,
            "method continuation validation rejected attempt="+std::to_string(attempt+1)+": "+previous_error,
            plan.id,{});
    }
    return false;
}
