#include "agent/agent-runtime.h"
#include "agent/memory-learning.h"
#include "plan/plan-bindings.h"
#include "plan/plan-goal.h"
#include "plan/plan-memory.h"
#include "plan/plan-scheduler.h"

#include <nlohmann/json.hpp>
#include <cctype>
#include <regex>
#include <set>

using json = nlohmann::ordered_json;

static void append_trace(
        common_agent_result & result,
        common_runtime_trace_stage stage,
        common_runtime_trace_kind kind,
        std::string detail,
        std::string plan_id = {},
        std::string step_id = {},
        std::string tool_name = {},
        std::string observation_id = {},
        std::string related_id = {}) {
    result.trace.push_back({
        stage,
        kind,
        std::move(detail),
        std::move(plan_id),
        std::move(step_id),
        std::move(tool_name),
        std::move(observation_id),
        std::move(related_id),
    });
}

static bool infer_calculator_expression(const std::string & text, std::string & expression) {
    static const std::regex arithmetic(R"((\(?\s*\d+(?:\.\d+)?(?:\s*[-+*/]\s*\d+(?:\.\d+)?)+\s*\)?))");
    std::smatch match;
    if (!std::regex_search(text, match, arithmetic) || match.size() < 2) {
        return false;
    }
    expression = match[1].str();
    return true;
}

static bool infer_memory_search_query(const std::string & prompt, std::string & query) {
    const auto first = prompt.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return false;
    }
    const auto last = prompt.find_last_not_of(" \t\r\n");
    query = prompt.substr(first, last - first + 1);
    return query.size() <= 1024;
}

static bool apply_request_objective(const common_agent_request & request, common_plan_state & plan, std::string & error) {
    const auto bounded = [](const std::string & value) { return value.size() <= 512; };
    const auto purpose = request.objective && !request.objective->purpose.empty() ? request.objective->purpose : request.prompt;
    const auto outcome = request.objective && !request.objective->desired_outcome.empty() ? request.objective->desired_outcome : request.prompt;
    if (purpose.empty() || outcome.empty() || !bounded(purpose) || !bounded(outcome)) {
        error = "agent objective requires bounded purpose and desired outcome";
        return false;
    }
    if (request.objective) {
        if (request.objective->success_criteria.size() > 8 || request.objective->constraints.size() > 8) {
            error = "agent objective exceeds bounded criteria or constraints";
            return false;
        }
        for (const auto & value : request.objective->success_criteria) if (value.empty() || !bounded(value)) { error = "agent objective contains invalid success criteria"; return false; }
        for (const auto & value : request.objective->constraints) if (value.empty() || !bounded(value)) { error = "agent objective contains invalid constraints"; return false; }
    }
    // Purpose is owned by the caller, rather than an untrusted planner
    // proposal. A planner may refine the executable goal, but cannot replace
    // why the turn was requested.
    plan.purpose = purpose;
    if (plan.goal.empty()) plan.goal = outcome;
    if (request.objective && !request.objective->success_criteria.empty()) {
        plan.success_criteria.clear();
        for (const auto & criterion : request.objective->success_criteria) {
            if (!plan.success_criteria.empty()) plan.success_criteria += "; ";
            plan.success_criteria += criterion;
        }
        if (plan.success_criteria.size() > 1024) { error = "agent objective success criteria are too long"; return false; }
    } else if (plan.success_criteria.empty()) {
        plan.success_criteria = "Provide a grounded, concise response.";
    }
    if (request.objective) {
        for (size_t index = 0; index < request.objective->constraints.size(); ++index) {
            plan.constraints.push_back({"objective-constraint-" + std::to_string(index + 1), request.objective->constraints[index], true});
        }
    }
    for (auto & step : plan.steps) if (step.intended_contribution.empty()) step.intended_contribution = step.objective;
    error.clear();
    return true;
}

static common_agent_failure tool_failure(
        const std::string & tool_name,
        const std::string & step_id,
        const std::string & evidence_id,
        const std::string & code,
        common_agent_failure_class classification,
        bool retryable,
        const std::string & safe_summary) {
    return {code, classification, "tool_execution", tool_name, step_id, evidence_id, retryable, safe_summary};
}

static json render_failure(const common_agent_failure & failure) {
    return {
        {"code", failure.code},
        {"class", common_agent_failure_class_name(failure.classification)},
        {"stage", failure.stage},
        {"tool", failure.tool_name},
        {"step_id", failure.step_id},
        {"retryable", failure.retryable},
        {"safe_summary", failure.safe_summary},
        {"evidence_id", failure.evidence_id},
    };
}

static common_agent_failure structured_tool_failure(const std::string & tool_name, const std::string & step_id,
        const std::string & evidence_id, const common_tool_execution_result & result) {
    const auto classification = result.failure_class == common_tool_failure_class::validation ? common_agent_failure_class::validation :
        result.failure_class == common_tool_failure_class::policy ? common_agent_failure_class::policy :
        result.failure_class == common_tool_failure_class::not_found ? common_agent_failure_class::not_found :
        result.failure_class == common_tool_failure_class::timeout ? common_agent_failure_class::timeout :
        result.failure_class == common_tool_failure_class::network ? common_agent_failure_class::network :
        result.failure_class == common_tool_failure_class::limit ? common_agent_failure_class::limit : common_agent_failure_class::execution;
    return tool_failure(tool_name, step_id, evidence_id, result.failure_code.empty() ? "tool.execution_failed" : result.failure_code,
        classification, result.retryable, result.safe_summary.empty() ? "The tool failed." : result.safe_summary);
}

static std::string next_tool_observation_id(const common_plan_state & plan, const std::string & step_id, const std::string & tool_name) {
    const std::string base = "tool:" + step_id + ":" + tool_name;
    const std::string attempt_prefix = base + ":attempt:";
    bool seen_base = false;
    size_t next_attempt = 2;
    for (const auto & observation : plan.observations) {
        if (observation.id == base) {
            seen_base = true;
            continue;
        }
        if (observation.id.rfind(attempt_prefix, 0) != 0) continue;
        seen_base = true;
        try {
            next_attempt = std::max(next_attempt, static_cast<size_t>(std::stoull(observation.id.substr(attempt_prefix.size())) + 1));
        } catch (const std::exception &) {
            next_attempt = std::max(next_attempt, static_cast<size_t>(3));
        }
    }
    return seen_base ? attempt_prefix + std::to_string(next_attempt) : base;
}

// Defaults are deliberately limited to deterministic read-only values. They
// reduce the amount a small model must emit, but never fabricate a write path,
// a mutation payload, or a selection among ambiguous results.
static void apply_safe_tool_defaults(const common_agent_request & request, common_registered_tool_call & call) {
    auto arguments = json::parse(call.arguments_json, nullptr, false);
    if (!arguments.is_object()) return;
    bool changed = false;
    const auto set_prompt_query = [&](size_t max_length) {
        if (arguments.contains("query")) return;
        std::string query;
        if (infer_memory_search_query(request.prompt, query) && query.size() <= max_length) { arguments["query"] = std::move(query); changed = true; }
    };
    if (call.name == "calculator" && !arguments.contains("expression")) {
        std::string expression;
        if (infer_calculator_expression(request.prompt, expression)) { arguments["expression"] = std::move(expression); changed = true; }
    } else if (call.name == "memory_search") {
        set_prompt_query(1024);
    } else if (call.name == "repository_search") {
        set_prompt_query(256);
        if (!arguments.contains("path")) { arguments["path"] = ""; changed = true; }
        if (!arguments.contains("max_results")) { arguments["max_results"] = 16; changed = true; }
    } else if (call.name == "web_search") {
        set_prompt_query(256);
        if (!arguments.contains("limit")) { arguments["limit"] = 5; changed = true; }
    } else if (call.name == "repository_read") {
        if (!arguments.contains("start_line")) { arguments["start_line"] = 1; changed = true; }
        if (!arguments.contains("end_line")) { arguments["end_line"] = 200; changed = true; }
    } else if (call.name == "resource_read") {
        if (!arguments.contains("max_bytes")) { arguments["max_bytes"] = 8192; changed = true; }
    } else if (call.name == "repository_list") {
        if (!arguments.contains("path")) { arguments["path"] = ""; changed = true; }
        if (!arguments.contains("depth")) { arguments["depth"] = 1; changed = true; }
    }
    if (changed) call.arguments_json = arguments.dump();
}

common_agent_runtime::common_agent_runtime(common_plan_store & store, common_planner & planner, common_action_executor & executor, common_reflection_engine & reflector, const common_agent_tool_runtime * tools, common_memory_post_turn_learner * memory_learner) : store(store), planner(planner), executor(executor), reflector(reflector), tools(tools), memory_learner(memory_learner) {}

common_agent_result common_agent_runtime::run(const common_agent_request & request) {
    common_agent_result result;
    std::string error;
    append_trace(result, common_runtime_trace_stage::turn, common_runtime_trace_kind::started,
        "agent runtime turn started", {}, {}, {}, {}, request.turn_id);
    if (!request.enable_planning) { result.error = "planning is disabled"; return result; }
    const auto normalize_planned_tool_step = [&](common_plan_step & step, bool degrade_on_any_invalid, std::string & detail) {
        detail.clear();
        if (!step.tool_call) return false;
        common_registered_tool_call call{step.tool_call->name, step.tool_call->arguments_json};
        apply_safe_tool_defaults(request, call);
        step.tool_call->arguments_json = call.arguments_json;
        if (!tools) {
            if (!degrade_on_any_invalid) return false;
            step.selected_tool.reset();
            step.tool_call.reset();
            step.mode = common_plan_step_mode::reasoning;
            step.required_evidence.clear();
            detail = "registered tool execution is unavailable";
            return true;
        }
        std::string validation_error;
        const bool policy_allowed = tools->is_read_only(call.name) || (request.allow_policy_gated_tool_proposals && tools->is_policy_gated(call.name));
        const bool valid_tool_call = policy_allowed && tools->validate(call, validation_error);
        if (valid_tool_call) return false;
        if (validation_error.empty()) validation_error = "tool is not approved by policy";
        const auto arguments = json::parse(call.arguments_json, nullptr, false);
        const bool empty_arguments = arguments.is_object() && arguments.empty();
        const bool incomplete_tool_call = validation_error == "required contract field is missing" && empty_arguments;
        if (!degrade_on_any_invalid && !incomplete_tool_call) return false;
        step.selected_tool.reset();
        step.tool_call.reset();
        step.mode = common_plan_step_mode::reasoning;
        step.required_evidence.clear();
        detail = validation_error;
        return true;
    };
    for (const auto & hit : request.memories) {
        result.memory_ids.push_back(hit.memory.id);
        result.events.push_back({common_agent_event_type::memory_retrieved, "memory supplied to agent runtime", hit.memory.id, std::nullopt});
        append_trace(result, common_runtime_trace_stage::observation, common_runtime_trace_kind::recorded,
            "memory supplied to runtime", {}, {}, {}, hit.memory.id);
    }
    common_plan_state plan;
    if (request.plan_id && !request.plan_id->empty()) {
        const auto existing = store.get(*request.plan_id, error);
        if (!error.empty()) { result.error = error; return result; }
        if (existing) {
            if (!common_plan_scope_matches(*existing, request.plan_scope, request.namespace_id, request.session_id, request.project_id, request.turn_id)) {
                result.error = "existing plan identity does not match requested scope";
                return result;
            }
            plan = *existing;
            result.events.push_back({common_agent_event_type::plan_updated, "existing plan resumed", {}, plan.id});
            append_trace(result, common_runtime_trace_stage::plan, common_runtime_trace_kind::updated,
                "existing plan resumed", plan.id);
        }
    }
    if (plan.id.empty()) {
        auto proposal = planner.create_plan_result(request, error);
        if (!error.empty()) { result.error = error; return result; }
        if (proposal.generation) {
            result.generation_records.push_back(common_agent_generation_record_from_result(
                common_agent_generation_stage::planner,
                *proposal.generation));
        }
        if (!apply_request_objective(request, proposal.plan, error)) { result.error = error; return result; }
        if (!proposal.operations.empty() && !proposal.plan.steps.empty()) {
            proposal.plan.steps.clear();
            proposal.plan.active_step_id.reset();
        }
        std::vector<std::string> initial_guardrail_events;
        for (auto & step : proposal.plan.steps) {
            std::string detail;
            if (normalize_planned_tool_step(step, false, detail)) {
                initial_guardrail_events.push_back("initial tool step degraded to reasoning: " + detail);
            }
        }
        for (auto & operation : proposal.operations) {
            if (operation.step && operation.step->intended_contribution.empty()) {
                operation.step->intended_contribution = operation.step->objective;
            }
            if (operation.step) {
                std::string detail;
                if (normalize_planned_tool_step(*operation.step, false, detail)) {
                    initial_guardrail_events.push_back("initial tool step degraded to reasoning: " + detail);
                }
            }
        }
        if (request.plan_id) proposal.plan.id = *request.plan_id;
        proposal.plan.scope = request.plan_scope;
        proposal.plan.namespace_id = request.namespace_id;
        if (proposal.plan.session_id.empty()) proposal.plan.session_id = request.session_id;
        proposal.plan.project_id = request.project_id;
        proposal.plan.turn_id = request.turn_id;
        if (!store.create(proposal.plan, error)) { result.error = error; return result; }
        plan = proposal.plan;
        result.events.push_back({common_agent_event_type::plan_created, "plan created", {}, plan.id});
        append_trace(result, common_runtime_trace_stage::plan, common_runtime_trace_kind::started,
            "plan created", plan.id);
        for (const auto & detail : initial_guardrail_events) {
            result.events.push_back({common_agent_event_type::tool_rejected, detail, {}, plan.id});
            append_trace(result, common_runtime_trace_stage::tool, common_runtime_trace_kind::failed,
                detail, plan.id);
        }
        for (auto op : proposal.operations) {
            common_plan_bind_memory_provenance(op, request.memories);
            op.plan_id = plan.id;
            op.expected_version = plan.version;
            if (!store.apply(op, plan, error)) { result.error = error; return result; }
            result.events.push_back({common_agent_event_type::plan_updated, "initial plan operation applied", {}, plan.id});
            append_trace(result, common_runtime_trace_stage::plan, common_runtime_trace_kind::updated,
                "initial plan operation applied", plan.id,
                op.step_id.value_or(op.step ? op.step->id : std::string()));
        }
    }
    result.plan_id = plan.id;

    if (request.user_correction) {
        const auto & correction = *request.user_correction;
        if (correction.source_turn_id.empty() || correction.statement.empty() || correction.statement.size() > 512) {
            result.error = "explicit user correction requires bounded source turn and statement";
            return result;
        }
        const std::string observation_id = "feedback:correction:" + std::to_string(plan.version);
        common_plan_operation observed;
        observed.kind = common_plan_operation_kind::record_observation;
        observed.plan_id = plan.id;
        observed.expected_version = plan.version;
        observed.reason_summary = "explicit user correction";
        observed.observation = common_plan_observation{observation_id, "user_correction",
            json({{"source_turn_id", correction.source_turn_id}, {"statement", correction.statement}}).dump(), 1.0f, {}, {}, 0};
        if (!store.apply(observed, plan, error)) { result.error = error; return result; }
        result.learning_signals.push_back({common_learning_signal_type::user_correction, plan.id, {}, {}, observation_id,
            "explicit user correction supplied by the caller"});
        result.events.push_back({common_agent_event_type::plan_updated, "explicit user correction recorded", {}, plan.id});
        append_trace(result, common_runtime_trace_stage::observation, common_runtime_trace_kind::recorded,
            "explicit user correction recorded", plan.id, {}, {}, observation_id);
    }

    const auto activate_next_ready_step = [&]() -> bool {
        bool has_active_step = false;
        for (const auto & step : plan.steps) if (step.status == common_plan_step_status::active) { has_active_step = true; break; }
        if (has_active_step) return true;

        const auto schedule = common_plan_schedule(plan);
        if (schedule.complete && plan.status == common_plan_status::active) {
            auto completed_candidate = plan;
            completed_candidate.status = common_plan_status::completed;
            const auto evaluation = common_plan_evaluate_goal(completed_candidate);
            if (!evaluation.evidence_sufficient) {
                result.error = "plan goal is not sufficiently evidenced";
                if (!evaluation.unmet_criteria.empty()) result.error += ": " + evaluation.unmet_criteria.front();
                return false;
            }
            common_plan_operation complete_plan;
            complete_plan.kind = common_plan_operation_kind::complete_plan;
            complete_plan.plan_id = plan.id;
            complete_plan.expected_version = plan.version;
            complete_plan.reason_summary = "all mandatory plan steps completed";
            if (!store.apply(complete_plan, plan, error)) return false;
            result.events.push_back({common_agent_event_type::plan_updated, "plan completed by scheduler", {}, plan.id});
            return false;
        }
        if (schedule.ready_step_ids.empty()) return false;

        common_plan_operation activate;
        activate.kind = common_plan_operation_kind::activate_step;
        activate.plan_id = plan.id;
        activate.expected_version = plan.version;
        activate.step_id = schedule.ready_step_ids.front();
        activate.reason_summary = "scheduler selected dependency-ready step";
        if (!store.apply(activate, plan, error)) return false;
        result.events.push_back({common_agent_event_type::plan_updated, "scheduler activated plan step", {}, plan.id});
        append_trace(result, common_runtime_trace_stage::step, common_runtime_trace_kind::started,
            "scheduler activated plan step", plan.id, *activate.step_id);
        return true;
    };

    const auto complete_active_synthesis_step = [&]() -> bool {
        if (!plan.active_step_id) return true;
        common_plan_step * active = nullptr;
        for (auto & step : plan.steps) if (step.id == *plan.active_step_id) { active = &step; break; }
        if (!active || active->status != common_plan_step_status::active || common_plan_step_effective_mode(*active) != common_plan_step_mode::final_response) return true;

        common_plan_operation complete;
        complete.kind = common_plan_operation_kind::complete_step;
        complete.plan_id = plan.id;
        complete.expected_version = plan.version;
        complete.step_id = active->id;
        complete.reason_summary = "final response synthesis completed";
        for (const auto & observation : plan.observations) {
            complete.evidence_ids.push_back(observation.id);
            complete.evidence_ids.insert(complete.evidence_ids.end(), observation.evidence_ids.begin(), observation.evidence_ids.end());
        }
        if (!store.apply(complete, plan, error)) return false;
        result.events.push_back({common_agent_event_type::plan_updated, "final synthesis step completed", {}, plan.id});
        append_trace(result, common_runtime_trace_stage::step, common_runtime_trace_kind::completed,
            "final synthesis step completed", plan.id, active->id);
        activate_next_ready_step();
        return error.empty();
    };

    std::vector<std::string> guidance;
    std::set<std::string> executed_step_ids;
    bool executed_request_tool = false;
    for (size_t iteration = 0; iteration < request.max_iterations; ++iteration) {
        size_t tool_batches = 0;
        // Execute the contiguous, dependency-ready tool chain before drafting.
        // This makes normal plan progression deterministic; reflection remains
        // reserved for repair or replanning.
        while (true) {
            std::optional<common_registered_tool_call> tool_call;
            std::string tool_step_id = "request";
            if (plan.active_step_id) for (const auto & step : plan.steps) if (step.id == *plan.active_step_id && step.status == common_plan_step_status::active && common_plan_step_effective_mode(step) == common_plan_step_mode::reasoning && !executed_step_ids.count(step.id)) {
                const auto reasoning_result = executor.generate_reasoning_result(request, plan, step, error);
                std::string reasoning = reasoning_result.content;
                if (!error.empty()) { result.error = "reasoning step failed: " + error; return result; }
                result.generation_records.push_back(common_agent_generation_record_from_result(
                    common_agent_generation_stage::reasoning,
                    reasoning_result));
                result.reasoning_decoded_tokens += reasoning_result.decoded_tokens;
                result.total_decoded_tokens += reasoning_result.decoded_tokens;
                auto parsed = json::parse(reasoning, nullptr, false);
                // Reasoning is evidence only. Small local models occasionally
                // ignore the requested JSON envelope; preserve that bounded
                // output as explicitly unstructured evidence instead of
                // failing an otherwise valid blueprint execution.
                if (parsed.is_object()) {
                    reasoning = parsed.dump();
                } else {
                    reasoning = json({{"summary", reasoning}, {"format", "unstructured"}}).dump();
                }
                if (reasoning.size() > 4096) { result.error = "reasoning step result is too large"; return result; }
                common_plan_operation observed;
                observed.kind = common_plan_operation_kind::record_observation;
                observed.plan_id = plan.id;
                observed.expected_version = plan.version;
                observed.reason_summary = "reasoning step result";
                observed.observation = common_plan_observation{"reasoning:" + step.id, "reasoning", reasoning, 1.0f, {}, {}, 0};
                if (!store.apply(observed, plan, error)) { result.error = error; return result; }
                common_plan_operation complete;
                complete.kind = common_plan_operation_kind::complete_step;
                complete.plan_id = plan.id;
                complete.expected_version = plan.version;
                complete.step_id = step.id;
                complete.reason_summary = "reasoning step completed";
                if (!store.apply(complete, plan, error)) { result.error = error; return result; }
                executed_step_ids.insert(step.id);
                result.events.push_back({common_agent_event_type::plan_updated, "reasoning observation recorded", {}, plan.id});
                append_trace(result, common_runtime_trace_stage::observation, common_runtime_trace_kind::recorded,
                    "reasoning observation recorded", plan.id, step.id, {}, "reasoning:" + step.id);
                append_trace(result, common_runtime_trace_stage::step, common_runtime_trace_kind::completed,
                    "reasoning step completed", plan.id, step.id);
                if (!activate_next_ready_step()) break;
                continue;
            }
            if (plan.active_step_id) for (const auto & step : plan.steps) if (step.id == *plan.active_step_id && step.status == common_plan_step_status::active && common_plan_step_effective_mode(step) == common_plan_step_mode::tool && !executed_step_ids.count(step.id)) {
                if (step.selected_tool && *step.selected_tool != step.tool_call->name) { result.error = "active step selected tool does not match its tool call"; return result; }
                tool_call = common_registered_tool_call{step.tool_call->name, step.tool_call->arguments_json};
                if (!common_plan_materialize_tool_arguments(plan, step, tool_call->arguments_json, tool_call->arguments_json, error)) { result.events.push_back({common_agent_event_type::tool_rejected, error, {}, plan.id}); result.error = "tool argument binding failed: " + error; return result; }
                tool_step_id = step.id;
                break;
            }
            if (!tool_call && request.tool_call && !executed_request_tool) tool_call = request.tool_call;
            if (!tool_call) {
                if (plan.active_step_id || !activate_next_ready_step()) break;
                continue;
            }
            if (tool_batches >= request.max_tool_batches) break;
            if (!tools || request.max_tool_batches == 0) { result.failures.push_back(tool_failure(tool_call->name, tool_step_id, {}, "tool.unavailable", common_agent_failure_class::execution, false, "Registered tool execution is unavailable.")); result.events.push_back({common_agent_event_type::tool_rejected, "registered tool execution is unavailable", {}, plan.id}); result.error = "registered tool execution is unavailable"; return result; }
            if (!tools->is_read_only(tool_call->name) && !(request.allow_policy_gated_tool_proposals && tools->is_policy_gated(tool_call->name))) { result.failures.push_back(tool_failure(tool_call->name, tool_step_id, {}, "tool.policy_denied", common_agent_failure_class::policy, false, "The tool is not approved by the active policy.")); result.events.push_back({common_agent_event_type::tool_rejected, "tool is not approved for this batch", {}, plan.id}); result.error = "planned tool is not approved for this batch"; return result; }
            apply_safe_tool_defaults(request, *tool_call);
            if (!tools->validate(*tool_call, error)) {
                const std::string failure_observation_id = next_tool_observation_id(plan, tool_step_id, tool_call->name);
                const auto validation_error = error;
                auto failure = tool_failure(tool_call->name, tool_step_id, failure_observation_id, "tool.invalid_arguments", common_agent_failure_class::validation, false, "Tool arguments do not satisfy the registered contract: " + validation_error);
                result.failures.push_back(failure);
                common_plan_operation observed;
                observed.kind = common_plan_operation_kind::record_observation;
                observed.plan_id = plan.id;
                observed.expected_version = plan.version;
                observed.reason_summary = "registered tool validation failure";
                observed.observation = common_plan_observation{failure_observation_id, tool_call->name, json({{"failure", render_failure(failure)}}).dump(), 0.0f, {}, {}, 0};
                if (!store.apply(observed, plan, error)) { result.error = error; return result; }
                common_plan_operation failed;
                failed.kind = common_plan_operation_kind::fail_step;
                failed.plan_id = plan.id;
                failed.expected_version = plan.version;
                failed.step_id = tool_step_id;
                failed.reason_summary = "registered tool validation failed";
                if (!store.apply(failed, plan, error)) { result.error = error; return result; }
                result.learning_signals.push_back({common_learning_signal_type::tool_failure, plan.id, tool_step_id,
                    tool_call->name, failure_observation_id, "registered tool validation failed"});
                result.events.push_back({common_agent_event_type::tool_rejected, validation_error, {}, plan.id});
                result.events.push_back({common_agent_event_type::plan_updated, "tool validation failure recorded for repair", {}, plan.id});
                append_trace(result, common_runtime_trace_stage::tool, common_runtime_trace_kind::failed,
                    validation_error, plan.id, tool_step_id, tool_call->name, failure_observation_id);
                break;
            }
            const auto execution = tools->execute(*tool_call);
            if (!execution.ok) {
                if (tool_step_id == "request") { result.events.push_back({common_agent_event_type::tool_rejected, execution.safe_summary, {}, plan.id}); result.error = "registered request tool failed: " + execution.safe_summary; return result; }
                const std::string failure_observation_id = next_tool_observation_id(plan, tool_step_id, tool_call->name);
                const auto failure = structured_tool_failure(tool_call->name, tool_step_id, failure_observation_id, execution);
                result.failures.push_back(failure);
                common_plan_operation observed;
                observed.kind = common_plan_operation_kind::record_observation;
                observed.plan_id = plan.id;
                observed.expected_version = plan.version;
                observed.reason_summary = "registered tool failure";
                observed.observation = common_plan_observation{failure_observation_id, tool_call->name, json({{"failure", render_failure(failure)}}).dump(), 0.0f, {}, execution.resource_refs, 0};
                if (!store.apply(observed, plan, error)) { result.error = error; return result; }
                common_plan_operation failed;
                failed.kind = common_plan_operation_kind::fail_step;
                failed.plan_id = plan.id;
                failed.expected_version = plan.version;
                failed.step_id = tool_step_id;
                failed.reason_summary = "registered tool failed";
                if (!store.apply(failed, plan, error)) { result.error = error; return result; }
                result.learning_signals.push_back({common_learning_signal_type::tool_failure, plan.id, tool_step_id,
                    tool_call->name, failure_observation_id, "registered tool failed"});
                result.events.push_back({common_agent_event_type::tool_rejected, execution.safe_summary, {}, plan.id});
                result.events.push_back({common_agent_event_type::plan_updated, "tool failure recorded for repair", {}, plan.id});
                append_trace(result, common_runtime_trace_stage::tool, common_runtime_trace_kind::failed,
                    execution.safe_summary, plan.id, tool_step_id, tool_call->name, failure_observation_id);
                break;
            }
            std::string tool_result = execution.output;
            if (tool_result.size() > 4096) tool_result.resize(4096);
            common_plan_operation observed;
            observed.kind = common_plan_operation_kind::record_observation;
            observed.plan_id = plan.id;
            observed.expected_version = plan.version;
            observed.reason_summary = "registered tool result";
            observed.observation = common_plan_observation{next_tool_observation_id(plan, tool_step_id, tool_call->name), tool_call->name, tool_result, 1.0f, {}, execution.resource_refs, 0};
            if (!store.apply(observed, plan, error)) { result.error = error; return result; }
            if (tool_step_id == "request") executed_request_tool = true; else {
                executed_step_ids.insert(tool_step_id);
                common_plan_operation complete;
                complete.kind = common_plan_operation_kind::complete_step;
                complete.plan_id = plan.id;
                complete.expected_version = plan.version;
                complete.step_id = tool_step_id;
                complete.reason_summary = "registered tool completed";
                if (!store.apply(complete, plan, error)) { result.error = error; return result; }
                result.events.push_back({common_agent_event_type::plan_updated, "tool step completed", {}, plan.id});
                append_trace(result, common_runtime_trace_stage::step, common_runtime_trace_kind::completed,
                    "tool step completed", plan.id, tool_step_id, tool_call->name);
                activate_next_ready_step();
                if (!error.empty()) { result.error = error; return result; }
            }
            ++tool_batches;
            result.events.push_back({common_agent_event_type::tool_executed, "registered tool result recorded", {}, plan.id});
            result.events.push_back({common_agent_event_type::plan_updated, "tool observation recorded", {}, plan.id});
            append_trace(result, common_runtime_trace_stage::tool, common_runtime_trace_kind::succeeded,
                "registered tool result recorded", plan.id, tool_step_id, tool_call->name, observed.observation->id);
            append_trace(result, common_runtime_trace_stage::observation, common_runtime_trace_kind::recorded,
                "tool observation recorded", plan.id, tool_step_id, tool_call->name, observed.observation->id);
        }

        const auto draft_result = executor.generate_draft_result(request, plan, guidance, error);
        auto draft = draft_result.content;
        if (!error.empty()) { result.error = error; return result; }
        result.generation_records.push_back(common_agent_generation_record_from_result(
            common_agent_generation_stage::draft,
            draft_result));
        result.response_decoded_tokens = draft_result.decoded_tokens;
        result.total_decoded_tokens += draft_result.decoded_tokens;
        result.response_generation_status = draft_result.status;
        result.response_stop_reason = draft_result.stop_reason;
        append_trace(result, common_runtime_trace_stage::response, common_runtime_trace_kind::recorded,
            "draft generated", plan.id);
        if (!request.enable_reflection || iteration >= request.max_reflection_rounds) {
            if (!complete_active_synthesis_step()) { result.error = error; return result; }
            result.response = draft;
            result.limit_reached = request.enable_reflection;
            append_trace(result, common_runtime_trace_stage::response, common_runtime_trace_kind::completed,
                "response accepted without reflection revision", plan.id);
            break;
        }
        auto reflection = reflector.evaluate_result(request, plan, draft, error);
        if (!error.empty()) { result.response = draft; result.error = "reflection failed safely: " + error; break; }
        if (reflection.generation) {
            result.generation_records.push_back(common_agent_generation_record_from_result(
                common_agent_generation_stage::reflection,
                *reflection.generation));
        }
        result.reflected = true;
        result.events.push_back({common_agent_event_type::reflection_completed, "reflection completed", {}, plan.id});
        append_trace(result, common_runtime_trace_stage::reflection, common_runtime_trace_kind::completed,
            "reflection completed", plan.id);
        if (reflection.learning_hint) {
            const auto & hint = *reflection.learning_hint;
            common_plan_operation observed;
            observed.kind = common_plan_operation_kind::record_observation;
            observed.plan_id = plan.id;
            observed.expected_version = plan.version;
            observed.reason_summary = "reflection learning hint";
            const std::string observation_id = "reflection:learning:" + std::to_string(plan.version) + ":" + std::to_string(iteration);
            observed.observation = common_plan_observation{observation_id, "reflection_hint",
                json({{"category", hint.category}, {"statement", hint.statement}, {"expected_reuse", hint.expected_reuse}}).dump(),
                reflection.confidence, {}, {}, 0};
            if (store.apply(observed, plan, error)) {
                result.learning_signals.push_back({common_learning_signal_type::reflection_hint, plan.id, {}, {}, observation_id,
                    "reflection supplied a bounded reusable learning hint"});
                append_trace(result, common_runtime_trace_stage::observation, common_runtime_trace_kind::recorded,
                    "reflection learning hint recorded", plan.id, {}, {}, observation_id);
            } else {
                error.clear();
            }
        }
        for (auto op : reflection.proposed_plan_operations) {
            common_plan_bind_memory_provenance(op, request.memories);
            if ((op.kind == common_plan_operation_kind::add_step || op.kind == common_plan_operation_kind::replace_step) && op.step && op.step->tool_call) {
                std::string validation_error;
                if (normalize_planned_tool_step(*op.step, true, validation_error)) {
                    result.events.push_back({common_agent_event_type::tool_rejected,
                        "reflection-added tool step degraded to reasoning: " + validation_error, {}, plan.id});
                }
            }
            op.plan_id = plan.id;
            op.expected_version = plan.version;
            if (!store.apply(op, plan, error)) { error.clear(); continue; }
            result.events.push_back({common_agent_event_type::plan_updated, "reflection plan operation applied", {}, plan.id});
            append_trace(result, common_runtime_trace_stage::plan, common_runtime_trace_kind::updated,
                "reflection plan operation applied", plan.id,
                op.step_id.value_or(op.step ? op.step->id : std::string()));
        }
        if (reflection.ready_to_answer || reflection.decision == common_reflection_decision::accept) {
            if (!complete_active_synthesis_step()) { result.error = error; return result; }
            result.response = draft;
            append_trace(result, common_runtime_trace_stage::reflection, common_runtime_trace_kind::decided,
                "reflection accepted response", plan.id);
            append_trace(result, common_runtime_trace_stage::response, common_runtime_trace_kind::completed,
                "response accepted after reflection", plan.id);
            break;
        }
        if (reflection.decision == common_reflection_decision::abort) {
            result.error = "reflection aborted answer";
            append_trace(result, common_runtime_trace_stage::reflection, common_runtime_trace_kind::failed,
                "reflection aborted answer", plan.id);
            break;
        }
        guidance = reflection.revision_guidance;
        result.revised = true;
        result.events.push_back({common_agent_event_type::response_revised, "reflection requested revision", {}, plan.id});
        append_trace(result, common_runtime_trace_stage::reflection, common_runtime_trace_kind::decided,
            "reflection requested response revision", plan.id);
    }
    if (result.response.empty() && result.error.empty()) {
        result.error = "agent loop reached its iteration limit";
        append_trace(result, common_runtime_trace_stage::turn, common_runtime_trace_kind::failed,
            result.error, plan.id);
    }
    if (result.error.empty() && !result.response.empty() && plan.status == common_plan_status::active) {
        for (size_t pass = 0; pass < 3 && plan.status == common_plan_status::active; ++pass) {
            const auto before_version = plan.version;
            if (!complete_active_synthesis_step()) { result.error = error; return result; }
            if (plan.status != common_plan_status::active) break;
            activate_next_ready_step();
            if (!error.empty()) { result.error = error; return result; }
            if (plan.version == before_version) break;
        }
    }
    result.plan_version = plan.version;
    if (result.error.empty() && !result.response.empty() && plan.status == common_plan_status::completed) {
        const auto failure = std::find_if(result.learning_signals.begin(), result.learning_signals.end(), [](const auto & signal) {
            return signal.type == common_learning_signal_type::tool_failure;
        });
        if (failure != result.learning_signals.end()) result.learning_signals.push_back({common_learning_signal_type::successful_recovery, plan.id, failure->step_id,
            failure->tool_name, failure->evidence_id, "plan completed after a recorded tool failure"});
    }
    if (memory_learner && result.error.empty() && !result.response.empty() && plan.status == common_plan_status::completed) {
        const auto learning = memory_learner->learn(request, plan, result);
        if (learning.generation) {
            result.generation_records.push_back(common_agent_generation_record_from_result(
                common_agent_generation_stage::memory_learning,
                *learning.generation));
        }
        result.learned_memory_candidate = learning.candidate;
        result.memory_learning_summary = std::string(common_memory_learning_decision_name(learning.decision)) + ": " + learning.reason;
        result.memory_learning_related_count = learning.related_count;
        if ((learning.decision == common_memory_learning_decision::accepted || learning.decision == common_memory_learning_decision::duplicate) &&
                learning.candidate && learning.candidate->kind == common_memory_kind::procedure && learning.stored_memory_id) {
            const auto promotion = memory_learner->promote_completed_procedure(request, plan, store, *learning.stored_memory_id);
            if (promotion.blueprint_id) {
                result.events.push_back({common_agent_event_type::blueprint_promoted,
                    "procedure promoted after " + std::to_string(promotion.verified_uses) + " verified uses",
                    *learning.stored_memory_id, *promotion.blueprint_id});
                append_trace(result, common_runtime_trace_stage::memory_learning, common_runtime_trace_kind::updated,
                    "procedure promoted after verified uses", plan.id, {}, {}, *learning.stored_memory_id, *promotion.blueprint_id);
            }
        }
        append_trace(result, common_runtime_trace_stage::memory_learning, common_runtime_trace_kind::summary,
            std::string(common_memory_learning_decision_name(learning.decision)) + ": " + learning.reason, plan.id);
        if (learning.decision == common_memory_learning_decision::accepted) {
            result.events.push_back({common_agent_event_type::memory_remembered, "post-turn candidate stored", learning.stored_memory_id.value_or(""), plan.id});
        } else if (learning.decision == common_memory_learning_decision::no_candidate) {
            result.events.push_back({common_agent_event_type::memory_candidate_extracted, "post-turn no candidate", {}, plan.id});
        } else {
            result.events.push_back({common_agent_event_type::memory_candidate_not_stored, "post-turn candidate not stored: " + learning.reason, {}, plan.id});
        }
    } else if (memory_learner && result.error.empty() && !result.response.empty()) {
        result.memory_learning_summary = "skipped: plan did not complete";
        result.events.push_back({common_agent_event_type::memory_candidate_not_stored, "post-turn learning skipped because the plan did not complete", {}, plan.id});
        append_trace(result, common_runtime_trace_stage::memory_learning, common_runtime_trace_kind::skipped,
            "post-turn learning skipped because the plan did not complete", plan.id);
    }
    if (!result.response.empty()) {
        append_trace(result, common_runtime_trace_stage::turn, common_runtime_trace_kind::completed,
            "agent runtime turn completed", plan.id);
    } else if (!result.error.empty()) {
        append_trace(result, common_runtime_trace_stage::turn, common_runtime_trace_kind::failed,
            result.error, plan.id);
    }
    return result;
}
