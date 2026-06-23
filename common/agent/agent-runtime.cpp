#include "agent/agent-runtime.h"
#include "agent/memory-learning.h"
#include "plan/plan-bindings.h"
#include "plan/plan-memory.h"
#include "plan/plan-scheduler.h"

#include <nlohmann/json.hpp>
#include <regex>
#include <set>

using json = nlohmann::ordered_json;

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
    } else if (call.name == "repository_list") {
        if (!arguments.contains("path")) { arguments["path"] = ""; changed = true; }
        if (!arguments.contains("depth")) { arguments["depth"] = 1; changed = true; }
    }
    if (changed) call.arguments_json = arguments.dump();
}

common_agent_runtime::common_agent_runtime(common_plan_store & store, common_planner & planner, common_action_executor & executor, common_reflection_engine & reflector, const common_tool_registry * tools, common_memory_post_turn_learner * memory_learner) : store(store), planner(planner), executor(executor), reflector(reflector), tools(tools), memory_learner(memory_learner) {}

common_agent_result common_agent_runtime::run(const common_agent_request & request) {
    common_agent_result result;
    std::string error;
    if (!request.enable_planning) { result.error = "planning is disabled"; return result; }
    for (const auto & hit : request.memories) {
        result.memory_ids.push_back(hit.memory.id);
        result.events.push_back({common_agent_event_type::memory_retrieved, "memory supplied to agent runtime", hit.memory.id, std::nullopt});
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
        }
    }
    if (plan.id.empty()) {
        auto proposal = planner.create_plan(request, error);
        if (!error.empty()) { result.error = error; return result; }
        if (request.plan_id) proposal.plan.id = *request.plan_id;
        proposal.plan.scope = request.plan_scope;
        proposal.plan.namespace_id = request.namespace_id;
        if (proposal.plan.session_id.empty()) proposal.plan.session_id = request.session_id;
        proposal.plan.project_id = request.project_id;
        proposal.plan.turn_id = request.turn_id;
        if (!store.create(proposal.plan, error)) { result.error = error; return result; }
        plan = proposal.plan;
        result.events.push_back({common_agent_event_type::plan_created, "plan created", {}, plan.id});
        for (auto op : proposal.operations) {
            common_plan_bind_memory_provenance(op, request.memories);
            op.plan_id = plan.id;
            op.expected_version = plan.version;
            if (!store.apply(op, plan, error)) { result.error = error; return result; }
            result.events.push_back({common_agent_event_type::plan_updated, "initial plan operation applied", {}, plan.id});
        }
    }
    result.plan_id = plan.id;

    const auto activate_next_ready_step = [&]() -> bool {
        bool has_active_step = false;
        for (const auto & step : plan.steps) if (step.status == common_plan_step_status::active) { has_active_step = true; break; }
        if (has_active_step) return true;

        const auto schedule = common_plan_schedule(plan);
        if (schedule.complete && plan.status == common_plan_status::active) {
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
        activate_next_ready_step();
        return error.empty();
    };

    std::vector<std::string> guidance;
    std::set<std::string> executed_step_ids;
    bool executed_request_tool = false;
    size_t tool_batches = 0;
    for (size_t iteration = 0; iteration < request.max_iterations; ++iteration) {
        // Execute the contiguous, dependency-ready tool chain before drafting.
        // This makes normal plan progression deterministic; reflection remains
        // reserved for repair or replanning.
        while (true) {
            std::optional<common_registered_tool_call> tool_call;
            std::string tool_step_id = "request";
            if (plan.active_step_id) for (const auto & step : plan.steps) if (step.id == *plan.active_step_id && step.status == common_plan_step_status::active && common_plan_step_effective_mode(step) == common_plan_step_mode::reasoning && !executed_step_ids.count(step.id)) {
                std::string reasoning = executor.generate_reasoning(request, plan, step, error);
                if (!error.empty()) { result.error = "reasoning step failed: " + error; return result; }
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
                observed.observation = common_plan_observation{"reasoning:" + step.id, "reasoning", reasoning, 1.0f, {}, 0};
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
            if (!tools || request.max_tool_batches == 0) { result.events.push_back({common_agent_event_type::tool_rejected, "registered tool execution is unavailable", {}, plan.id}); result.error = "registered tool execution is unavailable"; return result; }
            if (!tools->is_read_only(tool_call->name) && !(request.allow_policy_gated_tool_proposals && tools->is_policy_gated(tool_call->name))) { result.events.push_back({common_agent_event_type::tool_rejected, "tool is not approved for this batch", {}, plan.id}); result.error = "planned tool is not approved for this batch"; return result; }
            apply_safe_tool_defaults(request, *tool_call);
            if (!tools->validate(*tool_call, error)) { result.events.push_back({common_agent_event_type::tool_rejected, error, {}, plan.id}); result.error = "invalid registered tool contract: " + error; return result; }
            std::string tool_result;
            if (!tools->execute(*tool_call, tool_result, error)) {
                const std::string tool_error = error;
                error.clear();
                if (tool_step_id == "request") { result.events.push_back({common_agent_event_type::tool_rejected, tool_error, {}, plan.id}); result.error = "registered request tool failed: " + tool_error; return result; }
                const std::string failure_observation_id = "tool:" + tool_step_id + ":" + tool_call->name;
                common_plan_operation observed;
                observed.kind = common_plan_operation_kind::record_observation;
                observed.plan_id = plan.id;
                observed.expected_version = plan.version;
                observed.reason_summary = "registered tool failure";
                observed.observation = common_plan_observation{failure_observation_id, tool_call->name, json({{"error", tool_error}, {"tool", tool_call->name}}).dump(), 0.0f, {}, 0};
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
                result.events.push_back({common_agent_event_type::tool_rejected, tool_error, {}, plan.id});
                result.events.push_back({common_agent_event_type::plan_updated, "tool failure recorded for repair", {}, plan.id});
                break;
            }
            if (tool_result.size() > 4096) tool_result.resize(4096);
            common_plan_operation observed;
            observed.kind = common_plan_operation_kind::record_observation;
            observed.plan_id = plan.id;
            observed.expected_version = plan.version;
            observed.reason_summary = "registered tool result";
            observed.observation = common_plan_observation{"tool:" + tool_step_id + ":" + tool_call->name, tool_call->name, tool_result, 1.0f, {}, 0};
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
                activate_next_ready_step();
                if (!error.empty()) { result.error = error; return result; }
            }
            ++tool_batches;
            result.events.push_back({common_agent_event_type::tool_executed, "registered tool result recorded", {}, plan.id});
            result.events.push_back({common_agent_event_type::plan_updated, "tool observation recorded", {}, plan.id});
        }

        auto draft = executor.generate_draft(request, plan, guidance, error);
        if (!error.empty()) { result.error = error; return result; }
        if (!request.enable_reflection || iteration >= request.max_reflection_rounds) {
            if (!complete_active_synthesis_step()) { result.error = error; return result; }
            result.response = draft;
            result.limit_reached = request.enable_reflection;
            break;
        }
        auto reflection = reflector.evaluate(request, plan, draft, error);
        if (!error.empty()) { result.response = draft; result.error = "reflection failed safely: " + error; break; }
        result.reflected = true;
        result.events.push_back({common_agent_event_type::reflection_completed, "reflection completed", {}, plan.id});
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
                reflection.confidence, {}, 0};
            if (store.apply(observed, plan, error)) {
                result.learning_signals.push_back({common_learning_signal_type::reflection_hint, plan.id, {}, {}, observation_id,
                    "reflection supplied a bounded reusable learning hint"});
            } else {
                error.clear();
            }
        }
        for (auto op : reflection.proposed_plan_operations) {
            common_plan_bind_memory_provenance(op, request.memories);
            op.plan_id = plan.id;
            op.expected_version = plan.version;
            if (!store.apply(op, plan, error)) { error.clear(); continue; }
            result.events.push_back({common_agent_event_type::plan_updated, "reflection plan operation applied", {}, plan.id});
        }
        if (reflection.ready_to_answer || reflection.decision == common_reflection_decision::accept) {
            if (!complete_active_synthesis_step()) { result.error = error; return result; }
            result.response = draft;
            break;
        }
        if (reflection.decision == common_reflection_decision::abort) { result.error = "reflection aborted answer"; break; }
        guidance = reflection.revision_guidance;
        result.revised = true;
        result.events.push_back({common_agent_event_type::response_revised, "reflection requested revision", {}, plan.id});
    }
    if (result.response.empty() && result.error.empty()) result.error = "agent loop reached its iteration limit";
    result.plan_version = plan.version;
    if (result.error.empty() && !result.response.empty() && plan.status == common_plan_status::completed) {
        for (const auto & signal : result.learning_signals) if (signal.type == common_learning_signal_type::tool_failure) {
            result.learning_signals.push_back({common_learning_signal_type::successful_recovery, plan.id, signal.step_id,
                signal.tool_name, signal.evidence_id, "plan completed after a recorded tool failure"});
            break;
        }
    }
    if (memory_learner && result.error.empty() && !result.response.empty()) {
        const auto learning = memory_learner->learn(request, plan, result);
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
            }
        }
        if (learning.decision == common_memory_learning_decision::accepted) {
            result.events.push_back({common_agent_event_type::memory_remembered, "post-turn candidate stored", learning.stored_memory_id.value_or(""), plan.id});
        } else if (learning.decision == common_memory_learning_decision::no_candidate) {
            result.events.push_back({common_agent_event_type::memory_candidate_extracted, "post-turn no candidate", {}, plan.id});
        } else {
            result.events.push_back({common_agent_event_type::memory_candidate_not_stored, "post-turn candidate not stored: " + learning.reason, {}, plan.id});
        }
    }
    return result;
}
