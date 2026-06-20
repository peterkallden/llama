#include "agent/agent-runtime.h"

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

common_agent_runtime::common_agent_runtime(common_plan_store & store, common_planner & planner, common_action_executor & executor, common_reflection_engine & reflector, const common_tool_registry * tools) : store(store), planner(planner), executor(executor), reflector(reflector), tools(tools) {}

common_agent_result common_agent_runtime::run(const common_agent_request & request) {
    common_agent_result result;
    std::string error;
    if (!request.enable_planning) { result.error = "planning is disabled"; return result; }
    for (const auto & hit : request.memories) {
        result.memory_ids.push_back(hit.memory.id);
        result.events.push_back({common_agent_event_type::memory_retrieved, "memory supplied to agent runtime", hit.memory.id, std::nullopt});
    }
    auto proposal = planner.create_plan(request, error);
    if (!error.empty()) { result.error = error; return result; }
    proposal.plan.scope = request.plan_scope;
    if (proposal.plan.session_id.empty()) proposal.plan.session_id = request.session_id;
    if (!store.create(proposal.plan, error)) { result.error = error; return result; }
    common_plan_state plan = proposal.plan;
    result.plan_id = plan.id;
    result.events.push_back({common_agent_event_type::plan_created, "plan created", {}, plan.id});
    for (auto op : proposal.operations) {
        op.plan_id = plan.id;
        op.expected_version = plan.version;
        if (!store.apply(op, plan, error)) { result.error = error; return result; }
        result.events.push_back({common_agent_event_type::plan_updated, "initial plan operation applied", {}, plan.id});
    }

    std::vector<std::string> guidance;
    std::set<std::string> executed_step_ids;
    bool executed_request_tool = false;
    size_t tool_batches = 0;
    for (size_t iteration = 0; iteration < request.max_iterations; ++iteration) {
        std::optional<common_registered_tool_call> tool_call;
        std::string tool_step_id = "request";
        if (tool_batches < request.max_tool_batches) {
            if (plan.active_step_id) for (const auto & step : plan.steps) if (step.id == *plan.active_step_id && step.status == common_plan_step_status::active && step.tool_call && !executed_step_ids.count(step.id)) {
                if (step.selected_tool && *step.selected_tool != step.tool_call->name) { result.error = "active step selected tool does not match its tool call"; return result; }
                tool_call = common_registered_tool_call{step.tool_call->name, step.tool_call->arguments_json};
                tool_step_id = step.id;
                break;
            }
            if (!tool_call && request.tool_call && !executed_request_tool) tool_call = request.tool_call;
        }
        if (tool_call) {
            if (!tools || request.max_tool_batches == 0) { result.events.push_back({common_agent_event_type::tool_rejected, "registered tool execution is unavailable", {}, plan.id}); result.error = "registered tool execution is unavailable"; return result; }
            if (!tools->is_read_only(tool_call->name) && !(request.allow_policy_gated_tool_proposals && tools->is_policy_gated(tool_call->name))) { result.events.push_back({common_agent_event_type::tool_rejected, "tool is not approved for this batch", {}, plan.id}); result.error = "planned tool is not approved for this batch"; return result; }
            if (tool_call->name == "calculator") {
                auto arguments = json::parse(tool_call->arguments_json, nullptr, false);
                if (arguments.is_object() && !arguments.contains("expression")) {
                    std::string expression;
                    if (infer_calculator_expression(request.prompt, expression)) {
                        arguments["expression"] = expression;
                        tool_call->arguments_json = arguments.dump();
                    }
                }
            }
            std::string tool_result;
            if (!tools->execute(*tool_call, tool_result, error)) { result.events.push_back({common_agent_event_type::tool_rejected, error, {}, plan.id}); result.error = "registered tool failed: " + error; return result; }
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
                for (const auto & step : plan.steps) {
                    if (step.status != common_plan_step_status::pending) continue;
                    bool ready = true;
                    for (const auto & dependency : step.depends_on) for (const auto & candidate : plan.steps) if (candidate.id == dependency && candidate.status != common_plan_step_status::completed) ready = false;
                    if (!ready) continue;
                    common_plan_operation activate;
                    activate.kind = common_plan_operation_kind::activate_step;
                    activate.plan_id = plan.id;
                    activate.expected_version = plan.version;
                    activate.step_id = step.id;
                    activate.reason_summary = "next dependency-ready step";
                    if (!store.apply(activate, plan, error)) { result.error = error; return result; }
                    result.events.push_back({common_agent_event_type::plan_updated, "next plan step activated", {}, plan.id});
                    break;
                }
            }
            ++tool_batches;
            result.events.push_back({common_agent_event_type::tool_executed, "registered tool result recorded", {}, plan.id});
            result.events.push_back({common_agent_event_type::plan_updated, "tool observation recorded", {}, plan.id});
        }

        auto draft = executor.generate_draft(request, plan, guidance, error);
        if (!error.empty()) { result.error = error; return result; }
        if (!request.enable_reflection || iteration >= request.max_reflection_rounds) {
            result.response = draft;
            result.limit_reached = request.enable_reflection;
            break;
        }
        auto reflection = reflector.evaluate(request, plan, draft, error);
        if (!error.empty()) { result.response = draft; result.error = "reflection failed safely: " + error; break; }
        result.reflected = true;
        result.events.push_back({common_agent_event_type::reflection_completed, "reflection completed", {}, plan.id});
        for (auto op : reflection.proposed_plan_operations) {
            op.plan_id = plan.id;
            op.expected_version = plan.version;
            if (!store.apply(op, plan, error)) { error.clear(); continue; }
            result.events.push_back({common_agent_event_type::plan_updated, "reflection plan operation applied", {}, plan.id});
        }
        if (reflection.ready_to_answer || reflection.decision == common_reflection_decision::accept) { result.response = draft; break; }
        if (reflection.decision == common_reflection_decision::abort) { result.error = "reflection aborted answer"; break; }
        guidance = reflection.revision_guidance;
        result.revised = true;
        result.events.push_back({common_agent_event_type::response_revised, "reflection requested revision", {}, plan.id});
    }
    if (result.response.empty() && result.error.empty()) result.error = "agent loop reached its iteration limit";
    result.plan_version = plan.version;
    return result;
}
