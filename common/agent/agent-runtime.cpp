#include "agent/agent-runtime.h"

common_agent_runtime::common_agent_runtime(common_plan_store & store, common_planner & planner, common_action_executor & executor, common_reflection_engine & reflector, const common_tool_registry * tools) : store(store), planner(planner), executor(executor), reflector(reflector), tools(tools) {}
common_agent_result common_agent_runtime::run(const common_agent_request & request) {
    common_agent_result result; std::string error;
    if (!request.enable_planning) { result.error = "planning is disabled"; return result; }
    for (const auto & hit : request.memories) {
        result.memory_ids.push_back(hit.memory.id);
        result.events.push_back({common_agent_event_type::memory_retrieved, "memory supplied to agent runtime", hit.memory.id, std::nullopt});
    }
    auto proposal = planner.create_plan(request, error); if (!error.empty()) { result.error = error; return result; }
    proposal.plan.scope = request.plan_scope;
    if (proposal.plan.session_id.empty()) proposal.plan.session_id = request.session_id;
    if (!store.create(proposal.plan, error)) { result.error = error; return result; } common_plan_state plan = proposal.plan;
    result.plan_id = plan.id;
    result.events.push_back({common_agent_event_type::plan_created, "plan created", {}, plan.id});
    for (auto op : proposal.operations) { op.plan_id = plan.id; op.expected_version = plan.version; if (!store.apply(op, plan, error)) { result.error = error; return result; } result.events.push_back({common_agent_event_type::plan_updated, "initial plan operation applied", {}, plan.id}); }
    if (request.tool_call) {
        if (!tools || request.max_tool_batches == 0) { result.error = "registered tool execution is unavailable"; return result; }
        std::string tool_result; if (!tools->execute(*request.tool_call, tool_result, error)) { result.error = "registered tool failed: " + error; return result; }
        common_plan_operation observed; observed.kind = common_plan_operation_kind::record_observation; observed.plan_id = plan.id; observed.expected_version = plan.version; observed.reason_summary = "registered tool result"; observed.observation = common_plan_observation{"tool:" + request.tool_call->name, request.tool_call->name, tool_result, 1.0f, {}, 0};
        if (!store.apply(observed, plan, error)) { result.error = error; return result; }
    }
    std::vector<std::string> guidance;
    for (size_t iteration = 0; iteration < request.max_iterations; ++iteration) {
        auto draft = executor.generate_draft(request, plan, guidance, error); if (!error.empty()) { result.error = error; return result; }
        if (!request.enable_reflection || iteration >= request.max_reflection_rounds) { result.response = draft; result.limit_reached = request.enable_reflection; break; }
        auto reflection = reflector.evaluate(request, plan, draft, error); if (!error.empty()) { result.response = draft; result.error = "reflection failed safely: " + error; break; }
        result.reflected = true;
        result.events.push_back({common_agent_event_type::reflection_completed, "reflection completed", {}, plan.id});
        for (auto op : reflection.proposed_plan_operations) { op.plan_id = plan.id; op.expected_version = plan.version; if (!store.apply(op, plan, error)) { error.clear(); continue; } result.events.push_back({common_agent_event_type::plan_updated, "reflection plan operation applied", {}, plan.id}); }
        if (reflection.ready_to_answer || reflection.decision == common_reflection_decision::accept) { result.response = draft; break; }
        if (reflection.decision == common_reflection_decision::abort) { result.error = "reflection aborted answer"; break; }
        guidance = reflection.revision_guidance; result.revised = true; result.events.push_back({common_agent_event_type::response_revised, "reflection requested revision", {}, plan.id});
    }
    if (result.response.empty() && result.error.empty()) result.error = "agent loop reached its iteration limit";
    result.plan_version = plan.version; return result;
}
