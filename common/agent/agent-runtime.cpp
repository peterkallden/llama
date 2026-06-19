#include "agent/agent-runtime.h"

common_agent_runtime::common_agent_runtime(common_plan_store & store, common_planner & planner, common_action_executor & executor, common_reflection_engine & reflector) : store(store), planner(planner), executor(executor), reflector(reflector) {}
common_agent_result common_agent_runtime::run(const common_agent_request & request) {
    common_agent_result result; std::string error; auto proposal = planner.create_plan(request, error); if (!error.empty()) { result.error = error; return result; }
    if (!store.create(proposal.plan, error)) { result.error = error; return result; } common_plan_state plan = proposal.plan;
    for (auto op : proposal.operations) { op.plan_id = plan.id; op.expected_version = plan.version; if (!store.apply(op, plan, error)) { result.error = error; return result; } }
    std::vector<std::string> guidance;
    for (size_t iteration = 0; iteration < request.max_iterations; ++iteration) {
        auto draft = executor.generate_draft(request, plan, guidance, error); if (!error.empty()) { result.error = error; return result; }
        if (iteration >= request.max_reflection_rounds) { result.answer = draft; result.limit_reached = true; break; }
        auto reflection = reflector.evaluate(request, plan, draft, error); if (!error.empty()) { result.answer = draft; result.error = "reflection failed safely: " + error; break; }
        for (auto op : reflection.proposed_plan_operations) { op.plan_id = plan.id; op.expected_version = plan.version; if (!store.apply(op, plan, error)) { error.clear(); continue; } }
        if (reflection.ready_to_answer || reflection.decision == common_reflection_decision::accept) { result.answer = draft; break; }
        if (reflection.decision == common_reflection_decision::abort) { result.error = "reflection aborted answer"; break; }
        guidance = reflection.revision_guidance; result.revised = true;
    }
    if (result.answer.empty() && result.error.empty()) result.error = "agent loop reached its iteration limit";
    result.plan_id = plan.id; result.plan_version = plan.version; return result;
}
