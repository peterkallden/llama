#include "tools/agent/runtime/agent-plan-orchestration.h"

#include <cassert>

int main() {
    common_plan_state plan;
    plan.purpose = "Analyze the dataset source and inspect relevant rows";
    plan.goal = plan.purpose;
    plan.steps.push_back({"schema", "Read dataset schema", "Inspect the dataset schema"});

    assert(!common_agent_plan_candidate_relevant_to_request("hi", plan));
    assert(!common_agent_plan_candidate_relevant_to_request("hello, thanks", plan));
    assert(common_agent_plan_candidate_relevant_to_request("show the dataset schema", plan));
    assert(common_agent_plan_candidate_relevant_to_request("continue", plan));
    assert(common_agent_plan_candidate_relevant_to_request("fortsätt med analysen", plan));
    return 0;
}
