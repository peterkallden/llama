#include "plan/plan-scheduler.h"

#include <cassert>

int main() {
    common_plan_state plan;
    plan.status = common_plan_status::active;
    plan.steps = {
        {"inspect", "Inspect", "Find the affected code"},
        {"verify", "Verify", "Run the focused test"},
        {"report", "Report", "Summarize the verified result"},
    };
    plan.steps[1].depends_on = {"inspect"};
    plan.steps[2].depends_on = {"verify"};
    plan.steps[2].required_evidence = {"test-result"};

    auto schedule = common_plan_schedule(plan);
    assert((schedule.ready_step_ids == std::vector<std::string>{"inspect"}));
    assert(schedule.blocked_step_ids.size() == 2 && !schedule.terminal && !schedule.complete);

    plan.steps[0].status = common_plan_step_status::completed;
    schedule = common_plan_schedule(plan);
    assert((schedule.ready_step_ids == std::vector<std::string>{"verify"}));

    plan.steps[1].status = common_plan_step_status::completed;
    schedule = common_plan_schedule(plan);
    assert(schedule.ready_step_ids.empty() && (schedule.blocked_step_ids == std::vector<std::string>{"report"}) && schedule.blocked && !schedule.terminal && schedule.state == common_plan_schedule_state::blocked);

    plan.observations.push_back({"tool:verify", "run_test", "passed", 1.0f, {"test-result"}, {}, 0});
    schedule = common_plan_schedule(plan);
    assert((schedule.ready_step_ids == std::vector<std::string>{"report"}));

    plan.steps[2].status = common_plan_step_status::completed;
    schedule = common_plan_schedule(plan);
    assert(schedule.complete && schedule.terminal);
    return 0;
}
