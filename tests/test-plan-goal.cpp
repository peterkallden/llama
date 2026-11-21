#include "plan/plan-goal.h"

#include <cassert>

int main() {
    common_plan_state plan;
    plan.purpose = "Answer the user's question";
    plan.goal = "Return the verified result";
    plan.success_criteria = "The answer is grounded in the lookup result.";
    plan.status = common_plan_status::completed;

    common_plan_step lookup{"lookup", "Lookup", "Find the result"};
    lookup.mode = common_plan_step_mode::reasoning;
    lookup.intended_contribution = "Provide the evidence used by the answer.";
    lookup.status = common_plan_step_status::completed;
    plan.steps.push_back(lookup);
    plan.observations.push_back({"tool:lookup:repository.search", "repository.search", "result found", 1.0f, {}, {}, 0});

    const auto complete = common_plan_evaluate_goal(plan);
    assert(complete.structurally_complete);
    assert(complete.evidence_sufficient);
    assert(complete.confidence == 1.0f);

    plan.observations.clear();
    const auto missing = common_plan_evaluate_goal(plan);
    assert(!missing.evidence_sufficient);
    assert(!missing.unmet_criteria.empty());

    common_plan_state report_plan;
    report_plan.purpose = "Compose a grounded report";
    report_plan.goal = "Persist the report";
    report_plan.success_criteria = "Every section is validated and gaps remain explicit.";
    report_plan.status = common_plan_status::completed;
    common_plan_step section{"findings", "Findings", "Compose the findings section"};
    section.mode = common_plan_step_mode::artifact_compose;
    section.status = common_plan_step_status::completed;
    report_plan.steps.push_back(section);
    // A host-validated gap section proves the composition step completed; it
    // does not assert that the report contains source-supported findings.
    report_plan.observations.push_back({"artifact-section:findings", "artifact_section",
        R"({"coverage":"incomplete"})", 1.0f, {}, {}, 0});
    const auto report = common_plan_evaluate_goal(report_plan);
    assert(report.evidence_sufficient);
    return 0;
}
