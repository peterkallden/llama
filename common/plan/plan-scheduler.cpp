#include "plan/plan-scheduler.h"

#include <algorithm>

namespace {

const common_plan_step * find_step(const common_plan_state & plan, const std::string & id) {
    for (const auto & step : plan.steps) if (step.id == id) return &step;
    return nullptr;
}

bool has_evidence(const common_plan_state & plan, const std::string & id) {
    for (const auto & observation : plan.observations) {
        if (observation.id == id || std::find(observation.evidence_ids.begin(), observation.evidence_ids.end(), id) != observation.evidence_ids.end()) return true;
    }
    return false;
}

} // namespace

common_plan_schedule_result common_plan_schedule(const common_plan_state & plan) {
    common_plan_schedule_result result;
    if (plan.status == common_plan_status::failed || plan.status == common_plan_status::cancelled) {
        result.terminal = true;
        return result;
    }
    bool has_active = false;
    bool has_incomplete_mandatory = false;

    for (const auto & step : plan.steps) {
        if (step.status == common_plan_step_status::active) has_active = true;
        if (!step.optional && step.status != common_plan_step_status::completed && step.status != common_plan_step_status::skipped) has_incomplete_mandatory = true;
        if (step.status != common_plan_step_status::pending) continue;

        bool ready = true;
        for (const auto & dependency : step.depends_on) {
            const auto * prerequisite = find_step(plan, dependency);
            if (!prerequisite || prerequisite->status != common_plan_step_status::completed) { ready = false; break; }
        }
        if (ready) for (const auto & evidence : step.required_evidence) {
            if (!has_evidence(plan, evidence)) { ready = false; break; }
        }
        (ready ? result.ready_step_ids : result.blocked_step_ids).push_back(step.id);
    }

    result.complete = !has_incomplete_mandatory;
    if (result.complete) {
        result.state = common_plan_schedule_state::complete;
        result.terminal = true;
    } else if (has_active || !result.ready_step_ids.empty()) {
        result.state = common_plan_schedule_state::runnable;
    } else if (plan.status == common_plan_status::active || plan.status == common_plan_status::blocked) {
        result.state = common_plan_schedule_state::blocked;
        result.blocked = true;
    }
    return result;
}
