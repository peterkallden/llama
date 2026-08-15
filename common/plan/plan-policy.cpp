#include "plan/plan-policy.h"

#include <algorithm>
#include <functional>
#include <map>
#include <set>

const char * common_plan_operation_kind_name(common_plan_operation_kind kind) {
    switch (kind) {
        case common_plan_operation_kind::create_plan: return "create_plan"; case common_plan_operation_kind::revise_goal: return "revise_goal";
        case common_plan_operation_kind::add_step: return "add_step"; case common_plan_operation_kind::revise_step: return "revise_step";
        case common_plan_operation_kind::replace_step: return "replace_step"; case common_plan_operation_kind::remove_step: return "remove_step";
        case common_plan_operation_kind::activate_step: return "activate_step"; case common_plan_operation_kind::reset_step: return "reset_step";
        case common_plan_operation_kind::complete_step: return "complete_step"; case common_plan_operation_kind::block_step: return "block_step";
        case common_plan_operation_kind::unblock_step: return "unblock_step"; case common_plan_operation_kind::fail_step: return "fail_step";
        case common_plan_operation_kind::skip_step: return "skip_step"; case common_plan_operation_kind::add_dependency: return "add_dependency";
        case common_plan_operation_kind::remove_dependency: return "remove_dependency"; case common_plan_operation_kind::add_constraint: return "add_constraint";
        case common_plan_operation_kind::add_assumption: return "add_assumption"; case common_plan_operation_kind::invalidate_assumption: return "invalidate_assumption";
        case common_plan_operation_kind::record_observation: return "record_observation"; case common_plan_operation_kind::set_next_action: return "set_next_action";
        case common_plan_operation_kind::request_replan: return "request_replan"; case common_plan_operation_kind::complete_plan: return "complete_plan";
        case common_plan_operation_kind::fail_plan: return "fail_plan";
    } return "unknown";
}

common_plan_policy::common_plan_policy(common_plan_policy_config config) : config(config) {}

static const common_plan_step * find_step(const common_plan_state & plan, const std::string & id) {
    for (const auto & step : plan.steps) if (step.id == id) return &step;
    return nullptr;
}
static bool transition(common_plan_step_status from, common_plan_step_status to) {
    return (from == common_plan_step_status::pending && (to == common_plan_step_status::active || to == common_plan_step_status::skipped)) ||
           (from == common_plan_step_status::active && (to == common_plan_step_status::completed || to == common_plan_step_status::blocked || to == common_plan_step_status::failed)) ||
           (from == common_plan_step_status::blocked && (to == common_plan_step_status::active || to == common_plan_step_status::failed)) ||
           (from == common_plan_step_status::failed && to == common_plan_step_status::active);
}
static bool reset_transition(common_plan_step_status from) {
    return from == common_plan_step_status::active || from == common_plan_step_status::blocked ||
           from == common_plan_step_status::failed || from == common_plan_step_status::skipped;
}
static bool plan_transition(common_plan_status from, common_plan_status to) {
    return (from == common_plan_status::proposed && to == common_plan_status::active) ||
           (from == common_plan_status::active && (to == common_plan_status::completed || to == common_plan_status::blocked || to == common_plan_status::failed)) ||
           (from == common_plan_status::blocked && (to == common_plan_status::active || to == common_plan_status::failed));
}
static bool has_cycle(const common_plan_state & plan, const common_plan_operation & op) {
    std::map<std::string, std::vector<std::string>> graph;
    for (const auto & step : plan.steps) graph[step.id] = step.depends_on;
    if (op.kind == common_plan_operation_kind::add_dependency && op.step_id && op.target_id) graph[*op.step_id].push_back(*op.target_id);
    std::set<std::string> visiting, visited;
    std::function<bool(const std::string &)> visit = [&](const std::string & id) { if (visiting.count(id)) return true; if (visited.count(id)) return false; visiting.insert(id); for (const auto & d : graph[id]) if (visit(d)) return true; visiting.erase(id); visited.insert(id); return false; };
    for (const auto & item : graph) if (visit(item.first)) return true;
    return false;
}

common_plan_policy_result common_plan_policy::validate(const common_plan_state & plan, const common_plan_operation & op) const {
    auto deny = [](const std::string & reason) { return common_plan_policy_result{false, reason}; };
    if (op.plan_id.empty() || op.plan_id != plan.id) return deny("operation plan id does not match plan");
    if (op.reason_summary.size() > config.max_string_length) return deny("reason summary is too long");
    if (op.expected_version != plan.version) return deny("stale plan version");
    if (op.kind == common_plan_operation_kind::add_step) {
        if (!op.step || op.step->id.empty() || op.step->title.empty()) return deny("add_step requires an identified step with a title");
        if (op.step->objective.size() > config.max_string_length || op.step->intended_contribution.size() > config.max_string_length) return deny("step objective or contribution is too long");
        const auto mode = common_plan_step_effective_mode(*op.step);
        if ((mode == common_plan_step_mode::tool) != op.step->tool_call.has_value()) return deny("step mode does not match tool binding");
        if (op.step->generated_from_memory && op.step->source_memory_ids.empty()) return deny("memory-generated step requires source memory id");
        if (!op.step->generated_from_memory && !op.step->source_memory_ids.empty()) return deny("source memory ids require memory-generated step");
        if (plan.steps.size() >= config.max_steps) return deny("plan step limit reached");
        if (find_step(plan, op.step->id)) return deny("step already exists");
        for (const auto & dep : op.step->depends_on) if (!find_step(plan, dep)) return deny("step dependency does not exist");
    }
    if (op.kind == common_plan_operation_kind::revise_step) {
        if (!op.step) return deny("revise_step requires a replacement step");
        const auto * prior = find_step(plan, op.step->id);
        if (!prior) return deny("revise_step references an unknown step");
        if (prior->title != op.step->title || prior->objective != op.step->objective || prior->intended_contribution != op.step->intended_contribution || prior->depends_on != op.step->depends_on ||
                prior->required_evidence != op.step->required_evidence || prior->source_memory_ids != op.step->source_memory_ids ||
                prior->status != op.step->status || prior->optional != op.step->optional || prior->generated_from_memory != op.step->generated_from_memory ||
                prior->blocked_by != op.step->blocked_by || prior->result_summary != op.step->result_summary) return deny("revise_step may only bind a tool to an unchanged step");
        if (!op.step->tool_call || !op.step->selected_tool || *op.step->selected_tool != op.step->tool_call->name || common_plan_step_effective_mode(*op.step) != common_plan_step_mode::tool) return deny("revise_step requires a consistent tool binding");
    }
    if (op.kind == common_plan_operation_kind::replace_step) {
        if (!op.step_id || !op.step) return deny("replace_step requires step_id and step");
        const auto * prior = find_step(plan, *op.step_id);
        if (!prior) return deny("replace_step references an unknown step");
        if (prior->status == common_plan_step_status::completed) return deny("replace_step cannot replace a completed step");
        if (op.step->id != *op.step_id) return deny("replace_step step id must match target step id");
        if (op.step->title.empty()) return deny("replace_step requires a step title");
        if (op.step->objective.size() > config.max_string_length || op.step->intended_contribution.size() > config.max_string_length) return deny("step objective or contribution is too long");
        const auto mode = common_plan_step_effective_mode(*op.step);
        if ((mode == common_plan_step_mode::tool) != op.step->tool_call.has_value()) return deny("step mode does not match tool binding");
        if (op.step->generated_from_memory && op.step->source_memory_ids.empty()) return deny("memory-generated step requires source memory id");
        if (!op.step->generated_from_memory && !op.step->source_memory_ids.empty()) return deny("source memory ids require memory-generated step");
        for (const auto & dep : op.step->depends_on) {
            if (dep == *op.step_id) return deny("step dependency cycle rejected");
            if (!find_step(plan, dep)) return deny("step dependency does not exist");
        }
        (void) prior;
    }
    if (op.kind == common_plan_operation_kind::add_dependency) {
        if (!op.step_id || !op.target_id || !find_step(plan, *op.step_id) || !find_step(plan, *op.target_id)) return deny("dependency references an unknown step");
        if (*op.step_id == *op.target_id || has_cycle(plan, op)) return deny("dependency cycle rejected");
    }
    if (op.kind == common_plan_operation_kind::activate_step || op.kind == common_plan_operation_kind::reset_step || op.kind == common_plan_operation_kind::complete_step || op.kind == common_plan_operation_kind::block_step || op.kind == common_plan_operation_kind::unblock_step || op.kind == common_plan_operation_kind::fail_step || op.kind == common_plan_operation_kind::skip_step) {
        if (!op.step_id) return deny("step operation requires step id"); const auto * step = find_step(plan, *op.step_id); if (!step) return deny("unknown step");
        common_plan_step_status target = common_plan_step_status::pending;
        if (op.kind == common_plan_operation_kind::activate_step || op.kind == common_plan_operation_kind::unblock_step) target = common_plan_step_status::active;
        if (op.kind == common_plan_operation_kind::reset_step) target = common_plan_step_status::pending;
        if (op.kind == common_plan_operation_kind::complete_step) target = common_plan_step_status::completed;
        if (op.kind == common_plan_operation_kind::block_step) target = common_plan_step_status::blocked;
        if (op.kind == common_plan_operation_kind::fail_step) target = common_plan_step_status::failed;
        if (op.kind == common_plan_operation_kind::skip_step) target = common_plan_step_status::skipped;
        if (op.kind == common_plan_operation_kind::reset_step) {
            if (!reset_transition(step->status)) return deny("illegal step reset");
        } else if (!transition(step->status, target)) return deny("illegal step state transition");
        if (target == common_plan_step_status::active) for (const auto & dep : step->depends_on) { const auto * d = find_step(plan, dep); if (!d || d->status != common_plan_step_status::completed) return deny("step has unmet dependency"); }
        if (target == common_plan_step_status::completed && config.require_evidence_for_completion) {
            for (const auto & id : step->required_evidence) {
                if (std::find(op.evidence_ids.begin(), op.evidence_ids.end(), id) == op.evidence_ids.end()) return deny("required evidence is missing");
            }
            if (step->tool_call) {
                bool observed = false;
                for (const auto & evidence_id : op.evidence_ids) {
                    for (const auto & observation : plan.observations) {
                        if (observation.id == evidence_id && observation.source == step->tool_call->name) {
                            observed = true;
                            break;
                        }
                    }
                    if (observed) break;
                }
                if (!observed) return deny("tool step requires a successful tool observation");
            }
        }
    }
    if (op.kind == common_plan_operation_kind::complete_plan) { if (!plan_transition(plan.status, common_plan_status::completed)) return deny("illegal plan state transition"); for (const auto & step : plan.steps) if (!step.optional && step.status != common_plan_step_status::completed && step.status != common_plan_step_status::skipped) return deny("mandatory steps remain incomplete"); }
    if (op.kind == common_plan_operation_kind::fail_plan && !plan_transition(plan.status, common_plan_status::failed)) return deny("illegal plan state transition");
    if (op.kind == common_plan_operation_kind::record_observation && (!op.observation || plan.observations.size() >= config.max_observations)) return deny("invalid or excessive observation");
    return {true, {}};
}
