#include "plan/plan-in-memory.h"
#include <algorithm>
#include <utility>

common_plan_in_memory_store::common_plan_in_memory_store(common_plan_policy policy) : policy(std::move(policy)) {}
bool common_plan_in_memory_store::open(const std::string &, std::string & error) { opened = true; error.clear(); return true; }
void common_plan_in_memory_store::close() { opened = false; plans.clear(); events.clear(); }
bool common_plan_in_memory_store::create(const common_plan_state & plan, std::string & error) { if (!opened) { error = "plan store is not open"; return false; } if (plan.id.empty() || plans.count(plan.id)) { error = "plan id is empty or already exists"; return false; } plans.emplace(plan.id, plan); error.clear(); return true; }
bool common_plan_in_memory_store::restore_history(const std::string & plan_id, std::vector<common_plan_event> history, std::string & error) { if (!opened || !plans.count(plan_id)) { error = "plan store is not open or plan is unavailable"; return false; } events[plan_id] = std::move(history); error.clear(); return true; }
std::optional<common_plan_state> common_plan_in_memory_store::get(const std::string & id, std::string & error) { if (!opened) { error = "plan store is not open"; return std::nullopt; } error.clear(); auto it = plans.find(id); return it == plans.end() ? std::nullopt : std::optional<common_plan_state>(it->second); }
std::vector<common_plan_state> common_plan_in_memory_store::list(std::string & error) { std::vector<common_plan_state> result; if (!opened) { error = "plan store is not open"; return result; } for (const auto & entry : plans) result.push_back(entry.second); error.clear(); return result; }
std::vector<common_plan_event> common_plan_in_memory_store::history(const std::string & id, std::string & error) { if (!opened) { error = "plan store is not open"; return {}; } error.clear(); return events[id]; }
bool common_plan_in_memory_store::erase(const std::string & id, std::string & error) { if (!opened) { error = "plan store is not open"; return false; } plans.erase(id); events.erase(id); error.clear(); return true; }
bool common_plan_in_memory_store::apply(const common_plan_operation & op, common_plan_state & updated, std::string & error) {
    if (!opened) { error = "plan store is not open"; return false; } auto it = plans.find(op.plan_id); if (it == plans.end()) { error = "plan not found"; return false; }
    common_plan_state next = it->second; auto result = policy.validate(next, op); if (!result.allowed) { events[op.plan_id].push_back({(uint64_t) events[op.plan_id].size() + 1, next.version, next.version, op, false, result.reason, 0}); error = result.reason; return false; }
    auto find = [&](const std::string & id) -> common_plan_step * { for (auto & step : next.steps) if (step.id == id) return &step; return nullptr; };
    switch (op.kind) {
        case common_plan_operation_kind::revise_goal: next.goal = op.value.value_or(""); break;
        case common_plan_operation_kind::add_step: next.steps.push_back(*op.step); break;
        case common_plan_operation_kind::revise_step: *find(op.step->id) = *op.step; break;
        case common_plan_operation_kind::replace_step: {
            auto replacement = *op.step;
            replacement.status = common_plan_step_status::pending;
            replacement.result_summary.reset();
            replacement.blocked_by.clear();
            *find(*op.step_id) = std::move(replacement);
            if (next.active_step_id == op.step_id) next.active_step_id.reset();
            next.status = common_plan_status::active;
            break;
        }
        case common_plan_operation_kind::remove_step: next.steps.erase(std::remove_if(next.steps.begin(), next.steps.end(), [&](const auto & s) { return s.id == *op.step_id; }), next.steps.end()); break;
        case common_plan_operation_kind::add_dependency: find(*op.step_id)->depends_on.push_back(*op.target_id); break;
        case common_plan_operation_kind::remove_dependency: { auto & v = find(*op.step_id)->depends_on; v.erase(std::remove(v.begin(), v.end(), *op.target_id), v.end()); break; }
        case common_plan_operation_kind::add_constraint: next.constraints.push_back(*op.constraint); break;
        case common_plan_operation_kind::add_assumption: next.assumptions.push_back(*op.assumption); break;
        case common_plan_operation_kind::invalidate_assumption: for (auto & a : next.assumptions) if (a.id == *op.target_id) a.valid = false; break;
        case common_plan_operation_kind::record_observation: next.observations.push_back(*op.observation); break;
        case common_plan_operation_kind::set_next_action: next.next_action = op.value; break;
        case common_plan_operation_kind::activate_step: case common_plan_operation_kind::unblock_step: find(*op.step_id)->status = common_plan_step_status::active; next.active_step_id = op.step_id; next.status = common_plan_status::active; break;
        case common_plan_operation_kind::reset_step: find(*op.step_id)->status = common_plan_step_status::pending; find(*op.step_id)->result_summary.reset(); find(*op.step_id)->blocked_by.clear(); if (next.active_step_id == op.step_id) next.active_step_id.reset(); next.status = common_plan_status::active; break;
        case common_plan_operation_kind::complete_step: find(*op.step_id)->status = common_plan_step_status::completed; if (next.active_step_id == op.step_id) next.active_step_id.reset(); break;
        case common_plan_operation_kind::block_step: find(*op.step_id)->status = common_plan_step_status::blocked; if (next.active_step_id == op.step_id) next.active_step_id.reset(); break;
        case common_plan_operation_kind::fail_step: find(*op.step_id)->status = common_plan_step_status::failed; if (next.active_step_id == op.step_id) next.active_step_id.reset(); break;
        case common_plan_operation_kind::skip_step: find(*op.step_id)->status = common_plan_step_status::skipped; break;
        case common_plan_operation_kind::complete_plan: next.status = common_plan_status::completed; break;
        case common_plan_operation_kind::fail_plan: next.status = common_plan_status::failed; break;
        default: break;
    }
    ++next.version; it->second = next; updated = next; events[op.plan_id].push_back({(uint64_t) events[op.plan_id].size() + 1, op.expected_version, next.version, op, true, op.reason_summary, 0}); error.clear(); return true;
}
