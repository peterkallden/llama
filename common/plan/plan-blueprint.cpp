#include "plan/plan-blueprint.h"

#include <algorithm>
#include <queue>
#include <unordered_set>
#include <unordered_map>

namespace {

bool bounded_text(const std::string & value, size_t maximum) {
    return value.size() <= maximum;
}

bool unique_id(const std::string & id, std::unordered_set<std::string> & ids) {
    return !id.empty() && ids.insert(id).second;
}

} // namespace

bool common_plan_validate_blueprint(
        const common_plan_state & blueprint,
        const common_plan_blueprint_validation_config & config,
        std::string & error) {
    error.clear();
    if (blueprint.kind != common_plan_kind::blueprint) {
        error = "source plan is not a blueprint";
        return false;
    }
    if (blueprint.id.empty() || blueprint.namespace_id.empty()) {
        error = "blueprint requires an id and namespace";
        return false;
    }
    if (blueprint.source_revision.size() > 128) {
        error = "blueprint source revision exceeds bounds";
        return false;
    }
    if (blueprint.steps.empty() || blueprint.steps.size() > config.maximum_steps) {
        error = "blueprint step count is outside bounds";
        return false;
    }
    if (!bounded_text(blueprint.purpose, config.maximum_text_bytes) ||
            !bounded_text(blueprint.goal, config.maximum_text_bytes) ||
            !bounded_text(blueprint.success_criteria, config.maximum_text_bytes) ||
            (blueprint.next_action && !bounded_text(*blueprint.next_action, config.maximum_text_bytes))) {
        error = "blueprint text exceeds bounds";
        return false;
    }

    std::unordered_set<std::string> step_ids;
    for (const auto & step : blueprint.steps) {
        if (!unique_id(step.id, step_ids) || step.id.size() > config.maximum_text_bytes ||
                !bounded_text(step.title, config.maximum_text_bytes) ||
                !bounded_text(step.objective, config.maximum_text_bytes) ||
                !bounded_text(step.intended_contribution, config.maximum_text_bytes) ||
                step.depends_on.size() > config.maximum_dependencies_per_step) {
            error = "blueprint contains an invalid or oversized step";
            return false;
        }
    }

    std::unordered_map<std::string, size_t> indegree;
    std::unordered_map<std::string, std::vector<std::string>> dependents;
    for (const auto & step : blueprint.steps) {
        indegree[step.id] = step.depends_on.size();
        std::unordered_set<std::string> dependencies;
        for (const auto & dependency : step.depends_on) {
            if (!step_ids.count(dependency) || dependency == step.id ||
                    !dependencies.insert(dependency).second) {
                error = "blueprint contains an invalid or duplicate dependency";
                return false;
            }
            dependents[dependency].push_back(step.id);
        }
    }
    std::queue<std::string> ready;
    for (const auto & step : blueprint.steps) if (indegree[step.id] == 0) ready.push(step.id);
    size_t visited = 0;
    while (!ready.empty()) {
        const auto id = ready.front();
        ready.pop();
        ++visited;
        for (const auto & dependent : dependents[id]) {
            if (--indegree[dependent] == 0) ready.push(dependent);
        }
    }
    if (visited != blueprint.steps.size()) {
        error = "blueprint step dependencies contain a cycle";
        return false;
    }

    std::unordered_set<std::string> constraint_ids;
    for (const auto & constraint : blueprint.constraints) {
        if (!unique_id(constraint.id, constraint_ids) ||
                !bounded_text(constraint.description, config.maximum_text_bytes)) {
            error = "blueprint contains an invalid constraint";
            return false;
        }
    }
    std::unordered_set<std::string> assumption_ids;
    for (const auto & assumption : blueprint.assumptions) {
        if (!unique_id(assumption.id, assumption_ids) ||
                !bounded_text(assumption.statement, config.maximum_text_bytes) ||
                !assumption.valid) {
            error = "blueprint contains an invalid assumption";
            return false;
        }
    }
    return true;
}

bool common_plan_instantiate_blueprint(
        const common_plan_state & blueprint,
        const std::string & instance_id,
        const std::string & session_id,
        common_plan_state & instance,
        std::string & error,
        common_plan_scope scope,
        int64_t now) {
    if (!common_plan_validate_blueprint(blueprint, {}, error)) return false;
    if (blueprint.id.empty() || instance_id.empty() || session_id.empty()) { error = "blueprint id, instance id and session id are required"; return false; }
    std::unordered_map<std::string, std::string> ids;
    for (const auto & step : blueprint.steps) {
        if (step.id.empty() || !ids.emplace(step.id, instance_id + ":" + step.id).second) { error = "blueprint contains an empty or duplicate step id"; return false; }
    }
    common_plan_state out;
    out.id = instance_id;
    out.session_id = session_id;
    out.source_revision = blueprint.source_revision;
    out.kind = common_plan_kind::task;
    out.derived_from_plan_id = blueprint.id;
    out.scope = scope;
    out.purpose = blueprint.purpose.empty() ? blueprint.goal : blueprint.purpose;
    out.goal = blueprint.goal;
    out.success_criteria = blueprint.success_criteria;
    out.required_capabilities = blueprint.required_capabilities;
    out.constraints = blueprint.constraints;
    out.assumptions = blueprint.assumptions;
    out.next_action = blueprint.next_action;
    out.created_at = now;
    out.updated_at = now;
    for (const auto & source : blueprint.steps) {
        auto step = source;
        step.id = ids.at(source.id);
        step.status = common_plan_step_status::pending;
        step.blocked_by.clear();
        step.selected_tool.reset();
        step.tool_call.reset();
        step.result_summary.reset();
        step.created_at = now;
        step.updated_at = now;
        for (auto & dependency : step.depends_on) {
            const auto found = ids.find(dependency);
            if (found == ids.end()) { error = "blueprint step references an unknown dependency"; return false; }
            dependency = found->second;
        }
        out.steps.push_back(std::move(step));
    }
    for (auto & step : out.steps) if (step.depends_on.empty()) {
        step.status = common_plan_step_status::active;
        out.active_step_id = step.id;
        out.status = common_plan_status::active;
        break;
    }
    instance = std::move(out);
    error.clear();
    return true;
}
