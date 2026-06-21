#include "agent/blueprint-selector.h"

#include <algorithm>

common_explicit_blueprint_selector::common_explicit_blueprint_selector(std::string logical_id) : logical_id(std::move(logical_id)) {}

common_blueprint_selection common_explicit_blueprint_selector::select(
        const common_agent_request &,
        const std::vector<common_blueprint_candidate> &,
        std::string & error) {
    error.clear();
    return {common_blueprint_selection_decision::instantiate, logical_id, 1.0f, "explicitly selected"};
}

bool common_agent_select_and_instantiate_blueprint(
        common_plan_store & plan_store,
        const common_agent_request & request,
        common_blueprint_selector & selector,
        const std::vector<common_blueprint_candidate> & candidates,
        const common_blueprint_selection_config & config,
        common_blueprint_selection_result & result,
        std::string & error) {
    result = {};
    error.clear();
    if (config.task_plan_id.empty() || config.session_id.empty()) {
        error = "blueprint selection requires task plan and session ids";
        return false;
    }
    if (candidates.empty() || candidates.size() > config.maximum_candidates) {
        result.outcome = common_blueprint_selection_outcome::failed_safely;
        result.reason = candidates.empty() ? "no installed blueprint candidates" : "too many blueprint candidates";
        return true;
    }

    const auto existing = plan_store.get(config.task_plan_id, error);
    if (!error.empty()) return false;
    if (existing) {
        if (existing->kind != common_plan_kind::task || existing->scope != config.scope || existing->session_id != config.session_id) {
            result.outcome = common_blueprint_selection_outcome::failed_safely;
            result.reason = "existing plan is not a compatible task plan";
            return true;
        }
        result.outcome = common_blueprint_selection_outcome::resumed;
        result.reason = "existing task plan takes precedence";
        return true;
    }

    std::string selection_error;
    const auto choice = selector.select(request, candidates, selection_error);
    result.confidence = choice.confidence;
    result.reason = choice.reason.empty() ? selection_error : choice.reason;
    if (!selection_error.empty() || choice.decision == common_blueprint_selection_decision::failed) {
        result.outcome = common_blueprint_selection_outcome::failed_safely;
        return true;
    }
    if (choice.decision != common_blueprint_selection_decision::instantiate || !choice.logical_id || choice.confidence < config.minimum_confidence) {
        result.outcome = common_blueprint_selection_outcome::declined;
        return true;
    }
    const auto candidate = std::find_if(candidates.begin(), candidates.end(), [&](const auto & value) {
        return value.logical_id == *choice.logical_id;
    });
    if (candidate == candidates.end()) {
        result.outcome = common_blueprint_selection_outcome::failed_safely;
        result.reason = "selector returned an unavailable blueprint";
        return true;
    }
    const auto blueprint = plan_store.get(candidate->persisted_id, error);
    if (!error.empty()) return false;
    if (!blueprint || blueprint->kind != common_plan_kind::blueprint) {
        result.outcome = common_blueprint_selection_outcome::failed_safely;
        result.reason = "installed candidate is not a blueprint";
        return true;
    }
    common_plan_state instance;
    if (!common_plan_instantiate_blueprint(*blueprint, config.task_plan_id, config.session_id, instance, error, config.scope, config.now) ||
            !plan_store.create(instance, error)) return false;
    result.outcome = common_blueprint_selection_outcome::instantiated;
    result.logical_id = candidate->logical_id;
    return true;
}
