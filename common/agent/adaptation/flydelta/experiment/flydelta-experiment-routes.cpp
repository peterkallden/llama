#include "agent/adaptation/flydelta/experiment/flydelta-experiment-routes.h"

namespace {

bool run_existing_slice_policy(
        const common_flydelta_experiment_plan & current,
        const common_flydelta_utility_gate_config & utility_config,
        const std::vector<common_flydelta_subspace_utility_observation> & observations,
        const common_flydelta_utility_history & history,
        common_flydelta_slice_orchestration_result & result,
        std::string & error) {
    error.clear();
    result = {};
    if (!common_flydelta_search_budget_validate(current.budget, error) ||
            observations.empty()) {
        if (error.empty()) error = "FlyDelta bounded-slice orchestration input is invalid";
        return false;
    }
    if (!common_flydelta_decide_subspace_utility(
            utility_config, current.depth, current.phase, observations, history,
            result.utility, error)) return false;
    if (result.utility.action != common_flydelta_utility_gate_action::stop &&
            result.utility.action != common_flydelta_utility_gate_action::retain &&
            !result.utility.utility_qualified) {
        error = "FlyDelta non-retain transition requires qualified utility";
        return false;
    }
    if (!common_flydelta_advance_experiment_plan(
            current, result.utility, result.plan, result.plan_advanced, error)) return false;
    switch (result.utility.action) {
        case common_flydelta_utility_gate_action::stop:
            result.next_action = common_flydelta_next_action::stop;
            break;
        case common_flydelta_utility_gate_action::retain:
            result.next_action = common_flydelta_next_action::retain;
            break;
        case common_flydelta_utility_gate_action::refine_bootstrap:
            result.next_action = common_flydelta_next_action::refine_bootstrap;
            break;
        case common_flydelta_utility_gate_action::orthogonal_search:
            result.next_action = common_flydelta_next_action::run_orthogonal_search;
            break;
        case common_flydelta_utility_gate_action::escalate_shallow:
            result.next_action = common_flydelta_next_action::run_shallow_controls;
            break;
        case common_flydelta_utility_gate_action::escalate_deep:
            result.next_action = common_flydelta_next_action::run_deep_controls;
            break;
        case common_flydelta_utility_gate_action::allow_tfo_lite:
            result.next_action = common_flydelta_next_action::allow_tfo_lite;
            break;
    }
    result.reason = common_flydelta_utility_gate_action_name(result.utility.action);
    return true;
}

} // namespace

bool common_flydelta_route_bootstrap(
        const common_flydelta_experiment_plan & current,
        const common_flydelta_utility_gate_config & utility_config,
        const std::vector<common_flydelta_subspace_utility_observation> & observations,
        const common_flydelta_utility_history & history,
        common_flydelta_slice_orchestration_result & result,
        std::string & error) {
    return run_existing_slice_policy(
            current, utility_config, observations, history, result, error);
}

bool common_flydelta_route_shallow_controls(
        const common_flydelta_experiment_plan & current,
        const common_flydelta_utility_gate_config & utility_config,
        const std::vector<common_flydelta_subspace_utility_observation> & observations,
        const common_flydelta_utility_history & history,
        common_flydelta_slice_orchestration_result & result,
        std::string & error) {
    return run_existing_slice_policy(
            current, utility_config, observations, history, result, error);
}

bool common_flydelta_route_deep_controls(
        const common_flydelta_experiment_plan & current,
        const common_flydelta_utility_gate_config & utility_config,
        const std::vector<common_flydelta_subspace_utility_observation> & observations,
        const common_flydelta_utility_history & history,
        common_flydelta_slice_orchestration_result & result,
        std::string & error) {
    return run_existing_slice_policy(
            current, utility_config, observations, history, result, error);
}

bool common_flydelta_route_experiment_slice(
        const common_flydelta_experiment_plan & current,
        const common_flydelta_utility_gate_config & utility_config,
        const std::vector<common_flydelta_subspace_utility_observation> & observations,
        const common_flydelta_utility_history & history,
        common_flydelta_slice_orchestration_result & result,
        std::string & error) {
    switch (current.phase) {
        case common_flydelta_experiment_phase::bootstrap:
            return common_flydelta_route_bootstrap(
                    current, utility_config, observations, history, result, error);
        case common_flydelta_experiment_phase::shallow_controls:
            return common_flydelta_route_shallow_controls(
                    current, utility_config, observations, history, result, error);
        case common_flydelta_experiment_phase::deep_controls:
            return common_flydelta_route_deep_controls(
                    current, utility_config, observations, history, result, error);
    }
    error = "FlyDelta experiment route phase is invalid";
    result = {};
    return false;
}
