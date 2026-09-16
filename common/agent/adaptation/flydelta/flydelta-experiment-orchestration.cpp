#include "agent/adaptation/flydelta/flydelta-experiment-orchestration.h"

#include <cmath>
#include <limits>

namespace {
bool finite(float value) { return std::isfinite(value); }

bool valid_depth(const common_flydelta_evidence_depth_result & value) {
    return value.compatible_samples > 0 && value.effective_rank > 0 &&
        finite(value.stable_rank) && finite(value.median_alignment) &&
        finite(value.condition_number);
}

bool valid_depth_value(common_flydelta_search_depth value) {
    return value == common_flydelta_search_depth::bootstrap ||
        value == common_flydelta_search_depth::shallow ||
        value == common_flydelta_search_depth::deep;
}

bool safe_geometry(
        const common_flydelta_representation_diagnostics & value,
        const common_flydelta_utility_gate_config & config) {
    return value.cosine >= config.min_cosine && value.progress > 0.0f &&
        value.leakage <= config.max_leakage && value.shift_norm <= config.max_shift_norm;
}

size_t required_observations(
        common_flydelta_experiment_phase phase,
        const common_flydelta_utility_gate_config & config) {
    switch (phase) {
        case common_flydelta_experiment_phase::bootstrap: return config.shallow_enter_observations;
        case common_flydelta_experiment_phase::shallow_controls: return config.deep_enter_observations;
        case common_flydelta_experiment_phase::deep_controls: return config.tfo_enter_observations;
    }
    return config.tfo_enter_observations;
}

float required_margin(
        common_flydelta_experiment_phase phase,
        const common_flydelta_utility_gate_config & config) {
    switch (phase) {
        case common_flydelta_experiment_phase::bootstrap: return config.shallow_enter_margin;
        case common_flydelta_experiment_phase::shallow_controls: return config.deep_enter_margin;
        case common_flydelta_experiment_phase::deep_controls: return config.tfo_enter_margin;
    }
    return config.tfo_enter_margin;
}
} // namespace

const char * common_flydelta_experiment_phase_name(
        common_flydelta_experiment_phase phase) {
    switch (phase) {
        case common_flydelta_experiment_phase::bootstrap: return "bootstrap";
        case common_flydelta_experiment_phase::shallow_controls: return "shallow_controls";
        case common_flydelta_experiment_phase::deep_controls: return "deep_controls";
    }
    return "bootstrap";
}

const char * common_flydelta_utility_gate_action_name(
        common_flydelta_utility_gate_action action) {
    switch (action) {
        case common_flydelta_utility_gate_action::stop: return "stop";
        case common_flydelta_utility_gate_action::retain: return "retain";
        case common_flydelta_utility_gate_action::escalate_shallow: return "escalate_shallow";
        case common_flydelta_utility_gate_action::escalate_deep: return "escalate_deep";
        case common_flydelta_utility_gate_action::allow_tfo_lite: return "allow_tfo_lite";
    }
    return "retain";
}

bool common_flydelta_utility_gate_config_validate(
        const common_flydelta_utility_gate_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || !finite(config.shallow_enter_margin) ||
            !finite(config.deep_enter_margin) || !finite(config.tfo_enter_margin) ||
            config.shallow_enter_observations == 0 || config.deep_enter_observations == 0 ||
            config.tfo_enter_observations == 0 || config.exit_nonqualifying_observations == 0 ||
            !finite(config.min_cosine) || config.min_cosine < -1.0f || config.min_cosine > 1.0f ||
            !finite(config.max_leakage) || config.max_leakage < 0.0f ||
            !finite(config.max_shift_norm) || config.max_shift_norm <= 0.0f) {
        error = "FlyDelta utility gate configuration is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_subspace_utility_observation_validate(
        const common_flydelta_subspace_utility_observation & observation,
        std::string & error) {
    error.clear();
    if (!finite(observation.decision_margin_delta) ||
            (observation.geometry_available &&
                !common_flydelta_representation_diagnostics_validate(observation.geometry, error))) {
        if (error.empty()) error = "FlyDelta subspace utility observation is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_decide_subspace_utility(
        const common_flydelta_utility_gate_config & config,
        common_flydelta_search_depth max_allowed_depth,
        common_flydelta_experiment_phase current_phase,
        const std::vector<common_flydelta_subspace_utility_observation> & observations,
        const common_flydelta_utility_history & history,
        common_flydelta_utility_gate_decision & decision,
        std::string & error) {
    error.clear();
    decision = {};
    if (!common_flydelta_utility_gate_config_validate(config, error) ||
            !valid_depth_value(max_allowed_depth) || observations.empty()) {
        if (error.empty()) error = "FlyDelta utility gate input is invalid";
        return false;
    }
    bool qualified = false;
    for (const auto & observation : observations) {
        if (!common_flydelta_subspace_utility_observation_validate(observation, error)) return false;
        if (!observation.safe_to_continue) {
            decision.action = common_flydelta_utility_gate_action::stop;
            decision.history = {0, history.nonqualifying_streak + 1};
            return true;
        }
        const bool geometry_ok = !observation.geometry_available || safe_geometry(observation.geometry, config);
        if (observation.decision_margin_available && geometry_ok &&
                observation.decision_margin_delta > required_margin(current_phase, config)) {
            qualified = true;
        }
    }
    decision.utility_qualified = qualified;
    decision.history = qualified
        ? common_flydelta_utility_history{history.qualifying_streak + 1, 0}
        : common_flydelta_utility_history{0, history.nonqualifying_streak + 1};
    if (!qualified) {
        decision.action = decision.history.nonqualifying_streak >=
                config.exit_nonqualifying_observations
            ? common_flydelta_utility_gate_action::stop
            : common_flydelta_utility_gate_action::retain;
        return true;
    }
    if (decision.history.qualifying_streak < required_observations(current_phase, config)) {
        decision.action = common_flydelta_utility_gate_action::retain;
        return true;
    }
    if (current_phase == common_flydelta_experiment_phase::bootstrap &&
            max_allowed_depth != common_flydelta_search_depth::bootstrap) {
        decision.action = common_flydelta_utility_gate_action::escalate_shallow;
    } else if (current_phase == common_flydelta_experiment_phase::shallow_controls &&
            max_allowed_depth == common_flydelta_search_depth::deep) {
        decision.action = common_flydelta_utility_gate_action::escalate_deep;
    } else if (current_phase == common_flydelta_experiment_phase::deep_controls &&
            max_allowed_depth == common_flydelta_search_depth::deep) {
        decision.action = common_flydelta_utility_gate_action::allow_tfo_lite;
    } else {
        decision.action = common_flydelta_utility_gate_action::retain;
    }
    return true;
}

bool common_flydelta_select_search_continuation(
        const common_flydelta_search_pipeline_result & pipeline,
        common_flydelta_search_continuation & continuation,
        std::string & error) {
    error.clear();
    continuation = {};
    if (pipeline.directions.empty()) {
        error = "FlyDelta continuation requires a non-empty search pipeline result";
        return false;
    }
    bool found = false;
    float best_score = -std::numeric_limits<float>::infinity();
    for (size_t direction_index = 0; direction_index < pipeline.directions.size(); ++direction_index) {
        const auto & direction = pipeline.directions[direction_index];
        for (size_t trial_index = 0; trial_index < direction.region_trials.size(); ++trial_index) {
            const auto & trial = direction.region_trials[trial_index];
            if (!common_flydelta_intervention_region_trial_validate(trial, error)) return false;
            if (!trial.executed || !trial.safe_to_continue || !trial.promising ||
                    !finite(trial.search_score)) continue;
            const bool helped = trial.outcome == common_flydelta_counterfactual_outcome::helped &&
                trial.verifier_known;
            if (!found || (helped && !continuation.host_helped) ||
                    (helped == continuation.host_helped && trial.search_score > best_score)) {
                found = true;
                best_score = trial.search_score;
                continuation.direction_index = direction_index;
                continuation.region_trial_index = trial_index;
                continuation.region = trial.candidate;
                continuation.search_score = trial.search_score;
                continuation.host_helped = helped;
            }
        }
    }
    if (!found) {
        error = "FlyDelta search pipeline produced no safe promising region continuation";
        return false;
    }
    return true;
}

bool common_flydelta_plan_search_continuation(
        const common_flydelta_search_continuation & continuation,
        const common_flydelta_evidence_depth_result & evidence_depth,
        common_flydelta_experiment_plan & plan,
        std::string & error) {
    error.clear();
    plan = {};
    if (continuation.schema_version != 1 || continuation.region.layer_indices.empty() ||
            continuation.region.anchor_layer_index == 0 || !finite(continuation.search_score) ||
            !valid_depth(evidence_depth)) {
        error = "FlyDelta continuation or evidence depth is invalid";
        return false;
    }
    plan.continuation = continuation;
    plan.depth = evidence_depth.depth;
    plan.budget = common_flydelta_search_budget_for_depth(evidence_depth.depth);
    if (!common_flydelta_search_budget_validate(plan.budget, error)) return false;
    switch (evidence_depth.depth) {
        case common_flydelta_search_depth::bootstrap:
            plan.phase = common_flydelta_experiment_phase::bootstrap;
            break;
        case common_flydelta_search_depth::shallow:
            plan.phase = common_flydelta_experiment_phase::shallow_controls;
            plan.required_compatible_directions = 2;
            plan.require_decision_margin = true;
            plan.run_rank_two_controls_first = true;
            break;
        case common_flydelta_search_depth::deep:
            plan.phase = common_flydelta_experiment_phase::deep_controls;
            plan.required_compatible_directions = 2;
            plan.require_decision_margin = true;
            plan.run_rank_two_controls_first = true;
            plan.tfo_lite_permitted_by_evidence = true;
            plan.tfo_lite_requires_utility_gate = true;
            break;
    }
    if ((plan.required_compatible_directions == 2 && evidence_depth.effective_rank < 2) ||
            (plan.tfo_lite_permitted_by_evidence && !plan.budget.allow_tfo_lite)) {
        error = "FlyDelta evidence depth does not support its continuation plan";
        return false;
    }
    return true;
}
