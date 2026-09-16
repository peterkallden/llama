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
            plan.allow_tfo_lite_after_controls = true;
            break;
    }
    if ((plan.required_compatible_directions == 2 && evidence_depth.effective_rank < 2) ||
            (plan.allow_tfo_lite_after_controls && !plan.budget.allow_tfo_lite)) {
        error = "FlyDelta evidence depth does not support its continuation plan";
        return false;
    }
    return true;
}
