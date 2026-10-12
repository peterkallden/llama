#include "agent/adaptation/flydelta/flydelta-experiment-orchestration.h"
#include "agent/adaptation/flydelta/experiment/flydelta-experiment-routes.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace {
bool finite(float value) { return std::isfinite(value); }

bool valid_depth(const common_flydelta_evidence_depth_result & value) {
    // A search surface may be continued experimentally before any natural
    // HELPED evidence exists. That is still Bootstrap capacity; it must not
    // be mistaken for rank-backed Shallow/Deep capacity.
    const bool empty_bootstrap = value.compatible_samples == 0 &&
        value.experimental_samples > 0 &&
        value.depth == common_flydelta_search_depth::bootstrap &&
        value.effective_rank == 0;
    const bool ranked = value.compatible_samples > 0 && value.effective_rank > 0;
    return (empty_bootstrap || ranked) && finite(value.stable_rank) &&
        finite(value.median_alignment) && finite(value.condition_number);
}

bool valid_depth_value(common_flydelta_search_depth value) {
    return value == common_flydelta_search_depth::bootstrap ||
        value == common_flydelta_search_depth::shallow ||
        value == common_flydelta_search_depth::deep;
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

const char * common_flydelta_bootstrap_refinement_kind_name(
        common_flydelta_bootstrap_refinement_kind kind) {
    switch (kind) {
        case common_flydelta_bootstrap_refinement_kind::bootstrap_zoom:
            return "bootstrap_zoom";
        case common_flydelta_bootstrap_refinement_kind::adaptive_alpha:
            return "adaptive_alpha";
    }
    return "bootstrap_zoom";
}

const char * common_flydelta_utility_gate_action_name(
        common_flydelta_utility_gate_action action) {
    switch (action) {
        case common_flydelta_utility_gate_action::stop: return "stop";
        case common_flydelta_utility_gate_action::retain: return "retain";
        case common_flydelta_utility_gate_action::refine_bootstrap: return "refine_bootstrap";
        case common_flydelta_utility_gate_action::orthogonal_search: return "orthogonal_search";
        case common_flydelta_utility_gate_action::escalate_shallow: return "escalate_shallow";
        case common_flydelta_utility_gate_action::escalate_deep: return "escalate_deep";
        case common_flydelta_utility_gate_action::allow_tfo_lite: return "allow_tfo_lite";
    }
    return "retain";
}

const char * common_flydelta_next_action_name(common_flydelta_next_action action) {
    switch (action) {
        case common_flydelta_next_action::stop: return "stop";
        case common_flydelta_next_action::retain: return "retain";
        case common_flydelta_next_action::run_bootstrap: return "run_bootstrap";
        case common_flydelta_next_action::refine_bootstrap: return "refine_bootstrap";
        case common_flydelta_next_action::run_orthogonal_search: return "run_orthogonal_search";
        case common_flydelta_next_action::run_shallow_controls: return "run_shallow_controls";
        case common_flydelta_next_action::run_deep_controls: return "run_deep_controls";
        case common_flydelta_next_action::recenter_surface: return "recenter_surface";
        case common_flydelta_next_action::run_representation_augmentation:
            return "run_representation_augmentation";
        case common_flydelta_next_action::prepare_concept_material:
            return "prepare_concept_material";
        case common_flydelta_next_action::run_concept_synthesis:
            return "run_concept_synthesis";
        case common_flydelta_next_action::allow_tfo_lite: return "allow_tfo_lite";
    }
    return "retain";
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
            if (!trial.executed ||
                    trial.outcome == common_flydelta_counterfactual_outcome::harmed ||
                    !trial.safe_to_continue || !trial.promising ||
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
        // A bounded search may complete normally without finding a safe,
        // promising continuation. The worker records that terminal result;
        // it is not a callback or persistence failure.
        if (common_flydelta_search_status_is_terminal_without_candidate(
                pipeline.search_status)) return true;
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
    // Start every retained region at Bootstrap. Depth expresses capacity, not
    // permission to skip the rank-one/rank-two evidence chain.
    plan.depth = evidence_depth.depth;
    plan.phase = common_flydelta_experiment_phase::bootstrap;
    plan.bootstrap_refinement = common_flydelta_bootstrap_refinement_kind::bootstrap_zoom;
    plan.budget = common_flydelta_search_budget_for_depth(
        common_flydelta_search_depth::bootstrap);
    if (!common_flydelta_search_budget_validate(plan.budget, error)) return false;
    plan.tfo_lite_permitted_by_evidence = evidence_depth.depth ==
        common_flydelta_search_depth::deep;
    plan.tfo_lite_requires_utility_gate = plan.tfo_lite_permitted_by_evidence;
    return true;
}

bool common_flydelta_advance_experiment_plan(
        const common_flydelta_experiment_plan & current,
        const common_flydelta_utility_gate_decision & utility,
        common_flydelta_experiment_plan & next,
        bool & advanced,
        std::string & error) {
    error.clear();
    next = current;
    advanced = false;
    if (current.continuation.region.layer_indices.empty() ||
            !valid_depth_value(current.depth) ||
            !common_flydelta_search_budget_validate(current.budget, error)) {
        if (error.empty()) error = "FlyDelta experiment plan transition input is invalid";
        return false;
    }
    const auto configure_phase = [&](common_flydelta_experiment_phase phase,
            common_flydelta_search_depth budget_depth) {
        next.phase = phase;
        next.budget = common_flydelta_search_budget_for_depth(budget_depth);
        next.required_compatible_directions = phase == common_flydelta_experiment_phase::bootstrap
            ? 1 : 2;
        next.require_decision_margin = phase != common_flydelta_experiment_phase::bootstrap;
        next.run_rank_two_controls_first = phase != common_flydelta_experiment_phase::bootstrap;
        next.run_tfo_lite = false;
        next.run_orthogonal_search = false;
    };
    switch (utility.action) {
        case common_flydelta_utility_gate_action::stop:
        case common_flydelta_utility_gate_action::retain:
            return true;
        case common_flydelta_utility_gate_action::refine_bootstrap:
            if (current.phase != common_flydelta_experiment_phase::bootstrap ||
                    current.depth != common_flydelta_search_depth::bootstrap) {
                error = "FlyDelta BootstrapZoom refinement is not permitted by the current plan";
                return false;
            }
            configure_phase(common_flydelta_experiment_phase::bootstrap,
                common_flydelta_search_depth::bootstrap);
            next.bootstrap_refinement = common_flydelta_bootstrap_refinement_kind::adaptive_alpha;
            advanced = true;
            break;
        case common_flydelta_utility_gate_action::orthogonal_search:
            if (current.phase != common_flydelta_experiment_phase::bootstrap) {
                error = "FlyDelta orthogonal search requires the Bootstrap phase";
                return false;
            }
            // This is deliberately a same-phase experimental operation. The
            // host must keep evidence depth unchanged and later evaluate the
            // resulting rank-two controls before allowing Deep/TFO.
            next.run_orthogonal_search = true;
            next.run_rank_two_controls_first = true;
            next.run_tfo_lite = false;
            advanced = true;
            break;
        case common_flydelta_utility_gate_action::escalate_shallow:
            if (current.phase != common_flydelta_experiment_phase::bootstrap ||
                    current.depth == common_flydelta_search_depth::bootstrap) {
                error = "FlyDelta Shallow escalation is not permitted by the current plan";
                return false;
            }
            configure_phase(common_flydelta_experiment_phase::shallow_controls,
                common_flydelta_search_depth::shallow);
            advanced = true;
            break;
        case common_flydelta_utility_gate_action::escalate_deep:
            if (current.phase != common_flydelta_experiment_phase::shallow_controls ||
                    current.depth != common_flydelta_search_depth::deep) {
                error = "FlyDelta Deep escalation requires successful Shallow controls";
                return false;
            }
            configure_phase(common_flydelta_experiment_phase::deep_controls,
                common_flydelta_search_depth::deep);
            advanced = true;
            break;
        case common_flydelta_utility_gate_action::allow_tfo_lite:
            if (current.phase != common_flydelta_experiment_phase::deep_controls ||
                    current.depth != common_flydelta_search_depth::deep ||
                    !current.tfo_lite_permitted_by_evidence ||
                    !current.tfo_lite_requires_utility_gate) {
                error = "FlyDelta TFO-lite requires Deep controls and evidence permission";
                return false;
            }
            next.run_tfo_lite = true;
            advanced = true;
            break;
    }
    return common_flydelta_search_budget_validate(next.budget, error);
}

bool common_flydelta_orchestrate_search_slice(
        const common_flydelta_experiment_plan & current,
        const common_flydelta_utility_gate_config & utility_config,
        const std::vector<common_flydelta_subspace_utility_observation> & observations,
        const common_flydelta_utility_history & history,
        common_flydelta_slice_orchestration_result & result,
        std::string & error) {
    return common_flydelta_route_experiment_slice(
            current, utility_config, observations, history, result, error);
}
