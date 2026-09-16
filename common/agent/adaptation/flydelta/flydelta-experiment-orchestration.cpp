#include "agent/adaptation/flydelta/flydelta-experiment-orchestration.h"

#include <cmath>
#include <limits>

namespace {
bool finite(float value) { return std::isfinite(value); }

float l2_norm(const std::vector<float> & values) {
    float total = 0.0f;
    for (const float value : values) total += value * value;
    return std::sqrt(total);
}

bool valid_zoom_phase(common_flydelta_bootstrap_zoom_phase phase) {
    return phase == common_flydelta_bootstrap_zoom_phase::alpha_zoom ||
        phase == common_flydelta_bootstrap_zoom_phase::profile_zoom ||
        phase == common_flydelta_bootstrap_zoom_phase::sign_control;
}

bool valid_zoom_candidate(
        const common_flydelta_bootstrap_zoom_candidate & candidate,
        std::string & error) {
    error.clear();
    if (candidate.schema_version != 1 || !valid_zoom_phase(candidate.phase) ||
            candidate.layer_indices.empty() || candidate.layer_indices.size() > 3 ||
            candidate.layer_indices.size() != candidate.layer_weights.size() ||
            !std::is_sorted(candidate.layer_indices.begin(), candidate.layer_indices.end()) ||
            candidate.layer_indices.front() == 0 ||
            std::adjacent_find(candidate.layer_indices.begin(), candidate.layer_indices.end()) !=
                candidate.layer_indices.end() || !finite(candidate.total_scale) ||
            candidate.total_scale <= 0.0f || candidate.total_scale > 1.0f) {
        error = "FlyDelta BootstrapZoom candidate is invalid";
        return false;
    }
    for (const float weight : candidate.layer_weights) {
        if (!finite(weight)) {
            error = "FlyDelta BootstrapZoom layer weight is invalid";
            return false;
        }
    }
    if (std::fabs(l2_norm(candidate.layer_weights) - 1.0f) > 0.0001f) {
        error = "FlyDelta BootstrapZoom profile must have unit L2 energy";
        return false;
    }
    if (candidate.phase == common_flydelta_bootstrap_zoom_phase::alpha_zoom &&
            (candidate.layer_indices.size() != 1 || candidate.opposite_sign_control)) {
        error = "FlyDelta BootstrapZoom alpha candidate must be a positive singleton";
        return false;
    }
    if (candidate.phase == common_flydelta_bootstrap_zoom_phase::sign_control &&
            !candidate.opposite_sign_control) {
        error = "FlyDelta BootstrapZoom sign control must be marked";
        return false;
    }
    return true;
}

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
        case common_flydelta_utility_gate_action::refine_bootstrap: return "refine_bootstrap";
        case common_flydelta_utility_gate_action::escalate_shallow: return "escalate_shallow";
        case common_flydelta_utility_gate_action::escalate_deep: return "escalate_deep";
        case common_flydelta_utility_gate_action::allow_tfo_lite: return "allow_tfo_lite";
    }
    return "retain";
}

const char * common_flydelta_bootstrap_zoom_phase_name(
        common_flydelta_bootstrap_zoom_phase phase) {
    switch (phase) {
        case common_flydelta_bootstrap_zoom_phase::alpha_zoom: return "alpha_zoom";
        case common_flydelta_bootstrap_zoom_phase::profile_zoom: return "profile_zoom";
        case common_flydelta_bootstrap_zoom_phase::sign_control: return "sign_control";
    }
    return "alpha_zoom";
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
    if (current_phase == common_flydelta_experiment_phase::bootstrap) {
        decision.action = max_allowed_depth == common_flydelta_search_depth::bootstrap
            ? common_flydelta_utility_gate_action::refine_bootstrap
            : common_flydelta_utility_gate_action::escalate_shallow;
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

bool common_flydelta_bootstrap_zoom_config_validate(
        const common_flydelta_bootstrap_zoom_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || config.max_extra_model_trials == 0 ||
            config.max_extra_model_trials > 10 || config.alpha_multipliers.empty() ||
            config.alpha_multipliers.size() > 4 || !finite(config.min_margin_improvement) ||
            config.min_margin_improvement < 0.0f) {
        error = "FlyDelta BootstrapZoom configuration is invalid";
        return false;
    }
    for (size_t index = 0; index < config.alpha_multipliers.size(); ++index) {
        const float multiplier = config.alpha_multipliers[index];
        if (!finite(multiplier) || multiplier <= 0.0f || multiplier > 4.0f ||
                (index > 0 && multiplier <= config.alpha_multipliers[index - 1])) {
            error = "FlyDelta BootstrapZoom alpha multipliers are invalid";
            return false;
        }
    }
    return true;
}

bool common_flydelta_bootstrap_zoom_candidate_validate(
        const common_flydelta_bootstrap_zoom_candidate & candidate,
        std::string & error) {
    return valid_zoom_candidate(candidate, error);
}

bool common_flydelta_bootstrap_zoom_trial_validate(
        const common_flydelta_bootstrap_zoom_trial & trial,
        std::string & error) {
    error.clear();
    if (!valid_zoom_candidate(trial.candidate, error) || !trial.host_verified ||
            !finite(trial.margin_delta) ||
            (trial.diagnostics_available &&
                !common_flydelta_representation_diagnostics_validate(
                    trial.diagnostics, error))) {
        if (error.empty()) error = "FlyDelta BootstrapZoom trial is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_bootstrap_zoom_selection_validate(
        const common_flydelta_bootstrap_zoom_selection & selection,
        size_t trial_count,
        std::string & error) {
    error.clear();
    if (!finite(selection.search_score) ||
            (selection.selected && selection.trial_index >= trial_count) ||
            (!selection.selected && selection.trial_index != 0)) {
        error = "FlyDelta BootstrapZoom selection is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_select_bootstrap_zoom_trial(
        const std::vector<common_flydelta_bootstrap_zoom_trial> & trials,
        common_flydelta_bootstrap_zoom_selection & selection,
        std::string & error) {
    error.clear();
    selection = {};
    if (trials.empty() || trials.size() > 8) {
        error = "FlyDelta BootstrapZoom trial set is empty or exceeds its bound";
        return false;
    }
    for (size_t index = 0; index < trials.size(); ++index) {
        const auto & trial = trials[index];
        if (!common_flydelta_bootstrap_zoom_trial_validate(trial, error)) return false;
        if (trial.outcome == common_flydelta_counterfactual_outcome::harmed) continue;
        const bool geometry_safe = !trial.diagnostics_available ||
            (trial.diagnostics.cosine >= 0.3f && trial.diagnostics.progress > 0.0f &&
             trial.diagnostics.leakage <= 1.0f && trial.diagnostics.shift_norm <= 1.0f);
        const bool helped = trial.outcome == common_flydelta_counterfactual_outcome::helped;
        if (!helped && !geometry_safe) continue;
        const float score = trial.margin_available ? trial.margin_delta : 0.0f;
        if (!selection.selected) {
            selection = {true, index, score};
            continue;
        }
        const auto & best = trials[selection.trial_index];
        const bool best_helped = best.outcome == common_flydelta_counterfactual_outcome::helped;
        if ((helped && !best_helped) ||
                (helped == best_helped &&
                    (score > selection.search_score ||
                     (score == selection.search_score &&
                      trial.candidate.total_scale < best.candidate.total_scale)))) {
            selection = {true, index, score};
        }
    }
    if (!selection.selected) {
        error = "FlyDelta BootstrapZoom has no safe retained trial";
        return false;
    }
    return true;
}

bool common_flydelta_bootstrap_zoom_state_validate(
        const common_flydelta_bootstrap_zoom_state & state,
        std::string & error) {
    error.clear();
    if (state.schema_version != 1 || state.state_ref.size() > 512 ||
            state.behavior_key.empty() || state.behavior_key.size() > 512 ||
            state.model_profile_fingerprint.empty() ||
            state.model_profile_fingerprint.size() > 512 ||
            state.capture_layout_revision.empty() || state.capture_layout_revision.size() > 512 ||
            !valid_zoom_phase(state.phase) || state.anchor_layer == 0 ||
            !finite(state.selected_scale) || state.selected_scale <= 0.0f ||
            state.selected_scale > 1.0f || !finite(state.best_margin_delta) ||
            !finite(state.best_search_score) || state.extra_model_trials > 10 ||
            state.next_candidate_index > state.extra_model_trials ||
            state.local_layers.size() > 3 ||
            state.completed_trials.size() > 8 ||
            (!state.local_layers.empty() &&
                (!std::is_sorted(state.local_layers.begin(), state.local_layers.end()) ||
                 state.local_layers.front() == 0 ||
                 std::adjacent_find(state.local_layers.begin(), state.local_layers.end()) !=
                     state.local_layers.end()))) {
        error = "FlyDelta BootstrapZoom state is invalid";
        return false;
    }
    if (!state.local_layers.empty() && !std::binary_search(
            state.local_layers.begin(), state.local_layers.end(), state.anchor_layer)) {
        error = "FlyDelta BootstrapZoom state anchor is not local";
        return false;
    }
    for (const auto & trial : state.completed_trials) {
        if (!common_flydelta_bootstrap_zoom_trial_validate(trial, error)) return false;
    }
    if (!common_flydelta_bootstrap_zoom_selection_validate(
            state.selection, state.completed_trials.size(), error)) return false;
    return true;
}

bool common_flydelta_propose_bootstrap_alpha_zoom(
        uint32_t anchor_layer,
        float base_scale,
        const common_flydelta_bootstrap_zoom_config & config,
        std::vector<common_flydelta_bootstrap_zoom_candidate> & candidates,
        std::string & error) {
    error.clear();
    candidates.clear();
    if (anchor_layer == 0 || !finite(base_scale) || base_scale <= 0.0f ||
            !common_flydelta_bootstrap_zoom_config_validate(config, error)) {
        if (error.empty()) error = "FlyDelta BootstrapZoom alpha input is invalid";
        return false;
    }
    for (const float multiplier : config.alpha_multipliers) {
        const float scale = base_scale * multiplier;
        if (scale > 1.0f || candidates.size() >= config.max_extra_model_trials) continue;
        common_flydelta_bootstrap_zoom_candidate candidate;
        candidate.phase = common_flydelta_bootstrap_zoom_phase::alpha_zoom;
        candidate.layer_indices = {anchor_layer};
        candidate.layer_weights = {1.0f};
        candidate.total_scale = scale;
        if (!valid_zoom_candidate(candidate, error)) return false;
        candidates.push_back(std::move(candidate));
    }
    if (candidates.empty()) {
        error = "FlyDelta BootstrapZoom alpha probes exceed the intervention bound";
        return false;
    }
    return true;
}

bool common_flydelta_propose_bootstrap_profile_zoom(
        const std::vector<uint32_t> & local_layers,
        uint32_t anchor_layer,
        float selected_scale,
        const common_flydelta_bootstrap_zoom_config & config,
        std::vector<common_flydelta_bootstrap_zoom_candidate> & candidates,
        std::string & error) {
    error.clear();
    candidates.clear();
    if (local_layers.empty() || local_layers.size() > 3 || !std::is_sorted(
                local_layers.begin(), local_layers.end()) || local_layers.front() == 0 ||
            std::adjacent_find(local_layers.begin(), local_layers.end()) != local_layers.end() ||
            !std::binary_search(local_layers.begin(), local_layers.end(), anchor_layer) ||
            !finite(selected_scale) || selected_scale <= 0.0f || selected_scale > 1.0f ||
            !common_flydelta_bootstrap_zoom_config_validate(config, error)) {
        if (error.empty()) error = "FlyDelta BootstrapZoom profile input is invalid";
        return false;
    }
    const auto append = [&](std::vector<uint32_t> layers, bool sign_control) {
        common_flydelta_bootstrap_zoom_candidate candidate;
        candidate.phase = sign_control ? common_flydelta_bootstrap_zoom_phase::sign_control :
            common_flydelta_bootstrap_zoom_phase::profile_zoom;
        candidate.layer_indices = std::move(layers);
        candidate.layer_weights.assign(candidate.layer_indices.size(),
            (sign_control ? -1.0f : 1.0f) / std::sqrt(static_cast<float>(candidate.layer_indices.size())));
        candidate.total_scale = selected_scale;
        candidate.opposite_sign_control = sign_control;
        return candidate;
    };
    candidates.push_back(append({anchor_layer}, false));
    if (local_layers.size() >= 2 && candidates.size() < config.max_extra_model_trials) {
        const auto anchor = std::find(local_layers.begin(), local_layers.end(), anchor_layer);
        if (anchor != local_layers.begin() && candidates.size() < config.max_extra_model_trials) {
            candidates.push_back(append({*(anchor - 1), anchor_layer}, false));
        }
        if (anchor + 1 != local_layers.end() && candidates.size() < config.max_extra_model_trials) {
            candidates.push_back(append({anchor_layer, *(anchor + 1)}, false));
        }
    }
    if (config.include_triplet_profile && local_layers.size() == 3 &&
            candidates.size() < config.max_extra_model_trials) {
        candidates.push_back(append(local_layers, false));
    }
    if (config.include_opposite_sign_control && candidates.size() < config.max_extra_model_trials) {
        candidates.push_back(append({anchor_layer}, true));
    }
    for (const auto & candidate : candidates) {
        if (!valid_zoom_candidate(candidate, error)) return false;
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
    // Start every retained region at Bootstrap. Depth expresses capacity, not
    // permission to skip the rank-one/rank-two evidence chain.
    plan.depth = evidence_depth.depth;
    plan.phase = common_flydelta_experiment_phase::bootstrap;
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
