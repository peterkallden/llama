#include "agent/adaptation/flydelta/flydelta-experiment-orchestration.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace {
bool finite_value(float value) { return std::isfinite(value); }

float l2_norm(const std::vector<float> & values) {
    float total = 0.0f;
    for (const float value : values) total += value * value;
    return std::sqrt(total);
}

bool valid_zoom_phase(common_flydelta_bootstrap_zoom_phase phase) {
    return phase == common_flydelta_bootstrap_zoom_phase::alpha_zoom ||
        phase == common_flydelta_bootstrap_zoom_phase::profile_zoom ||
        phase == common_flydelta_bootstrap_zoom_phase::sign_control ||
        phase == common_flydelta_bootstrap_zoom_phase::adaptive_alpha;
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
                candidate.layer_indices.end() || !finite_value(candidate.total_scale) ||
            candidate.total_scale <= 0.0f || candidate.total_scale > 1.0f) {
        error = "FlyDelta BootstrapZoom candidate is invalid";
        return false;
    }
    for (const float weight : candidate.layer_weights) {
        if (!finite_value(weight)) {
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
}

const char * common_flydelta_bootstrap_zoom_phase_name(
        common_flydelta_bootstrap_zoom_phase phase) {
    switch (phase) {
        case common_flydelta_bootstrap_zoom_phase::alpha_zoom: return "alpha_zoom";
        case common_flydelta_bootstrap_zoom_phase::profile_zoom: return "profile_zoom";
        case common_flydelta_bootstrap_zoom_phase::sign_control: return "sign_control";
        case common_flydelta_bootstrap_zoom_phase::adaptive_alpha: return "adaptive_alpha";
    }
    return "alpha_zoom";
}

bool common_flydelta_bootstrap_zoom_config_validate(
        const common_flydelta_bootstrap_zoom_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || config.max_extra_model_trials == 0 ||
            config.max_extra_model_trials > 10 || config.alpha_multipliers.empty() ||
            config.alpha_multipliers.size() > 4 || !finite_value(config.min_margin_improvement) ||
            config.min_margin_improvement < 0.0f) {
        error = "FlyDelta BootstrapZoom configuration is invalid";
        return false;
    }
    for (size_t index = 0; index < config.alpha_multipliers.size(); ++index) {
        const float multiplier = config.alpha_multipliers[index];
        if (!finite_value(multiplier) || multiplier <= 0.0f || multiplier > 4.0f ||
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
    if (!valid_zoom_candidate(trial.candidate, error) ||
            (!trial.host_evaluated && !trial.diagnostics_available) ||
            (trial.verifier_known && !trial.host_evaluated) ||
            !finite_value(trial.margin_delta) ||
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
    if (!finite_value(selection.search_score) ||
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
            !finite_value(state.selected_scale) || state.selected_scale <= 0.0f ||
            state.selected_scale > 1.0f || !finite_value(state.best_margin_delta) ||
            !finite_value(state.best_search_score) || state.extra_model_trials > 10 ||
            state.next_candidate_index > state.extra_model_trials || state.surface_revision == 0 ||
            state.parent_surface_revision >= state.surface_revision || state.search_rank == 0 ||
            state.search_rank > 4 || !finite_value(state.evidence_rank) ||
            state.evidence_rank <= 0.0f || state.evidence_rank > 1024.0f ||
            state.surface_origin.empty() || state.surface_origin.size() > 128 ||
            state.parent_surface_ref.size() > 512 ||
            state.best_experimental_candidate_ref.size() > 512 || state.progress_iteration > 64 ||
            state.local_layers.size() > 3 || state.completed_trials.size() > 8 ||
            state.surface_trials.size() > 8 ||
            (state.phase == common_flydelta_bootstrap_zoom_phase::adaptive_alpha &&
                state.refinement_kind != common_flydelta_bootstrap_refinement_kind::adaptive_alpha) ||
            (state.alpha_response_available &&
                (!finite_value(state.alpha_response.scale) ||
                 !finite_value(state.alpha_response.utility) ||
                 !finite_value(state.alpha_response.last_scale) ||
                 !finite_value(state.alpha_response.last_utility) ||
                 !finite_value(state.alpha_response.utility_slope) ||
                 !finite_value(state.alpha_response.max_reachable_scale) ||
                 !finite_value(state.alpha_response.minimum_effective_scale) ||
                 !finite_value(state.alpha_response.best_margin_delta_total) ||
                 !finite_value(state.alpha_response.best_margin_delta_normalized))) ||
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
    for (const auto & trial : state.surface_trials) {
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
    if (anchor_layer == 0 || !finite_value(base_scale) || base_scale <= 0.0f ||
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
            !finite_value(selected_scale) || selected_scale <= 0.0f || selected_scale > 1.0f ||
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
            (sign_control ? -1.0f : 1.0f) /
                std::sqrt(static_cast<float>(candidate.layer_indices.size())));
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
