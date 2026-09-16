#include "agent/adaptation/flydelta/flydelta-intervention-region-search.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

bool valid_outcome(common_flydelta_counterfactual_outcome outcome) {
    switch (outcome) {
        case common_flydelta_counterfactual_outcome::unknown:
        case common_flydelta_counterfactual_outcome::helped:
        case common_flydelta_counterfactual_outcome::neutral:
        case common_flydelta_counterfactual_outcome::harmed:
            return true;
    }
    return false;
}

float per_layer_scale(float total_scale, size_t count) {
    return total_scale / std::sqrt(static_cast<float>(count));
}

bool contains_layers(const std::vector<std::vector<uint32_t>> & values,
        const std::vector<uint32_t> & candidate) {
    return std::find(values.begin(), values.end(), candidate) != values.end();
}

bool candidate_validate(
        const common_flydelta_intervention_region_candidate & candidate,
        std::string & error) {
    error.clear();
    if (candidate.schema_version != 1 || candidate.layer_indices.empty() ||
            candidate.layer_indices.size() > 2 ||
            !std::is_sorted(candidate.layer_indices.begin(), candidate.layer_indices.end()) ||
            std::adjacent_find(candidate.layer_indices.begin(), candidate.layer_indices.end()) !=
                candidate.layer_indices.end() || candidate.anchor_layer_index == 0 ||
            std::find(candidate.layer_indices.begin(), candidate.layer_indices.end(),
                candidate.anchor_layer_index) == candidate.layer_indices.end() ||
            !std::isfinite(candidate.total_scale) || candidate.total_scale <= 0.0f ||
            candidate.total_scale > 1.0f || !std::isfinite(candidate.per_layer_scale) ||
            candidate.per_layer_scale <= 0.0f ||
            std::fabs(candidate.per_layer_scale - per_layer_scale(
                candidate.total_scale, candidate.layer_indices.size())) > 0.00001f) {
        error = "FlyDelta intervention region candidate is invalid";
        return false;
    }
    if (candidate.source == common_flydelta_layer_search_candidate_source::diagnostic_singleton &&
            candidate.layer_indices.size() != 1) {
        error = "FlyDelta intervention diagnostic candidate must be a singleton";
        return false;
    }
    if (candidate.source == common_flydelta_layer_search_candidate_source::neighborhood_expansion &&
            (candidate.layer_indices.size() != 2 ||
             candidate.layer_indices[1] != candidate.layer_indices[0] + 1)) {
        error = "FlyDelta intervention neighborhood must be adjacent";
        return false;
    }
    return true;
}

bool geometry_safe(const common_flydelta_representation_diagnostics & geometry,
        bool available, const common_flydelta_intervention_region_search_config & config) {
    return available && geometry.cosine >= config.min_cosine &&
        geometry.leakage <= config.max_leakage &&
        geometry.shift_norm <= config.max_shift_norm;
}

bool geometry_promising(const common_flydelta_representation_diagnostics & geometry,
        bool available, const common_flydelta_intervention_region_search_config & config) {
    return geometry_safe(geometry, available, config) && geometry.progress > 0.0f;
}

bool margin_promising(const common_flydelta_decision_margin & baseline,
        const common_flydelta_decision_margin & candidate) {
    return baseline.available && candidate.available &&
        candidate.normalized_delta() > baseline.normalized_delta();
}

} // namespace

bool common_flydelta_intervention_region_candidate_validate(
        const common_flydelta_intervention_region_candidate & candidate,
        std::string & error) {
    return candidate_validate(candidate, error);
}

bool common_flydelta_intervention_region_trial_validate(
        const common_flydelta_intervention_region_trial & trial,
        std::string & error) {
    error.clear();
    if (!candidate_validate(trial.candidate, error) ||
            !valid_outcome(trial.outcome) || !std::isfinite(trial.quality_delta) ||
            !common_flydelta_decision_margin_validate(trial.margin, error)) {
        if (error.empty()) error = "FlyDelta intervention region trial is invalid";
        return false;
    }
    if (trial.geometry_available &&
            !common_flydelta_representation_diagnostics_validate(trial.geometry, error)) {
        return false;
    }
    if (trial.evidence_ref.size() > 512) {
        error = "FlyDelta intervention region trial evidence reference is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_intervention_region_search_config_validate(
        const common_flydelta_intervention_region_search_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || config.available_layers.empty() ||
            config.available_layers.size() > 16 ||
            !std::is_sorted(config.available_layers.begin(), config.available_layers.end()) ||
            config.available_layers.front() == 0 ||
            std::adjacent_find(config.available_layers.begin(), config.available_layers.end()) !=
                config.available_layers.end() || config.scales.empty() ||
            config.scales.size() > 8 ||
            !std::is_sorted(config.scales.begin(), config.scales.end()) ||
            config.max_singleton_layers == 0 ||
            config.max_singleton_layers > config.available_layers.size() ||
            config.max_neighborhoods > 16 || config.max_trials == 0 ||
            config.max_trials > 64 || config.max_stalled_scales == 0 ||
            config.max_stalled_scales > config.scales.size() ||
            !std::isfinite(config.min_cosine) || config.min_cosine < -1.0f ||
            config.min_cosine > 1.0f || !std::isfinite(config.max_leakage) ||
            config.max_leakage < 0.0f || !std::isfinite(config.max_shift_norm) ||
            config.max_shift_norm <= 0.0f) {
        error = "FlyDelta intervention region search configuration is invalid";
        return false;
    }
    for (size_t index = 0; index < config.scales.size(); ++index) {
        if (!std::isfinite(config.scales[index]) || config.scales[index] <= 0.0f ||
                config.scales[index] > 1.0f ||
                (index > 0 && config.scales[index] <= config.scales[index - 1])) {
            error = "FlyDelta intervention region scales are invalid";
            return false;
        }
    }
    return true;
}

bool common_flydelta_run_intervention_region_search(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_intervention_region_search_config & config,
        const common_flydelta_intervention_region_search_runner & runner,
        std::vector<common_flydelta_intervention_region_trial> & trials,
        common_flydelta_intervention_region_selection & selection,
        std::string & error) {
    error.clear();
    trials.clear();
    selection = {};
    if (!common_flydelta_experiment_fixture_validate(fixture, error) ||
            !common_flydelta_intervention_region_search_config_validate(config, error) ||
            !runner) {
        if (error.empty()) error = "FlyDelta intervention region runner is invalid";
        return false;
    }

    common_flydelta_counterfactual_trial baseline;
    common_flydelta_decision_margin baseline_margin;
    common_flydelta_representation_diagnostics baseline_geometry;
    bool baseline_geometry_available = false;
    if (!runner(fixture, nullptr, false, baseline, baseline_margin, baseline_geometry,
                baseline_geometry_available, error) ||
            !common_flydelta_counterfactual_trial_validate(baseline, error) ||
            !common_flydelta_decision_margin_validate(baseline_margin, error)) {
        return false;
    }

    std::vector<uint32_t> active_layers;
    std::vector<std::vector<uint32_t>> pair_layers;
    bool singleton_helped = false;
    auto evaluate = [&](const common_flydelta_intervention_region_candidate & candidate,
            bool apply_overlay, common_flydelta_intervention_region_trial & region_trial) {
        common_flydelta_counterfactual_trial counterfactual;
        common_flydelta_decision_margin margin;
        common_flydelta_representation_diagnostics geometry;
        bool geometry_available = false;
        if (!candidate_validate(candidate, error) ||
                !runner(fixture, &candidate, apply_overlay, counterfactual, margin, geometry,
                    geometry_available, error) ||
                !common_flydelta_counterfactual_trial_validate(counterfactual, error) ||
                !common_flydelta_decision_margin_validate(margin, error)) {
            return false;
        }
        if (geometry_available &&
                !common_flydelta_representation_diagnostics_validate(geometry, error)) {
            return false;
        }
        region_trial = {};
        region_trial.candidate = candidate;
        region_trial.outcome = common_flydelta_classify_counterfactual(baseline, counterfactual);
        region_trial.quality_delta = counterfactual.quality - baseline.quality;
        region_trial.margin = margin;
        region_trial.executed = counterfactual.executed;
        region_trial.verifier_known = baseline.verifier_known && counterfactual.verifier_known;
        region_trial.geometry_available = geometry_available;
        region_trial.geometry = geometry;
        region_trial.evidence_ref = counterfactual.evidence_ref;
        region_trial.safe_to_continue = !geometry_available ||
            geometry_safe(geometry, geometry_available, config);
        region_trial.promising = region_trial.outcome ==
            common_flydelta_counterfactual_outcome::helped ||
            geometry_promising(geometry, geometry_available, config) ||
            margin_promising(baseline_margin, margin);
        return true;
    };

    for (size_t layer_index = 0;
            layer_index < config.max_singleton_layers && trials.size() < config.max_trials;
            ++layer_index) {
        const uint32_t layer = config.available_layers[layer_index];
        size_t stalled = 0;
        for (const float scale : config.scales) {
            if (trials.size() >= config.max_trials) break;
            common_flydelta_intervention_region_candidate candidate;
            candidate.layer_indices = {layer};
            candidate.anchor_layer_index = layer;
            candidate.total_scale = scale;
            candidate.per_layer_scale = scale;
            candidate.source = common_flydelta_layer_search_candidate_source::diagnostic_singleton;
            common_flydelta_intervention_region_trial trial;
            if (!evaluate(candidate, true, trial)) return false;
            trials.push_back(std::move(trial));
            const auto & stored = trials.back();
            if (stored.promising) {
                if (std::find(active_layers.begin(), active_layers.end(), layer) ==
                        active_layers.end()) active_layers.push_back(layer);
                stalled = 0;
            } else if (stored.geometry_available) {
                ++stalled;
            }
            if (stored.outcome == common_flydelta_counterfactual_outcome::helped &&
                    stored.executed && stored.verifier_known) {
                singleton_helped = true;
            }
            if (stored.geometry_available && !stored.safe_to_continue) break;
            if (stalled >= config.max_stalled_scales) break;
        }
    }

    std::sort(active_layers.begin(), active_layers.end());
    if (!singleton_helped && config.max_neighborhoods > 0) {
        for (const uint32_t layer : active_layers) {
            for (const uint32_t neighbor : {layer > 1 ? layer - 1 : 0, layer + 1}) {
                if (neighbor == 0 ||
                        !std::binary_search(config.available_layers.begin(),
                            config.available_layers.end(), neighbor)) continue;
                std::vector<uint32_t> pair = {layer, neighbor};
                std::sort(pair.begin(), pair.end());
                if (contains_layers(pair_layers, pair) ||
                        pair_layers.size() >= config.max_neighborhoods) continue;
                pair_layers.push_back(pair);
                for (const float scale : config.scales) {
                    if (trials.size() >= config.max_trials) break;
                    common_flydelta_intervention_region_candidate candidate;
                    candidate.layer_indices = pair;
                    candidate.anchor_layer_index = layer;
                    candidate.total_scale = scale;
                    candidate.per_layer_scale = per_layer_scale(scale, pair.size());
                    candidate.source = common_flydelta_layer_search_candidate_source::neighborhood_expansion;
                    common_flydelta_intervention_region_trial trial;
                    if (!evaluate(candidate, true, trial)) return false;
                    trials.push_back(std::move(trial));
                }
            }
        }
    }

    float best_score = -std::numeric_limits<float>::infinity();
    for (size_t index = 0; index < trials.size(); ++index) {
        const auto & trial = trials[index];
        if (trial.outcome != common_flydelta_counterfactual_outcome::helped ||
                !trial.executed || !trial.verifier_known) continue;
        if (!selection.selected || trial.quality_delta > best_score ||
                (trial.quality_delta == best_score &&
                 trial.candidate.total_scale < trials[selection.trial_index].candidate.total_scale)) {
            selection.selected = true;
            selection.trial_index = index;
            selection.score = trial.quality_delta;
            best_score = trial.quality_delta;
        }
    }
    return true;
}
