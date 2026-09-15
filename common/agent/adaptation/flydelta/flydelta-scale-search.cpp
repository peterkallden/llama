#include "agent/adaptation/flydelta/flydelta-scale-search.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

bool finite_geometry(const common_flydelta_scale_geometry & geometry) {
    return std::isfinite(geometry.cosine) && geometry.cosine >= -1.0f &&
        geometry.cosine <= 1.0f && std::isfinite(geometry.progress) &&
        std::isfinite(geometry.leakage) && geometry.leakage >= 0.0f &&
        std::isfinite(geometry.shift_norm) && geometry.shift_norm >= 0.0f;
}

bool finite_scale(float scale) {
    return std::isfinite(scale) && scale > 0.0f && scale <= 1.0f;
}

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

bool better_trial(const common_flydelta_scale_trial & trial,
        size_t index, const common_flydelta_scale_trial & best, size_t best_index) {
    if (trial.quality_delta != best.quality_delta) {
        return trial.quality_delta > best.quality_delta;
    }
    if (trial.scale != best.scale) return trial.scale < best.scale;
    return index < best_index;
}

} // namespace

bool common_flydelta_scale_search_config_validate(
        const common_flydelta_scale_search_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || !finite_scale(config.initial_scale) ||
            !std::isfinite(config.growth_factor) || config.growth_factor <= 1.0f ||
            !finite_scale(config.max_scale) || config.initial_scale > config.max_scale ||
            config.max_geometric_trials == 0 || config.max_geometric_trials > 8 ||
            config.max_refinement_trials > 4 || !std::isfinite(config.min_cosine) ||
            config.min_cosine < -1.0f || config.min_cosine > 1.0f ||
            !std::isfinite(config.max_leakage) || config.max_leakage < 0.0f ||
            !std::isfinite(config.max_shift_norm) || config.max_shift_norm <= 0.0f ||
            !std::isfinite(config.saturation_epsilon) || config.saturation_epsilon < 0.0f) {
        error = "FlyDelta scale search configuration is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_scale_trial_validate(
        const common_flydelta_scale_trial & trial,
        std::string & error) {
    error.clear();
    if (!finite_scale(trial.scale) || !valid_outcome(trial.outcome) ||
            !std::isfinite(trial.quality_delta) || trial.quality_delta < -1.0f ||
            trial.quality_delta > 1.0f || (trial.verifier_known && trial.evidence_ref.empty())) {
        error = "FlyDelta scale trial is invalid";
        return false;
    }
    if (trial.geometry_available && (!trial.geometry.available ||
            !finite_geometry(trial.geometry))) {
        error = "FlyDelta scale trial geometry is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_run_scale_search(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_scale_search_config & config,
        const common_flydelta_scale_search_runner & runner,
        std::vector<common_flydelta_scale_trial> & trials,
        common_flydelta_scale_selection & selection,
        std::string & error) {
    error.clear();
    trials.clear();
    selection = {};
    if (!common_flydelta_experiment_fixture_validate(fixture, error) ||
            !common_flydelta_scale_search_config_validate(config, error) || !runner) {
        if (error.empty()) error = "FlyDelta scale search runner configuration is invalid";
        return false;
    }

    common_flydelta_counterfactual_trial baseline;
    common_flydelta_scale_geometry baseline_geometry;
    if (!runner(fixture, 0.0f, false, baseline, baseline_geometry, error) ||
            !common_flydelta_counterfactual_trial_validate(baseline, error)) {
        return false;
    }

    float previous_scale = 0.0f;
    common_flydelta_scale_geometry previous_geometry;
    bool previous_geometry_safe = true;
    size_t geometric_count = 0;
    size_t refinement_count = 0;
    size_t first_helped_index = std::numeric_limits<size_t>::max();
    while (geometric_count < config.max_geometric_trials) {
        const float scale = geometric_count == 0
            ? config.initial_scale : previous_scale * config.growth_factor;
        if (!finite_scale(scale) || scale > config.max_scale ||
                (geometric_count != 0 && scale <= previous_scale)) break;
        common_flydelta_counterfactual_trial candidate;
        common_flydelta_scale_geometry geometry;
        if (!runner(fixture, scale, true, candidate, geometry, error) ||
                !common_flydelta_counterfactual_trial_validate(candidate, error)) {
            return false;
        }
        common_flydelta_scale_trial trial;
        trial.scale = scale;
        trial.executed = candidate.executed;
        trial.verifier_known = baseline.verifier_known && candidate.verifier_known;
        trial.outcome = common_flydelta_classify_counterfactual(baseline, candidate);
        trial.quality_delta = candidate.quality - baseline.quality;
        trial.geometry = geometry;
        trial.geometry_available = geometry.available;
        trial.evidence_ref = candidate.evidence_ref;
        trial.safe_to_escalate = geometry.available && finite_geometry(geometry) &&
            geometry.cosine >= config.min_cosine && geometry.leakage <= config.max_leakage &&
            geometry.shift_norm <= config.max_shift_norm;
        if (trial.safe_to_escalate && previous_geometry_safe && previous_geometry.available &&
                geometry.progress <= previous_geometry.progress + config.saturation_epsilon &&
                geometry.shift_norm <= previous_geometry.shift_norm + config.saturation_epsilon) {
            trial.safe_to_escalate = false;
        }
        trials.push_back(trial);
        const size_t trial_index = trials.size() - 1;
        ++geometric_count;
        if (trial.outcome == common_flydelta_counterfactual_outcome::helped &&
                trial.executed && trial.verifier_known) {
            first_helped_index = trial_index;
            break;
        }
        previous_scale = scale;
        previous_geometry = geometry;
        previous_geometry_safe = trial.safe_to_escalate;
        if (!trial.safe_to_escalate) break;
    }

    if (first_helped_index != std::numeric_limits<size_t>::max()) {
        const float upper_scale = trials[first_helped_index].scale;
        const float lower_scale = first_helped_index == 0
            ? 0.0f : trials[first_helped_index - 1].scale;
        while (refinement_count < config.max_refinement_trials) {
            const float midpoint = lower_scale + (upper_scale - lower_scale) * 0.5f;
            if (!finite_scale(midpoint) || midpoint <= lower_scale || midpoint >= upper_scale) break;
            common_flydelta_counterfactual_trial candidate;
            common_flydelta_scale_geometry geometry;
            if (!runner(fixture, midpoint, true, candidate, geometry, error) ||
                    !common_flydelta_counterfactual_trial_validate(candidate, error)) {
                return false;
            }
            common_flydelta_scale_trial trial;
            trial.scale = midpoint;
            trial.executed = candidate.executed;
            trial.verifier_known = baseline.verifier_known && candidate.verifier_known;
            trial.outcome = common_flydelta_classify_counterfactual(baseline, candidate);
            trial.quality_delta = candidate.quality - baseline.quality;
            trial.geometry = geometry;
            trial.geometry_available = geometry.available;
            trial.refinement = true;
            trial.evidence_ref = candidate.evidence_ref;
            trial.safe_to_escalate = geometry.available && finite_geometry(geometry) &&
                geometry.cosine >= config.min_cosine && geometry.leakage <= config.max_leakage &&
                geometry.shift_norm <= config.max_shift_norm;
            trials.push_back(std::move(trial));
            ++refinement_count;
        }
    }

    float best_score = -std::numeric_limits<float>::infinity();
    size_t best_index = 0;
    bool found = false;
    for (size_t index = 0; index < trials.size(); ++index) {
        const auto & trial = trials[index];
        if (!common_flydelta_scale_trial_validate(trial, error)) return false;
        if (trial.outcome != common_flydelta_counterfactual_outcome::helped ||
                !trial.executed || !trial.verifier_known) continue;
        if (!found || better_trial(trial, index, trials[best_index], best_index)) {
            found = true;
            best_index = index;
            best_score = trial.quality_delta;
        }
    }
    if (found) {
        selection.selected = true;
        selection.scale = trials[best_index].scale;
        selection.score = best_score;
        selection.trial_index = best_index;
    }
    return true;
}
