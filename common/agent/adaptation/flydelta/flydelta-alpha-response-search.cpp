#include "agent/adaptation/flydelta/flydelta-alpha-response-search.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

namespace {

bool finite_geometry(const common_flydelta_representation_diagnostics & geometry) {
    return std::isfinite(geometry.cosine) && std::isfinite(geometry.progress) &&
        std::isfinite(geometry.leakage) && geometry.leakage >= 0.0f &&
        std::isfinite(geometry.shift_norm) && geometry.shift_norm >= 0.0f;
}

bool finite_scale(float scale) {
    return std::isfinite(scale) && scale > 0.0f && scale <= 1.0f;
}

float margin_delta(const common_flydelta_decision_margin & baseline,
        const common_flydelta_decision_margin & candidate, bool normalized) {
    if (!baseline.available || !candidate.available) return 0.0f;
    return normalized
        ? candidate.normalized_delta() - baseline.normalized_delta()
        : candidate.total_delta() - baseline.total_delta();
}

float utility_for(const common_flydelta_alpha_response_trial & trial,
        bool margin_available, float leakage_penalty) {
    if (margin_available) {
        return trial.margin_delta_normalized - leakage_penalty *
            (trial.geometry_available ? trial.geometry.leakage : 0.0f);
    }
    if (!trial.geometry_available) return 0.0f;
    return trial.geometry.progress * std::max(0.0f, trial.geometry.cosine) -
        leakage_penalty * trial.geometry.leakage;
}

bool better(const common_flydelta_alpha_response_trial & lhs,
        const common_flydelta_alpha_response_trial & rhs, float epsilon) {
    if (lhs.utility > rhs.utility + epsilon) return true;
    if (rhs.utility > lhs.utility + epsilon) return false;
    return lhs.scale < rhs.scale;
}

} // namespace

const char * common_flydelta_alpha_response_status_name(
        common_flydelta_alpha_response_status status) {
    switch (status) {
        case common_flydelta_alpha_response_status::inconclusive: return "inconclusive";
        case common_flydelta_alpha_response_status::helped: return "helped";
        case common_flydelta_alpha_response_status::saturated: return "saturated";
        case common_flydelta_alpha_response_status::safety_limited: return "safety_limited";
        case common_flydelta_alpha_response_status::budget_limited: return "budget_limited";
        case common_flydelta_alpha_response_status::upper_bound_reached:
            return "upper_bound_reached";
    }
    return "inconclusive";
}

bool common_flydelta_alpha_response_search_config_validate(
        const common_flydelta_alpha_response_search_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || !finite_scale(config.seed_scale) ||
            !finite_scale(config.max_scale) || config.seed_scale > config.max_scale ||
            !std::isfinite(config.growth_factor) || config.growth_factor <= 1.0f ||
            config.max_expansion_trials == 0 || config.max_expansion_trials > 16 ||
            config.max_zoom_trials > 16 || config.max_min_effective_trials > 8 ||
            config.max_expansion_non_improving == 0 ||
            config.max_expansion_non_improving > 8 ||
            !std::isfinite(config.utility_epsilon) || config.utility_epsilon < 0.0f ||
            !std::isfinite(config.max_leakage) || config.max_leakage < 0.0f ||
            !std::isfinite(config.max_shift_norm) || config.max_shift_norm <= 0.0f ||
            !std::isfinite(config.min_cosine) || config.min_cosine < -1.0f ||
            config.min_cosine > 1.0f || !std::isfinite(config.leakage_penalty) ||
            config.leakage_penalty < 0.0f) {
        error = "FlyDelta alpha response search configuration is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_alpha_response_trial_validate(
        const common_flydelta_alpha_response_trial & trial,
        std::string & error) {
    error.clear();
    if (!finite_scale(trial.scale) ||
            !common_flydelta_counterfactual_trial_validate(trial.counterfactual, error) ||
            !common_flydelta_decision_margin_validate(trial.margin, error) ||
            !std::isfinite(trial.margin_delta_total) ||
            !std::isfinite(trial.margin_delta_normalized) ||
            !std::isfinite(trial.utility)) {
        if (error.empty()) error = "FlyDelta alpha response trial is invalid";
        return false;
    }
    if (trial.geometry_available && !finite_geometry(trial.geometry)) {
        error = "FlyDelta alpha response geometry is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_run_alpha_response_search(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_alpha_response_search_config & config,
        const common_flydelta_alpha_response_runner & runner,
        std::vector<common_flydelta_alpha_response_trial> & trials,
        common_flydelta_alpha_response_selection & selection,
        std::string & error) {
    error.clear();
    trials.clear();
    selection = {};
    if (!common_flydelta_experiment_fixture_validate(fixture, error) ||
            !common_flydelta_alpha_response_search_config_validate(config, error) || !runner) {
        if (error.empty()) error = "FlyDelta alpha response runner is invalid";
        return false;
    }

    common_flydelta_counterfactual_trial baseline_counterfactual;
    common_flydelta_decision_margin baseline_margin;
    common_flydelta_representation_diagnostics baseline_geometry;
    bool baseline_geometry_available = false;
    if (!runner(fixture, 0.0f, false, baseline_counterfactual, baseline_margin,
            baseline_geometry, baseline_geometry_available, error) ||
            !common_flydelta_counterfactual_trial_validate(baseline_counterfactual, error) ||
            !common_flydelta_decision_margin_validate(baseline_margin, error)) return false;

    std::map<float, size_t> by_scale;
    auto evaluate = [&](float scale, bool refinement, size_t & index) -> bool {
        if (!finite_scale(scale) || scale > config.max_scale) return false;
        const auto found = by_scale.find(scale);
        if (found != by_scale.end()) {
            index = found->second;
            return true;
        }
        common_flydelta_alpha_response_trial value;
        value.scale = scale;
        value.refinement = refinement;
        bool geometry_available = false;
        if (!runner(fixture, scale, true, value.counterfactual, value.margin,
                value.geometry, geometry_available, error)) {
            if (error.empty()) error = "FlyDelta alpha response runner failed";
            return false;
        }
        if (!common_flydelta_counterfactual_trial_validate(value.counterfactual, error) ||
                !common_flydelta_decision_margin_validate(value.margin, error)) return false;
        value.geometry_available = geometry_available;
        value.outcome = common_flydelta_classify_counterfactual(
            baseline_counterfactual, value.counterfactual);
        value.margin_available = baseline_margin.available && value.margin.available;
        value.margin_delta_total = margin_delta(baseline_margin, value.margin, false);
        value.margin_delta_normalized = margin_delta(baseline_margin, value.margin, true);
        value.safe_to_continue = !geometry_available || (
            finite_geometry(value.geometry) &&
            value.geometry.cosine >= config.min_cosine &&
            value.geometry.leakage <= config.max_leakage &&
            value.geometry.shift_norm <= config.max_shift_norm);
        value.utility = utility_for(value, value.margin_available, config.leakage_penalty);
        if (!common_flydelta_alpha_response_trial_validate(value, error)) return false;
        index = trials.size();
        trials.push_back(std::move(value));
        by_scale.emplace(scale, index);
        return true;
    };

    size_t index = 0;
    float scale = config.seed_scale;
    float max_reachable_scale = config.seed_scale;
    for (size_t count = 1; count < config.max_expansion_trials; ++count) {
        if (max_reachable_scale > config.max_scale / config.growth_factor) {
            max_reachable_scale = config.max_scale;
            break;
        }
        max_reachable_scale *= config.growth_factor;
    }
    max_reachable_scale = std::min(max_reachable_scale, config.max_scale);
    float best_expansion_utility = -std::numeric_limits<float>::infinity();
    size_t non_improving_expansions = 0;
    bool helped_during_expansion = false;
    bool safety_limited = false;
    bool saturated = false;
    bool upper_bound_reached = false;
    std::vector<size_t> expansion_indices;
    for (size_t count = 0; count < config.max_expansion_trials; ++count) {
        if (!evaluate(scale, false, index)) return false;
        expansion_indices.push_back(index);
        const auto & current = trials[index];
        if (current.outcome == common_flydelta_counterfactual_outcome::helped) {
            helped_during_expansion = true;
            break;
        }
        if (!current.safe_to_continue) {
            safety_limited = true;
            break;
        }
        if (current.utility > best_expansion_utility + config.utility_epsilon) {
            best_expansion_utility = current.utility;
            non_improving_expansions = 0;
        } else if (++non_improving_expansions >= config.max_expansion_non_improving) {
            saturated = true;
            break;
        }
        if (scale > config.max_scale / config.growth_factor) {
            upper_bound_reached = true;
            break;
        }
        scale *= config.growth_factor;
        if (!finite_scale(scale) || scale > config.max_scale) {
            upper_bound_reached = true;
            break;
        }
    }

    const bool budget_limited = !helped_during_expansion && !safety_limited &&
        !saturated && !upper_bound_reached &&
        expansion_indices.size() >= config.max_expansion_trials;
    selection.max_reachable_scale = max_reachable_scale;
    if (!expansion_indices.empty()) {
        const auto & last = trials[expansion_indices.back()];
        selection.last_scale = last.scale;
        selection.last_utility = last.utility;
        if (expansion_indices.size() >= 2) {
            const auto & previous = trials[expansion_indices[expansion_indices.size() - 2]];
            const float scale_delta = last.scale - previous.scale;
            if (scale_delta > 0.0f) {
                selection.utility_slope = (last.utility - previous.utility) / scale_delta;
            }
            selection.range_not_exhausted = budget_limited && last.safe_to_continue &&
                last.utility > previous.utility + config.utility_epsilon;
        }
    }

    // Golden-section refinement is used only inside the observed response
    // interval. It is intentionally conservative: it never extrapolates
    // beyond the geometric expansion and never turns a diagnostic into a
    // host verdict.
    if (!helped_during_expansion && config.max_zoom_trials > 0 && trials.size() >= 2) {
        std::vector<size_t> order(trials.size());
        for (size_t i = 0; i < trials.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
            return trials[lhs].scale < trials[rhs].scale;
        });
        size_t best_position = 0;
        for (size_t position = 1; position < order.size(); ++position) {
            if (better(trials[order[position]], trials[order[best_position]],
                    config.utility_epsilon)) {
                best_position = position;
            }
        }
        float left = best_position == 0 ? 0.0f : trials[order[best_position - 1]].scale;
        float right = best_position + 1 >= order.size()
            ? trials[order.back()].scale : trials[order[best_position + 1]].scale;
        constexpr float phi = 1.61803398875f;
        for (size_t count = 0; count < config.max_zoom_trials && right - left > 0.000001f; ++count) {
            const float x1 = right - (right - left) / phi;
            const float x2 = left + (right - left) / phi;
            size_t i1 = 0, i2 = 0;
            if (!evaluate(x1, true, i1) || !evaluate(x2, true, i2)) return false;
            if (trials[i1].outcome == common_flydelta_counterfactual_outcome::helped ||
                    trials[i2].outcome == common_flydelta_counterfactual_outcome::helped) break;
            if (better(trials[i1], trials[i2], config.utility_epsilon)) right = x2;
            else left = x1;
        }
    }

    for (size_t i = 0; i < trials.size(); ++i) {
        const auto & trial = trials[i];
        const bool eligible = trial.safe_to_continue &&
            trial.outcome != common_flydelta_counterfactual_outcome::harmed &&
            trial.utility > config.utility_epsilon;
        if (eligible && (!selection.selected ||
                better(trial, trials[selection.trial_index], config.utility_epsilon))) {
            selection.selected = true;
            selection.trial_index = i;
            selection.scale = trial.scale;
            selection.utility = trial.utility;
            selection.best_margin_delta_total = trial.margin_delta_total;
            selection.best_margin_delta_normalized = trial.margin_delta_normalized;
            selection.best_margin_available = trial.margin_available;
        }
    }

    size_t first_helped = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < trials.size(); ++i) {
        if (trials[i].outcome == common_flydelta_counterfactual_outcome::helped &&
                trials[i].counterfactual.executed && trials[i].counterfactual.verifier_known &&
                trials[i].safe_to_continue &&
                (first_helped == std::numeric_limits<size_t>::max() ||
                 trials[i].scale < trials[first_helped].scale)) first_helped = i;
    }
    if (first_helped != std::numeric_limits<size_t>::max()) {
        selection.response_status = common_flydelta_alpha_response_status::helped;
        selection.minimum_effective_available = true;
        selection.minimum_effective_scale = trials[first_helped].scale;
        selection.minimum_effective_trial_index = first_helped;
        // A small bounded midpoint search finds the minimum verified scale
        // without assuming that margin utility itself is monotone.
        float lower = 0.0f;
        for (const auto & trial : trials) {
            if (trial.scale < selection.minimum_effective_scale &&
                    trial.outcome != common_flydelta_counterfactual_outcome::helped) {
                lower = std::max(lower, trial.scale);
            }
        }
        for (size_t count = 0; count < config.max_min_effective_trials; ++count) {
            const float midpoint = lower +
                (selection.minimum_effective_scale - lower) * 0.5f;
            if (!finite_scale(midpoint) || midpoint <= lower ||
                    midpoint >= selection.minimum_effective_scale) break;
            size_t midpoint_index = 0;
            if (!evaluate(midpoint, true, midpoint_index)) return false;
            if (trials[midpoint_index].outcome == common_flydelta_counterfactual_outcome::helped &&
                    trials[midpoint_index].counterfactual.verifier_known) {
                selection.minimum_effective_scale = midpoint;
                selection.minimum_effective_trial_index = midpoint_index;
            } else {
                lower = midpoint;
            }
        }
    } else if (safety_limited) {
        selection.response_status = common_flydelta_alpha_response_status::safety_limited;
    } else if (saturated) {
        selection.response_status = common_flydelta_alpha_response_status::saturated;
    } else if (budget_limited) {
        selection.response_status = common_flydelta_alpha_response_status::budget_limited;
    } else if (upper_bound_reached) {
        selection.response_status = common_flydelta_alpha_response_status::upper_bound_reached;
    }
    return true;
}
