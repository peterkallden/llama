#include "agent/adaptation/flydelta/flydelta-layer-search.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

struct scored_layer {
    common_flydelta_layer_diagnostic diagnostic;
    float score = 0.0f;
};

bool contains_layer(const std::vector<uint32_t> & layers, uint32_t layer) {
    return std::find(layers.begin(), layers.end(), layer) != layers.end();
}

bool candidate_has_layers(
        const std::vector<common_flydelta_layer_candidate> & candidates,
        const std::vector<uint32_t> & layers) {
    return std::any_of(candidates.begin(), candidates.end(),
        [&](const common_flydelta_layer_candidate & candidate) {
            return candidate.layer_indices == layers;
        });
}

float per_layer_scale(float total_scale, size_t layer_count) {
    return total_scale / std::sqrt(static_cast<float>(layer_count));
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

bool valid_source(common_flydelta_layer_search_candidate_source source) {
    return source == common_flydelta_layer_search_candidate_source::diagnostic_singleton ||
        source == common_flydelta_layer_search_candidate_source::neighborhood_expansion;
}

} // namespace

const char * common_flydelta_layer_search_candidate_source_name(
        common_flydelta_layer_search_candidate_source source) {
    switch (source) {
        case common_flydelta_layer_search_candidate_source::diagnostic_singleton:
            return "diagnostic_singleton";
        case common_flydelta_layer_search_candidate_source::neighborhood_expansion:
            return "neighborhood_expansion";
    }
    return "unknown";
}

bool common_flydelta_layer_search_config_validate(
        const common_flydelta_layer_search_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || config.max_regions == 0 ||
            config.max_regions > 8 || config.max_singletons == 0 ||
            config.max_singletons > 16 || config.max_neighborhoods > 16 ||
            config.max_candidates == 0 || config.max_candidates > 32 ||
            config.max_singletons > config.max_candidates ||
            config.min_region_separation == 0 ||
            !std::isfinite(config.min_cosine) || config.min_cosine < -1.0f ||
            config.min_cosine > 1.0f || !std::isfinite(config.total_scale) ||
            config.total_scale <= 0.0f || config.total_scale > 1.0f) {
        error = "FlyDelta layer search configuration is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_layer_candidate_validate(
        const common_flydelta_layer_candidate & candidate,
        std::string & error) {
    error.clear();
    if (candidate.schema_version != 1 || candidate.layer_indices.empty() ||
            candidate.layer_indices.size() > 2 ||
            !std::is_sorted(candidate.layer_indices.begin(), candidate.layer_indices.end()) ||
            std::adjacent_find(candidate.layer_indices.begin(), candidate.layer_indices.end()) !=
                candidate.layer_indices.end() ||
            candidate.anchor_layer_index == 0 ||
            !contains_layer(candidate.layer_indices, candidate.anchor_layer_index) ||
            !std::isfinite(candidate.diagnostic_score) || candidate.diagnostic_score < 0.0f ||
            !std::isfinite(candidate.total_scale) || candidate.total_scale <= 0.0f ||
            candidate.total_scale > 1.0f || !valid_source(candidate.source) ||
            !std::isfinite(candidate.per_layer_scale) ||
            candidate.per_layer_scale <= 0.0f ||
            std::fabs(candidate.per_layer_scale - per_layer_scale(
                candidate.total_scale, candidate.layer_indices.size())) > 0.00001f) {
        error = "FlyDelta layer candidate identity or bounds are invalid";
        return false;
    }
    if (candidate.source == common_flydelta_layer_search_candidate_source::diagnostic_singleton &&
            candidate.layer_indices.size() != 1) {
        error = "FlyDelta diagnostic layer candidate must be a singleton";
        return false;
    }
    if (candidate.source == common_flydelta_layer_search_candidate_source::neighborhood_expansion &&
            (candidate.layer_indices.size() != 2 ||
             candidate.layer_indices[1] != candidate.layer_indices[0] + 1)) {
        error = "FlyDelta neighborhood candidate must contain adjacent layers";
        return false;
    }
    return true;
}

bool common_flydelta_layer_search_plan_validate(
        const common_flydelta_layer_search_plan & plan,
        const common_flydelta_layer_search_config & config,
        std::string & error) {
    error.clear();
    if (!common_flydelta_layer_search_config_validate(config, error) ||
            plan.schema_version != 1 ||
            plan.singleton_candidates.size() > config.max_singletons ||
            plan.neighborhood_candidates.size() > config.max_neighborhoods ||
            plan.singleton_candidates.size() + plan.neighborhood_candidates.size() >
                config.max_candidates) {
        if (error.empty()) error = "FlyDelta layer search plan is invalid";
        return false;
    }
    std::vector<std::vector<uint32_t>> seen;
    for (const auto & candidate : plan.singleton_candidates) {
        if (!common_flydelta_layer_candidate_validate(candidate, error) ||
                candidate.source != common_flydelta_layer_search_candidate_source::diagnostic_singleton ||
                std::find(seen.begin(), seen.end(), candidate.layer_indices) != seen.end()) {
            if (error.empty()) error = "FlyDelta layer search plan contains duplicate singletons";
            return false;
        }
        seen.push_back(candidate.layer_indices);
    }
    for (const auto & candidate : plan.neighborhood_candidates) {
        if (!common_flydelta_layer_candidate_validate(candidate, error) ||
                candidate.source != common_flydelta_layer_search_candidate_source::neighborhood_expansion ||
                std::find(seen.begin(), seen.end(), candidate.layer_indices) != seen.end()) {
            if (error.empty()) error = "FlyDelta layer search plan contains an invalid or duplicate neighborhood";
            return false;
        }
        seen.push_back(candidate.layer_indices);
    }
    return true;
}

bool common_flydelta_build_layer_search_plan(
        const std::vector<common_flydelta_layer_diagnostic> & diagnostics,
        const std::vector<uint32_t> & available_layers,
        const common_flydelta_layer_search_config & config,
        common_flydelta_layer_search_plan & plan,
        std::string & error) {
    error.clear();
    plan = {};
    if (!common_flydelta_layer_search_config_validate(config, error) ||
            available_layers.empty() ||
            !std::is_sorted(available_layers.begin(), available_layers.end()) ||
            std::adjacent_find(available_layers.begin(), available_layers.end()) !=
                available_layers.end() || available_layers.front() == 0) {
        if (error.empty()) error = "FlyDelta layer search availability is invalid";
        return false;
    }

    std::vector<common_flydelta_layer_diagnostic> ordered = diagnostics;
    std::sort(ordered.begin(), ordered.end(),
        [](const auto & left, const auto & right) {
            return left.layer_index < right.layer_index;
        });
    if (std::adjacent_find(ordered.begin(), ordered.end(),
            [](const auto & left, const auto & right) {
                return left.layer_index == right.layer_index;
            }) != ordered.end()) {
        error = "FlyDelta layer diagnostics contain duplicate layers";
        return false;
    }
    std::vector<scored_layer> eligible;
    for (const auto & diagnostic : ordered) {
        if (!contains_layer(available_layers, diagnostic.layer_index) ||
                !std::isfinite(diagnostic.cosine) || diagnostic.cosine < -1.0f ||
                diagnostic.cosine > 1.0f || !std::isfinite(diagnostic.progress) ||
                !std::isfinite(diagnostic.leakage) || diagnostic.leakage < 0.0f ||
                !std::isfinite(diagnostic.shift_norm) || diagnostic.shift_norm < 0.0f) {
            error = "FlyDelta layer diagnostic is invalid or unavailable";
            return false;
        }
        if (diagnostic.cosine < config.min_cosine || diagnostic.progress <= 0.0f) continue;
        const float score = diagnostic.progress / (1.0f + diagnostic.leakage);
        if (std::isfinite(score) && score > 0.0f) eligible.push_back({diagnostic, score});
    }
    if (eligible.empty()) return true;

    std::vector<scored_layer> maxima;
    for (size_t i = 0; i < eligible.size(); ++i) {
        bool has_better_neighbor = false;
        if (i > 0 && eligible[i - 1].diagnostic.layer_index + 1 ==
                eligible[i].diagnostic.layer_index &&
                (eligible[i - 1].score > eligible[i].score ||
                (eligible[i - 1].score == eligible[i].score &&
                 eligible[i - 1].diagnostic.layer_index < eligible[i].diagnostic.layer_index))) {
            has_better_neighbor = true;
        }
        if (i + 1 < eligible.size() && eligible[i].diagnostic.layer_index + 1 ==
                eligible[i + 1].diagnostic.layer_index &&
                (eligible[i + 1].score > eligible[i].score ||
                (eligible[i + 1].score == eligible[i].score &&
                 eligible[i + 1].diagnostic.layer_index < eligible[i].diagnostic.layer_index))) {
            has_better_neighbor = true;
        }
        if (!has_better_neighbor) maxima.push_back(eligible[i]);
    }
    std::sort(maxima.begin(), maxima.end(),
        [](const auto & left, const auto & right) {
            if (left.score != right.score) return left.score > right.score;
            return left.diagnostic.layer_index < right.diagnostic.layer_index;
        });

    std::vector<scored_layer> anchors;
    for (const auto & maximum : maxima) {
        const bool too_close = std::any_of(anchors.begin(), anchors.end(),
            [&](const auto & anchor) {
                const uint32_t left = anchor.diagnostic.layer_index;
                const uint32_t right = maximum.diagnostic.layer_index;
                const uint32_t distance = left > right ? left - right : right - left;
                return distance < config.min_region_separation;
            });
        if (!too_close) anchors.push_back(maximum);
        if (anchors.size() >= config.max_regions) break;
    }

    for (const auto & anchor : anchors) {
        if (plan.singleton_candidates.size() >= config.max_singletons ||
                plan.singleton_candidates.size() + plan.neighborhood_candidates.size() >=
                    config.max_candidates) break;
        common_flydelta_layer_candidate candidate;
        candidate.layer_indices = {anchor.diagnostic.layer_index};
        candidate.anchor_layer_index = anchor.diagnostic.layer_index;
        candidate.diagnostic_score = anchor.score;
        candidate.total_scale = config.total_scale;
        candidate.per_layer_scale = config.total_scale;
        candidate.source = common_flydelta_layer_search_candidate_source::diagnostic_singleton;
        plan.singleton_candidates.push_back(std::move(candidate));
    }

    for (const auto & anchor : anchors) {
        if (plan.neighborhood_candidates.size() >= config.max_neighborhoods ||
                plan.singleton_candidates.size() + plan.neighborhood_candidates.size() >=
                    config.max_candidates) break;
        const uint32_t layer = anchor.diagnostic.layer_index;
        const std::vector<uint32_t> neighbors = {
            layer > 1 ? layer - 1 : 0,
            layer == std::numeric_limits<uint32_t>::max() ? 0 : layer + 1,
        };
        for (const uint32_t neighbor : neighbors) {
            if (neighbor == 0 || !contains_layer(available_layers, neighbor)) continue;
            std::vector<uint32_t> pair = {layer, neighbor};
            std::sort(pair.begin(), pair.end());
            if (candidate_has_layers(plan.neighborhood_candidates, pair)) continue;
            common_flydelta_layer_candidate candidate;
            candidate.layer_indices = pair;
            candidate.anchor_layer_index = layer;
            candidate.diagnostic_score = anchor.score;
            candidate.total_scale = config.total_scale;
            candidate.per_layer_scale = per_layer_scale(config.total_scale, pair.size());
            candidate.source = common_flydelta_layer_search_candidate_source::neighborhood_expansion;
            plan.neighborhood_candidates.push_back(std::move(candidate));
            if (plan.neighborhood_candidates.size() >= config.max_neighborhoods ||
                    plan.singleton_candidates.size() + plan.neighborhood_candidates.size() >=
                        config.max_candidates) break;
        }
    }
    return common_flydelta_layer_search_plan_validate(plan, config, error);
}

bool common_flydelta_layer_search_trial_validate(
        const common_flydelta_layer_search_trial & trial,
        std::string & error) {
    error.clear();
    if (!common_flydelta_layer_candidate_validate(trial.candidate, error) ||
            !valid_outcome(trial.outcome) || !std::isfinite(trial.quality_delta) ||
            trial.quality_delta < -1.0f || trial.quality_delta > 1.0f ||
            (trial.verifier_known && trial.evidence_ref.empty())) {
        if (error.empty()) error = "FlyDelta layer search trial is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_select_layer_candidate(
        const common_flydelta_layer_search_plan & plan,
        const std::vector<common_flydelta_layer_search_trial> & trials,
        common_flydelta_layer_search_selection & selection,
        std::string & error) {
    error.clear();
    selection = {};
    common_flydelta_layer_search_config config;
    config.max_singletons = std::max<size_t>(1, plan.singleton_candidates.size());
    config.max_neighborhoods = plan.neighborhood_candidates.size();
    config.max_candidates = std::max<size_t>(1,
        plan.singleton_candidates.size() + plan.neighborhood_candidates.size());
    if (!common_flydelta_layer_search_plan_validate(plan, config, error)) return false;

    float best_score = -std::numeric_limits<float>::infinity();
    size_t best_index = 0;
    bool found = false;
    for (size_t i = 0; i < trials.size(); ++i) {
        const auto & trial = trials[i];
        if (!common_flydelta_layer_search_trial_validate(trial, error)) return false;
        const bool planned = std::any_of(plan.singleton_candidates.begin(), plan.singleton_candidates.end(),
                [&](const auto & candidate) { return candidate.layer_indices == trial.candidate.layer_indices; }) ||
            std::any_of(plan.neighborhood_candidates.begin(), plan.neighborhood_candidates.end(),
                [&](const auto & candidate) { return candidate.layer_indices == trial.candidate.layer_indices; });
        if (!planned) {
            error = "FlyDelta layer search trial is not in the plan";
            return false;
        }
        if (trial.outcome != common_flydelta_counterfactual_outcome::helped ||
                !trial.executed || !trial.verifier_known) continue;
        const bool better = !found || trial.quality_delta > best_score ||
            (trial.quality_delta == best_score &&
             trial.candidate.layer_indices.size() < selection.candidate.layer_indices.size()) ||
            (trial.quality_delta == best_score &&
             trial.candidate.layer_indices.size() == selection.candidate.layer_indices.size() &&
             trial.candidate.total_scale < selection.candidate.total_scale);
        if (better) {
            found = true;
            best_score = trial.quality_delta;
            best_index = i;
            selection.candidate = trial.candidate;
        }
    }
    if (found) {
        selection.selected = true;
        selection.score = best_score;
        selection.trial_index = best_index;
    }
    return true;
}
