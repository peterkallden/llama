#include "agent/adaptation/flydelta/flydelta-layer-search.h"

#include <cmath>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_layer_diagnostic diagnostic(
        uint32_t layer, float cosine, float progress, float leakage) {
    return {layer, cosine, progress, leakage, 1.0f};
}

static common_flydelta_layer_search_trial trial(
        const common_flydelta_layer_candidate & candidate,
        common_flydelta_counterfactual_outcome outcome,
        float quality) {
    common_flydelta_layer_search_trial value;
    value.candidate = candidate;
    value.outcome = outcome;
    value.quality_delta = quality;
    value.executed = true;
    value.verifier_known = true;
    value.evidence_ref = "evidence:layer-search";
    return value;
}

int main() {
    std::string error;
    common_flydelta_layer_search_config config;
    config.total_scale = 0.08f;
    common_flydelta_layer_search_plan plan;
    CHECK(common_flydelta_build_layer_search_plan(
        {
            diagnostic(11, 0.4f, 0.2f, 0.1f),
            diagnostic(12, 0.8f, 0.7f, 0.1f),
            diagnostic(13, 0.7f, 0.5f, 0.1f),
            diagnostic(18, 0.7f, 0.4f, 0.1f),
            diagnostic(19, 0.6f, 0.3f, 0.1f),
            diagnostic(24, 0.9f, 0.01f, 0.1f),
        }, {11, 12, 13, 14, 17, 18, 19, 23, 24, 25}, config, plan, error));
    CHECK(plan.singleton_candidates.size() == 2);
    CHECK(plan.singleton_candidates[0].layer_indices == std::vector<uint32_t>({12}));
    CHECK(plan.singleton_candidates[1].layer_indices == std::vector<uint32_t>({18}));
    CHECK(plan.neighborhood_candidates.size() == 4);
    CHECK(plan.neighborhood_candidates[0].layer_indices == std::vector<uint32_t>({11, 12}));
    CHECK(plan.neighborhood_candidates[1].layer_indices == std::vector<uint32_t>({12, 13}));
    CHECK(std::fabs(plan.neighborhood_candidates[0].per_layer_scale - (0.08f / std::sqrt(2.0f))) < 0.00001f);
    CHECK(common_flydelta_layer_search_plan_validate(plan, config, error));

    // A singleton HELPED wins over a higher-cost neighborhood. UNKNOWN and
    // NEUTRAL remain non-selectable even when their geometry looks promising.
    std::vector<common_flydelta_layer_search_trial> trials;
    trials.push_back(trial(plan.singleton_candidates[0],
        common_flydelta_counterfactual_outcome::helped, 0.4f));
    trials.push_back(trial(plan.singleton_candidates[1],
        common_flydelta_counterfactual_outcome::unknown, 0.9f));
    trials.push_back(trial(plan.neighborhood_candidates[0],
        common_flydelta_counterfactual_outcome::neutral, 0.8f));
    common_flydelta_layer_search_selection selection;
    CHECK(common_flydelta_select_layer_candidate(plan, trials, selection, error));
    CHECK(selection.selected && selection.trial_index == 0);
    CHECK(selection.candidate.layer_indices == std::vector<uint32_t>({12}));

    // No host-certified HELPED means no selection.
    trials[0].outcome = common_flydelta_counterfactual_outcome::unknown;
    CHECK(common_flydelta_select_layer_candidate(plan, trials, selection, error));
    CHECK(!selection.selected);

    // A missing captured neighbor must not be invented by the planner.
    CHECK(common_flydelta_build_layer_search_plan(
        {diagnostic(12, 0.8f, 0.7f, 0.1f)}, {12}, config, plan, error));
    CHECK(plan.singleton_candidates.size() == 1 && plan.neighborhood_candidates.empty());
    return 0;
}
