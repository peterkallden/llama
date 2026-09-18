#include "agent/adaptation/flydelta/flydelta-alpha-response-search.h"

#include <cmath>
#include <vector>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    std::string error;
    common_flydelta_alpha_response_search_config config;
    config.seed_scale = 0.05f;
    config.growth_factor = 1.6f;
    config.max_scale = 0.5f;
    config.max_expansion_trials = 4;
    config.max_zoom_trials = 4;
    config.max_min_effective_trials = 2;
    common_flydelta_alpha_response_selection selection;
    std::vector<common_flydelta_alpha_response_trial> trials;
    size_t calls = 0;
    CHECK(common_flydelta_run_alpha_response_search(
        common_flydelta_experiment_fixture{
            1, "fixture:alpha-response", "task", "model", "tokenizer",
            "template", "context", "verifier"
        }, config,
        [&](const common_flydelta_experiment_fixture &, float scale, bool apply_overlay,
                common_flydelta_counterfactual_trial & counterfactual,
                common_flydelta_decision_margin & margin,
                common_flydelta_representation_diagnostics & geometry,
                bool & geometry_available, std::string &) {
            ++calls;
            counterfactual = {};
            counterfactual.executed = true;
            counterfactual.verifier_known = true;
            counterfactual.passed = apply_overlay && scale >= 0.2f;
            counterfactual.quality = counterfactual.passed ? 1.0f : 0.0f;
            counterfactual.overlay_applied = apply_overlay;
            counterfactual.evidence_ref = "evidence:alpha-response";
            margin = {};
            margin.available = true;
            margin.positive_token_count = 1;
            margin.negative_token_count = 1;
            margin.positive_total_logprob = apply_overlay ? -1.0f + scale * 4.0f : -1.0f;
            margin.negative_total_logprob = -2.0f;
            geometry = {};
            geometry_available = apply_overlay;
            geometry.cosine = 0.8f;
            geometry.progress = scale;
            geometry.leakage = 0.05f;
            geometry.shift_norm = scale;
            return true;
        }, trials, selection, error));
    CHECK(calls > 3);
    CHECK(selection.selected);
    CHECK(selection.minimum_effective_available);
    CHECK(selection.minimum_effective_scale >= 0.19f &&
        selection.minimum_effective_scale <= 0.21f);
    CHECK(std::fabs(selection.scale - selection.minimum_effective_scale) > 0.0001f ||
        selection.scale >= 0.2f);
    for (const auto & trial : trials) {
        CHECK(common_flydelta_alpha_response_trial_validate(trial, error));
        CHECK(trial.margin_available);
    }
    return 0;
}
