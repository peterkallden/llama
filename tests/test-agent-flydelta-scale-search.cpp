#include "agent/adaptation/flydelta/flydelta-scale-search.h"

#include <cmath>
#include <vector>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    std::string error;
    common_flydelta_scale_search_config config;
    config.initial_scale = 0.02f;
    config.growth_factor = 2.0f;
    config.max_scale = 0.32f;
    config.max_geometric_trials = 4;
    config.max_refinement_trials = 1;
    common_flydelta_scale_selection selection;
    std::vector<common_flydelta_scale_trial> trials;
    size_t calls = 0;
    CHECK(common_flydelta_run_scale_search(
        common_flydelta_experiment_fixture{
            1, "fixture:scale-search", "task", "model", "tokenizer",
            "template", "context", "verifier"
        }, config,
        [&](const common_flydelta_experiment_fixture &, float scale, bool apply_overlay,
                common_flydelta_counterfactual_trial & result,
                common_flydelta_scale_geometry & geometry, std::string &) {
            ++calls;
            result = {};
            result.executed = true;
            result.verifier_known = true;
            result.passed = apply_overlay && scale >= 0.08f;
            result.quality = result.passed ? 1.0f : 0.0f;
            result.overlay_applied = apply_overlay;
            result.evidence_ref = "evidence:scale-search";
            geometry.available = apply_overlay;
            geometry.cosine = 0.8f;
            geometry.progress = scale;
            geometry.leakage = 0.1f;
            geometry.shift_norm = scale;
            return true;
        }, trials, selection, error));
    CHECK(calls == 5); // baseline + .02, .04, .08, midpoint .06
    CHECK(trials.size() == 4);
    CHECK(trials[0].scale == 0.02f && !trials[0].refinement);
    CHECK(trials[2].scale == 0.08f && trials[2].outcome ==
        common_flydelta_counterfactual_outcome::helped);
    CHECK(trials[3].refinement && std::fabs(trials[3].scale - 0.06f) < 0.00001f);
    CHECK(selection.selected && std::fabs(selection.scale - 0.08f) < 0.00001f);

    // Unsafe geometry stops geometric escalation and cannot create a verdict.
    config.max_refinement_trials = 0;
    calls = 0;
    CHECK(common_flydelta_run_scale_search(
        common_flydelta_experiment_fixture{
            1, "fixture:scale-stop", "task", "model", "tokenizer",
            "template", "context", "verifier"
        }, config,
        [&](const common_flydelta_experiment_fixture &, float, bool apply_overlay,
                common_flydelta_counterfactual_trial & result,
                common_flydelta_scale_geometry & geometry, std::string &) {
            ++calls;
            result = {};
            result.executed = true;
            result.verifier_known = true;
            result.passed = false;
            result.evidence_ref = "evidence:scale-stop";
            geometry.available = apply_overlay;
            geometry.cosine = 0.1f;
            geometry.progress = 0.01f;
            geometry.leakage = 2.0f;
            geometry.shift_norm = 0.01f;
            return true;
        }, trials, selection, error));
    CHECK(calls == 2);
    CHECK(trials.size() == 1 && !selection.selected && !trials.front().safe_to_escalate);
    return 0;
}
