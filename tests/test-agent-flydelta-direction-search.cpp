#include "agent/adaptation/flydelta/flydelta-direction-search.h"

#include <cmath>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_contrast_sample sample(
        const char * id, const std::vector<float> & values,
        common_flydelta_counterfactual_outcome outcome =
            common_flydelta_counterfactual_outcome::helped) {
    common_flydelta_contrast_sample result;
    result.delta.id = id;
    result.delta.source = common_adaptation_evidence_source::tool_repair;
    result.delta.behavior_key = "tool_use/diagnostics/wrong-tool";
    result.delta.capture_manifest_id = "flydelta://capture/contrast";
    result.delta.host_evidence_ref = "evidence:contrast";
    result.delta.model_profile_fingerprint = "sha256:model";
    result.delta.capture_layout_revision = "layer-input:v1";
    result.delta.layer_index = 2;
    result.delta.values = values;
    result.credit.experiment_id = "flydelta://experiment/contrast";
    result.credit.candidate_id = id;
    result.credit.fixture_id = "flydelta://fixture/contrast";
    result.credit.outcome = outcome;
    result.credit.quality_delta = outcome == common_flydelta_counterfactual_outcome::helped ? 1.0f : 0.0f;
    result.credit.eligible_for_learning = outcome == common_flydelta_counterfactual_outcome::helped;
    return result;
}

int main() {
    common_flydelta_direction_search_config config;
    config.dimension = 4;
    config.layer_index = 2;
    config.min_samples = 2;
    config.max_samples = 8;
    // Include all samples first, then trim the least central one.
    config.min_median_alignment = -1.0f;
    config.trim_fraction = 0.25f;
    config.variance_ridge = 0.001f;
    config.source = common_adaptation_evidence_source::tool_repair;
    config.behavior_key = "tool_use/diagnostics/wrong-tool";
    config.model_profile_fingerprint = "sha256:model";
    config.capture_layout_revision = "layer-input:v1";

    const std::vector<common_flydelta_contrast_sample> samples = {
        sample("flydelta://delta/a", {1.0f, 0.0f, 0.0f, 0.0f}),
        sample("flydelta://delta/b", {0.98f, 0.10f, 0.0f, 0.0f}),
        sample("flydelta://delta/c", {0.96f, -0.05f, 0.0f, 0.0f}),
        sample("flydelta://delta/outlier", {0.0f, 1.0f, 0.0f, 0.0f}),
    };
    std::string error;
    std::vector<common_flydelta_direction_candidate> candidates;
    CHECK(common_flydelta_build_direction_candidates(config, samples, candidates, error));
    CHECK(candidates.size() == 3);
    CHECK(candidates[0].kind == common_flydelta_direction_kind::raw_repair);
    CHECK(candidates[1].kind == common_flydelta_direction_kind::normalized_trimmed_mean);
    CHECK(candidates[2].kind == common_flydelta_direction_kind::diagonal_whitened_mean);
    CHECK(candidates[1].source_samples == 4 && candidates[1].retained_samples == 3);
    CHECK(candidates[1].median_alignment > 0.9f);
    CHECK(candidates[1].values[0] > 0.99f);
    CHECK(std::fabs(candidates[1].values[3]) < 0.0001f);
    CHECK(std::isfinite(candidates[2].values[0]));
    CHECK(std::string(common_flydelta_direction_kind_name(
        common_flydelta_direction_kind::diagonal_whitened_mean)) == "diagonal_whitened_mean");

    std::vector<common_flydelta_direction_candidate> raw_only;
    CHECK(common_flydelta_build_direction_candidates(
        config, {samples.front()}, raw_only, error));
    CHECK(raw_only.size() == 1 && raw_only.front().retained_samples == 1);

    // Six samples is the model-facing deep-search threshold. The same
    // builder can still be used with a lower threshold for CPU-only studies.
    config.min_samples = 6;
    CHECK(common_flydelta_build_direction_candidates(config, samples, raw_only, error));
    CHECK(raw_only.size() == 1 && raw_only.front().source_samples == 4);

    auto unknown = samples.front();
    unknown.delta.id = "flydelta://delta/unknown";
    unknown.credit.outcome = common_flydelta_counterfactual_outcome::unknown;
    unknown.credit.eligible_for_learning = false;
    CHECK(!common_flydelta_build_direction_candidates(config, {unknown}, raw_only, error));
    return 0;
}
