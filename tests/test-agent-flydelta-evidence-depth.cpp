#include "agent/adaptation/flydelta/flydelta-evidence-depth.h"

#include <cmath>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_contrast_sample sample(
        const char * id, const std::vector<float> & values,
        common_flydelta_counterfactual_outcome outcome =
            common_flydelta_counterfactual_outcome::unknown) {
    common_flydelta_contrast_sample result;
    result.delta.id = id;
    result.delta.source = common_adaptation_evidence_source::tool_repair;
    result.delta.behavior_key = "tool_use/diagnostics/wrong-tool";
    result.delta.capture_manifest_id = "flydelta://capture/depth";
    result.delta.host_evidence_ref = "evidence:depth";
    result.delta.model_profile_fingerprint = "sha256:model";
    result.delta.execution_context_fingerprint = "sha256:context";
    result.delta.capture_layout_revision = "layer-input:v1";
    result.delta.layer_index = 2;
    result.delta.values = values;
    result.credit.experiment_id = "flydelta://experiment/depth";
    result.credit.candidate_id = id;
    result.credit.fixture_id = "flydelta://fixture/depth";
    result.credit.outcome = outcome;
    return result;
}

static common_flydelta_direction_search_config identity() {
    common_flydelta_direction_search_config result;
    result.dimension = 4;
    result.layer_index = 2;
    result.min_samples = 1;
    result.max_samples = 8;
    result.min_median_alignment = 0.25f;
    result.behavior_key = "tool_use/diagnostics/wrong-tool";
    result.model_profile_fingerprint = "sha256:model";
    result.execution_context_fingerprint = "sha256:context";
    result.capture_layout_revision = "layer-input:v1";
    return result;
}

int main() {
    std::string error;
    const auto config = common_flydelta_evidence_depth_config{};

    common_flydelta_evidence_depth_result result;
    CHECK(common_flydelta_assess_evidence_depth(
        identity(), config,
        {sample("flydelta://sample/one", {1.0f, 0.0f, 0.0f, 0.0f})}, result, error));
    CHECK(result.depth == common_flydelta_search_depth::bootstrap);
    CHECK(result.compatible_samples == 1 && result.effective_rank == 1);

    CHECK(common_flydelta_assess_evidence_depth(
        identity(), config,
        {sample("flydelta://sample/one", {1.0f, 0.0f, 0.0f, 0.0f}),
         sample("flydelta://sample/two", {0.0f, 1.0f, 0.0f, 0.0f})}, result, error));
    CHECK(result.depth == common_flydelta_search_depth::shallow);
    CHECK(result.shallow_ready && !result.deep_ready);
    CHECK(result.effective_rank == 2 && result.condition_number < 2.0f);

    CHECK(common_flydelta_assess_evidence_depth(
        identity(), config,
        {sample("flydelta://sample/near-one", {1.0f, 0.0f, 0.0f, 0.0f}),
         sample("flydelta://sample/near-two", {0.999f, 0.04f, 0.0f, 0.0f})}, result, error));
    CHECK(result.depth == common_flydelta_search_depth::bootstrap);
    CHECK(result.effective_rank == 1);

    CHECK(common_flydelta_assess_evidence_depth(
        identity(), config,
        {sample("flydelta://sample/a", {1.0f, 0.0f, 0.0f, 0.0f}),
         sample("flydelta://sample/b", {0.9f, 0.2f, 0.0f, 0.0f}),
         sample("flydelta://sample/c", {0.8f, 0.4f, 0.0f, 0.0f}),
         sample("flydelta://sample/d", {0.7f, 0.5f, 0.0f, 0.0f}),
         sample("flydelta://sample/e", {0.6f, 0.6f, 0.0f, 0.0f}),
         sample("flydelta://sample/f", {0.5f, 0.7f, 0.0f, 0.0f})}, result, error));
    CHECK(result.depth == common_flydelta_search_depth::deep);
    CHECK(result.deep_ready && result.compatible_samples == 6);
    CHECK(result.stable_rank > 1.0f);

    auto incompatible = sample("flydelta://sample/incompatible", {0.0f, 0.0f, 1.0f, 0.0f});
    incompatible.delta.behavior_key = "tool_use/other";
    auto harmed = sample("flydelta://sample/harmed", {0.0f, 0.0f, 1.0f, 0.0f},
        common_flydelta_counterfactual_outcome::harmed);
    CHECK(common_flydelta_assess_evidence_depth(
        identity(), config, {sample("flydelta://sample/valid", {1.0f, 0.0f, 0.0f, 0.0f}),
        incompatible, harmed}, result, error));
    CHECK(result.compatible_samples == 1 && result.incompatible_samples == 2);
    CHECK(std::string(common_flydelta_search_depth_name(result.depth)) == "bootstrap");

    auto bad_config = config;
    bad_config.max_condition_number = 0.0f;
    CHECK(!common_flydelta_assess_evidence_depth(identity(), bad_config, {}, result, error));
    CHECK(!error.empty());
    return 0;
}
