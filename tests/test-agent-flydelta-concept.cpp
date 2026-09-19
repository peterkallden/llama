#include "agent/adaptation/flydelta/flydelta-concept.h"

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_concept_spec spec() {
    common_flydelta_concept_spec value;
    value.concept_key = "concept://test/verify-before-tool";
    value.extraction_id = "extraction://test/verify-before-tool/1";
    value.behavior_key = "tool_use/validated-filter";
    value.procedure_ref = "procedure://test/verify-before-tool/v1";
    value.verifier_ref = "verifier://test/validated-filter/v1";
    value.model_profile_fingerprint = "sha256:model";
    value.tokenizer_fingerprint = "sha256:tokenizer";
    value.template_fingerprint = "sha256:template";
    value.capture_layout_revision = "generation-boundary:v1";
    value.scope_fingerprint = "sha256:scope";
    value.host_approved = true;
    value.redaction_attested = true;
    return value;
}

static common_flydelta_concept_trajectory trajectory(int index, bool verified = true) {
    common_flydelta_concept_trajectory value;
    value.id = "trajectory://test/" + std::to_string(index);
    value.fixture_ref = "fixture://test/" + std::to_string(index);
    value.baseline_capture_ref = value.fixture_ref + "/baseline";
    value.conditioned_capture_ref = value.fixture_ref + "/conditioned";
    value.control_capture_ref = value.fixture_ref + "/control";
    value.semantic_anchor = "generation_boundary";
    value.layer_index = 12;
    value.baseline = {0.1f + index, 0.2f, 0.3f};
    value.control = {0.2f + index, 0.1f, 0.4f};
    value.conditioned = {1.2f + index, 0.6f, 0.8f};
    value.aligned = true;
    value.conditioned_host_verified = verified;
    return value;
}

int main() {
    std::string error;
    const auto concept_spec = spec();
    CHECK(common_flydelta_concept_spec_validate(concept_spec, error));

    common_flydelta_concept_build_config config;
    config.dimension = 3;
    std::vector<common_flydelta_concept_trajectory> trajectories = {
        trajectory(0), trajectory(1), trajectory(2)};
    std::vector<common_flydelta_concept_candidate> candidates;
    CHECK(common_flydelta_build_concept_candidates(
        concept_spec, config, trajectories, candidates, error));
    CHECK(candidates.size() == 3);
    for (const auto & candidate : candidates) {
        CHECK(common_flydelta_concept_candidate_validate(candidate, 3, error));
        common_flydelta_direction_candidate direction;
        CHECK(common_flydelta_concept_candidate_to_direction(candidate, direction, error));
        CHECK(direction.experimental_only);
        CHECK(direction.origin == "host_taught_extracted");
        CHECK(direction.extraction_id == "extraction://test/verify-before-tool/1");
        CHECK(candidate.experimental_only);
        CHECK(!candidate.learning_eligible);
        CHECK(candidate.control_residualized);
    }

    auto unverified = trajectories;
    unverified.front().conditioned_host_verified = false;
    CHECK(!common_flydelta_build_concept_candidates(
        concept_spec, config, unverified, candidates, error));

    auto unapproved = concept_spec;
    unapproved.host_approved = false;
    CHECK(!common_flydelta_concept_spec_validate(unapproved, error));

    auto wrong_layer = trajectories;
    wrong_layer.back().layer_index = 13;
    CHECK(!common_flydelta_build_concept_candidates(
        concept_spec, config, wrong_layer, candidates, error));

    auto no_control = config;
    no_control.require_control = false;
    CHECK(!common_flydelta_concept_build_config_validate(no_control, error));
    return 0;
}
