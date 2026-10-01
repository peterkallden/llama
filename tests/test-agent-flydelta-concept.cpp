#include "agent/adaptation/flydelta/flydelta-concept.h"
#include "agent/adaptation/flydelta/flydelta-semantic-decision.h"

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_concept_spec spec() {
    common_flydelta_concept_spec value;
    value.concept_key = "concept://test/verify-before-tool";
    value.extraction_id = "extraction://test/verify-before-tool/1";
    value.behavior_key = "tool_use/validated-filter";
    value.source_ref = "relation://test/verify-before-tool/1";
    value.grounding_ref = "grounding://test/verify-before-tool/1";
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

static common_flydelta_concept_prototype_sample prototype_sample(
        const common_flydelta_concept_spec & concept_spec,
        int index, bool positive) {
    common_flydelta_concept_prototype_sample value;
    value.id = std::string(positive ? "positive://test/" : "control://test/") +
        std::to_string(index);
    value.capture_ref = value.id + "/capture";
    value.semantic_anchor = "generation_boundary";
    value.model_profile_fingerprint = concept_spec.model_profile_fingerprint;
    value.tokenizer_fingerprint = concept_spec.tokenizer_fingerprint;
    value.template_fingerprint = concept_spec.template_fingerprint;
    value.capture_layout_revision = concept_spec.capture_layout_revision;
    value.scope_fingerprint = concept_spec.scope_fingerprint;
    value.verifier_ref = concept_spec.verifier_ref;
    value.layer_index = 12;
    value.values = positive
        ? std::vector<float>{1.0f + index * 0.05f, 0.7f, 0.2f}
        : std::vector<float>{0.2f, 0.7f + index * 0.03f, 1.0f};
    value.host_verified = true;
    value.independent = true;
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

    auto trim_config = config;
    trim_config.trim_fraction = 0.25f;
    auto trim_trajectories = trajectories;
    trim_trajectories.push_back(trajectory(3));
    trim_trajectories.back().conditioned = {3.2f, 1.1f, 0.4f};
    CHECK(common_flydelta_build_concept_candidates(
        concept_spec, trim_config, trim_trajectories, candidates, error));
    CHECK(candidates.size() == 3);
    // Keep the three aligned trajectories and discard the deliberately
    // orthogonal fourth trajectory.
    CHECK(candidates[1].retained_trajectories == 3);
    CHECK(candidates[1].values[0] > candidates[1].values[1] + 0.1f);

    auto unverified = trajectories;
    unverified.front().conditioned_host_verified = false;
    CHECK(!common_flydelta_build_concept_candidates(
        concept_spec, config, unverified, candidates, error));

    auto unapproved = concept_spec;
    unapproved.host_approved = false;
    CHECK(!common_flydelta_concept_spec_validate(unapproved, error));

    auto user_grounded = concept_spec;
    user_grounded.source = common_adaptation_evidence_source::user_correction;
    user_grounded.source_ref = "relation://user-correction/verify-before-tool/1";
    user_grounded.grounding_ref = "grounding://user-concept/verify-before-tool/1";
    user_grounded.procedure_ref.clear();
    CHECK(common_flydelta_concept_spec_validate(user_grounded, error));

    auto wrong_layer = trajectories;
    wrong_layer.back().layer_index = 13;
    CHECK(!common_flydelta_build_concept_candidates(
        concept_spec, config, wrong_layer, candidates, error));

    auto no_control = config;
    no_control.require_control = false;
    CHECK(!common_flydelta_concept_build_config_validate(no_control, error));

    common_flydelta_concept_prototype_build_config prototype_config;
    prototype_config.dimension = 3;
    std::vector<common_flydelta_concept_prototype_sample> positive = {
        prototype_sample(concept_spec, 0, true),
        prototype_sample(concept_spec, 1, true),
        prototype_sample(concept_spec, 2, true),
    };
    std::vector<common_flydelta_concept_prototype_sample> controls = {
        prototype_sample(concept_spec, 0, false),
        prototype_sample(concept_spec, 1, false),
        prototype_sample(concept_spec, 2, false),
    };
    CHECK(common_flydelta_build_positive_prototype_candidates(
        concept_spec, prototype_config, positive, controls, candidates, error));
    CHECK(candidates.size() == 3);
    for (const auto & candidate : candidates) {
        CHECK(candidate.synthesis_semantics ==
            common_flydelta_concept_synthesis_semantics::positive_prototype);
        CHECK(candidate.origin == "host_taught_positive_prototype");
        CHECK(!candidate.control_residualized);
        CHECK(candidate.source_trajectories == positive.size());
        CHECK(candidate.control_trajectories == controls.size());
        CHECK(common_flydelta_concept_candidate_validate(candidate, 3, error));
        common_flydelta_direction_candidate direction;
        CHECK(common_flydelta_concept_candidate_to_direction(candidate, direction, error));
        CHECK(direction.experimental_only);
        const auto expected_kind = candidate.kind ==
                common_flydelta_concept_candidate_kind::raw_mean
            ? common_flydelta_direction_kind::positive_prototype
            : candidate.kind == common_flydelta_concept_candidate_kind::trimmed_mean
                ? common_flydelta_direction_kind::positive_prototype_trimmed_mean
                : common_flydelta_direction_kind::positive_prototype_diagonal_whitened_mean;
        CHECK(direction.kind == expected_kind);
    }
    auto incompatible_control = controls;
    incompatible_control.front().scope_fingerprint = "sha256:other-scope";
    CHECK(!common_flydelta_build_positive_prototype_candidates(
        concept_spec, prototype_config, positive, incompatible_control, candidates, error));
    auto unverified_positive = positive;
    unverified_positive.front().host_verified = false;
    CHECK(!common_flydelta_build_positive_prototype_candidates(
        concept_spec, prototype_config, unverified_positive, controls, candidates, error));

    common_flydelta_semantic_decision normalized_filter;
    common_flydelta_semantic_decision_status status;
    CHECK(common_flydelta_parse_semantic_decision(
        R"({"name":"data.filter","arguments":{"dataset":"sales.csv","predicate":"status == 'failed'"}})",
        normalized_filter, status, error));
    common_flydelta_semantic_decision expected_filter;
    CHECK(common_flydelta_parse_semantic_decision(
        R"({"operation":"filter","dataset":"sales.csv","predicate":"status = failed"})",
        expected_filter, status, error));
    CHECK(common_flydelta_semantic_decision_equal(normalized_filter, expected_filter));

    common_flydelta_semantic_decision normalized_query;
    CHECK(common_flydelta_parse_semantic_decision(
        R"({"operation":"query","dataset":"sales.csv","order_by":"timestamp newest first","limit":5})",
        normalized_query, status, error));
    CHECK(normalized_query.order_by_direction == "desc");

    common_flydelta_semantic_decision ignored_decision;
    CHECK(!common_flydelta_parse_semantic_decision(
        R"({"name":"data.aggregate","arguments":{"group_by":["region"]}})",
        ignored_decision, status, error));
    CHECK(status == common_flydelta_semantic_decision_status::missing_field);
    CHECK(!common_flydelta_parse_semantic_decision(
        R"({"operation":"unknown"})", ignored_decision, status, error));
    CHECK(status == common_flydelta_semantic_decision_status::unsupported_operation);
    return 0;
}
