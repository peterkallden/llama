#include "agent/adaptation/flydelta/flydelta-basis.h"

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_behavior_delta delta(const std::vector<float> & values) {
    common_flydelta_behavior_delta value;
    value.id = "flydelta://delta/1";
    value.source = common_adaptation_evidence_source::tool_repair;
    value.behavior_key = "tool_use/diagnostics/missing-argument";
    value.capture_manifest_id = "flydelta://capture/1";
    value.host_evidence_ref = "evidence:repair";
    value.model_profile_fingerprint = "sha256:model";
    value.execution_context_fingerprint = "sha256:execution-context";
    value.capture_layout_revision = "layout:v1";
    value.layer_index = 12;
    value.values = values;
    return value;
}

static common_flydelta_intervention_credit credit(common_flydelta_counterfactual_outcome outcome) {
    common_flydelta_intervention_credit value;
    value.experiment_id = "flydelta://experiment/1";
    value.candidate_id = "flydelta://candidate/1";
    value.fixture_id = "flydelta://fixture/1";
    value.outcome = outcome;
    value.quality_delta = outcome == common_flydelta_counterfactual_outcome::helped ? 0.5f : 0.0f;
    value.eligible_for_learning = outcome == common_flydelta_counterfactual_outcome::helped;
    return value;
}

int main() {
    std::string error;
    common_flydelta_basis_config config;
    config.dimension = 3;
    config.cluster_similarity = 0.8f;
    config.source = common_adaptation_evidence_source::tool_repair;
    config.behavior_key = "tool_use/diagnostics/missing-argument";
    config.model_profile_fingerprint = "sha256:model";
    config.execution_context_fingerprint = "sha256:execution-context";
    config.capture_layout_revision = "layout:v1";
    common_flydelta_basis_builder builder(config);

    CHECK(builder.add(delta({1.0f, 0.0f, 0.0f}), credit(common_flydelta_counterfactual_outcome::helped), error));
    CHECK(builder.directions().size() == 1);
    CHECK(builder.directions()[0].helped_observations == 1);

    auto similar = delta({0.9f, 0.1f, 0.0f});
    similar.id = "flydelta://delta/2";
    CHECK(builder.add(similar, credit(common_flydelta_counterfactual_outcome::helped), error));
    CHECK(builder.directions().size() == 1);
    CHECK(builder.directions()[0].helped_observations == 2);

    auto other_layer = delta({1.0f, 0.0f, 0.0f});
    other_layer.id = "flydelta://delta/other-layer";
    other_layer.layer_index = 13;
    CHECK(builder.add(other_layer, credit(common_flydelta_counterfactual_outcome::helped), error));
    CHECK(builder.directions().size() == 2);

    auto wrong_behavior = delta({1.0f, 0.0f, 0.0f});
    wrong_behavior.id = "flydelta://delta/wrong-behavior";
    wrong_behavior.behavior_key = "planning/plan-revision";
    CHECK(!builder.add(wrong_behavior,
        credit(common_flydelta_counterfactual_outcome::helped), error));

    auto wrong_source = delta({1.0f, 0.0f, 0.0f});
    wrong_source.id = "flydelta://delta/wrong-source";
    wrong_source.source = common_adaptation_evidence_source::planning_revision;
    CHECK(!builder.add(wrong_source,
        credit(common_flydelta_counterfactual_outcome::helped), error));

    auto harmed = delta({1.0f, 0.0f, 0.0f});
    harmed.id = "flydelta://delta/3";
    CHECK(builder.add(harmed, credit(common_flydelta_counterfactual_outcome::harmed), error));
    CHECK(builder.directions()[0].harmed_observations == 1);

    auto unknown = delta({0.0f, 1.0f, 0.0f});
    unknown.id = "flydelta://delta/4";
    unknown.layer_index = 13;
    CHECK(builder.add(unknown, credit(common_flydelta_counterfactual_outcome::unknown), error));
    CHECK(builder.directions().size() == 2);
    CHECK(!common_flydelta_behavior_delta_validate(delta({0.0f, 0.0f, 0.0f}), 3, 1024, error));
    return 0;
}
