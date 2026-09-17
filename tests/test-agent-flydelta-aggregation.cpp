#include "agent/adaptation/flydelta/flydelta-aggregation.h"

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
    result.delta.capture_manifest_id = "flydelta://capture/aggregation";
    result.delta.host_evidence_ref = "evidence:aggregation";
    result.delta.model_profile_fingerprint = "sha256:model";
    result.delta.execution_context_fingerprint = "sha256:context";
    result.delta.capture_layout_revision = "layer-input:v1";
    result.delta.layer_index = 2;
    result.delta.values = values;
    result.credit.experiment_id = "flydelta://experiment/aggregation";
    result.credit.candidate_id = id;
    result.credit.fixture_id = "flydelta://fixture/aggregation";
    result.credit.outcome = outcome;
    return result;
}

static common_flydelta_aggregation_config config() {
    common_flydelta_aggregation_config result;
    result.identity.dimension = 4;
    result.identity.layer_index = 2;
    result.identity.min_samples = 1;
    result.identity.max_samples = 8;
    result.identity.behavior_key = "tool_use/diagnostics/wrong-tool";
    result.identity.model_profile_fingerprint = "sha256:model";
    result.identity.execution_context_fingerprint = "sha256:context";
    result.identity.capture_layout_revision = "layer-input:v1";
    result.max_retained_samples = 6;
    return result;
}

int main() {
    std::string error;
    common_flydelta_incremental_aggregation aggregation(config());
    common_flydelta_evidence_depth_result depth;
    CHECK(aggregation.assess_depth(depth, error));
    CHECK(depth.depth == common_flydelta_search_depth::bootstrap);

    CHECK(aggregation.ingest(sample("flydelta://sample/one", {1.0f, 0.0f, 0.0f, 0.0f}), error));
    CHECK(aggregation.ingest(sample("flydelta://sample/two", {0.0f, 1.0f, 0.0f, 0.0f}), error));
    CHECK(aggregation.ingest(sample("flydelta://sample/one", {1.0f, 0.0f, 0.0f, 0.0f}), error));
    CHECK(aggregation.assess_depth(depth, error));
    CHECK(depth.depth == common_flydelta_search_depth::shallow);

    CHECK(aggregation.ingest(sample("flydelta://sample/three", {0.9f, 0.2f, 0.0f, 0.0f}), error));
    CHECK(aggregation.ingest(sample("flydelta://sample/four", {0.8f, 0.4f, 0.0f, 0.0f}), error));
    CHECK(aggregation.ingest(sample("flydelta://sample/five", {0.7f, 0.5f, 0.0f, 0.0f}), error));
    CHECK(aggregation.ingest(sample("flydelta://sample/six", {0.6f, 0.6f, 0.0f, 0.0f}), error));
    CHECK(aggregation.assess_depth(depth, error));
    CHECK(depth.depth == common_flydelta_search_depth::deep);

    CHECK(aggregation.ingest(sample("flydelta://sample/seven", {0.5f, 0.7f, 0.0f, 0.0f}), error));
    CHECK(aggregation.ingest(sample("flydelta://sample/eight", {0.4f, 0.8f, 0.0f, 0.0f}), error));

    auto incompatible = sample("flydelta://sample/incompatible", {0.0f, 0.0f, 1.0f, 0.0f});
    incompatible.delta.behavior_key = "tool_use/other";
    CHECK(aggregation.ingest(incompatible, error));
    auto harmed = sample("flydelta://sample/harmed", {0.0f, 0.0f, 1.0f, 0.0f},
        common_flydelta_counterfactual_outcome::harmed);
    CHECK(aggregation.ingest(harmed, error));

    const auto snapshot = aggregation.snapshot();
    CHECK(snapshot.observations_seen == 10);
    CHECK(snapshot.compatible_samples == 8 && snapshot.rejected_samples == 2);
    CHECK(snapshot.retained_samples.size() == 6);
    CHECK(snapshot.retained_sample_ids.size() == 6);
    CHECK(snapshot.seen_sample_ids.size() == 10);
    CHECK(snapshot.mean_direction.size() == 4 && snapshot.variance.size() == 4);
    CHECK(std::isfinite(snapshot.mean_direction[0]) && std::isfinite(snapshot.variance[1]));
    CHECK(std::string(common_flydelta_search_depth_name(depth.depth)) == "deep");
    return 0;
}
