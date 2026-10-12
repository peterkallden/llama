#pragma once

#include "agent/adaptation/flydelta/flydelta-evidence-depth.h"

#include <cstddef>
#include <string>
#include <unordered_set>
#include <vector>

// Bounded in-memory state for one compatible FlyDelta identity. The append-only
// corpus remains the source of truth; this object is a derived worker snapshot
// used to amortize cheap aggregation between experiment runs.
struct common_flydelta_aggregation_config {
    common_flydelta_direction_search_config identity;
    common_flydelta_evidence_depth_config depth;
    std::string scope_fingerprint;
    std::string tokenizer_fingerprint;
    std::string template_fingerprint;
    std::string generation_semantics_fingerprint;
    size_t max_retained_samples = 32;
};

struct common_flydelta_aggregation_snapshot {
    size_t observations_seen = 0;
    size_t compatible_samples = 0;
    size_t evidence_eligible_samples = 0;
    size_t experimental_samples = 0;
    size_t rejected_samples = 0;
    std::vector<float> mean_direction;
    std::vector<float> variance;
    std::vector<std::string> seen_sample_ids;
    std::vector<std::string> retained_sample_ids;
    std::vector<common_flydelta_contrast_sample> retained_samples;
    std::vector<std::string> evidence_retained_sample_ids;
    std::vector<common_flydelta_contrast_sample> evidence_retained_samples;
};

bool common_flydelta_aggregation_config_validate(
        const common_flydelta_aggregation_config & config,
        std::string & error);

class common_flydelta_incremental_aggregation final {
public:
    explicit common_flydelta_incremental_aggregation(
            common_flydelta_aggregation_config config);

    // Ingests exactly one host-derived sample. Incompatible or HARMED samples
    // are counted as rejected and do not affect aggregate directions. UNKNOWN
    // and NEUTRAL remain valid experimental material.
    bool ingest(
            const common_flydelta_contrast_sample & sample,
            std::string & error);

    bool assess_depth(
            common_flydelta_evidence_depth_result & result,
            std::string & error) const;

    common_flydelta_aggregation_snapshot snapshot() const;
    const common_flydelta_aggregation_config & config() const { return config_; }

private:
    common_flydelta_aggregation_config config_;
    size_t observations_seen_ = 0;
    size_t compatible_samples_ = 0;
    size_t evidence_eligible_samples_ = 0;
    size_t experimental_samples_ = 0;
    size_t rejected_samples_ = 0;
    std::vector<float> mean_direction_;
    std::vector<float> m2_;
    std::unordered_set<std::string> seen_sample_ids_;
    std::vector<std::string> retained_sample_ids_;
    std::vector<common_flydelta_contrast_sample> retained_samples_;
    std::vector<std::string> evidence_retained_sample_ids_;
    std::vector<common_flydelta_contrast_sample> evidence_retained_samples_;
};
