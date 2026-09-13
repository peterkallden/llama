#pragma once

#include "agent/adaptation/flydelta/flydelta-contracts.h"
#include "agent/adaptation/flydelta/flydelta-experiment.h"

#include <cstddef>
#include <string>
#include <vector>

struct common_flydelta_promotion_policy {
    size_t min_trials = 8;
    size_t min_known_trials = 6;
    size_t min_helped_trials = 3;
    size_t max_harmed_trials = 0;
    float min_help_confidence = 0.75f;
    float max_unknown_ratio = 0.25f;
    size_t max_trials = 256;
};

bool common_flydelta_promotion_policy_validate(
        const common_flydelta_promotion_policy & policy,
        std::string & error);

// Aggregates only already validated, host-owned counterfactual reports. This
// summary can become eligible, but never becomes approved automatically.
struct common_flydelta_promotion_summary {
    int schema_version = 1;
    std::string id;
    std::string candidate_id;
    std::string baseline_profile_id;
    std::string candidate_profile_id;
    size_t total_trials = 0;
    size_t known_trials = 0;
    size_t helped_trials = 0;
    size_t neutral_trials = 0;
    size_t harmed_trials = 0;
    size_t unknown_trials = 0;
    float help_confidence = 0.0f;
    float mean_quality_delta = 0.0f;
    common_flydelta_candidate_status status = common_flydelta_candidate_status::observed;
};

bool common_flydelta_promotion_summary_validate(
        const common_flydelta_promotion_summary & summary,
        const common_flydelta_promotion_policy & policy,
        std::string & error);

bool common_flydelta_promotion_summary_from_reports(
        const std::string & summary_id,
        const std::vector<common_flydelta_counterfactual_report> & reports,
        const common_flydelta_promotion_policy & policy,
        common_flydelta_promotion_summary & summary,
        std::string & error);
