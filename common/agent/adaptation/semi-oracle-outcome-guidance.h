#pragma once

#include "agent/adaptation/semi-oracle-store.h"

// Independent of FlyDelta activation: this family predicts search suffixes,
// never intervention utility or fact validity.
struct common_semi_oracle_outcome_guidance_config {
    std::string mode = "shadow";
    uint64_t seed = 0;
    size_t max_history_evaluations = 128;
    std::string search_algorithm = "exact";
    double suboptimality_bound = 1.0;
    std::string sampling_policy = "posterior_sampling";
};

bool common_semi_oracle_outcome_guidance_config_validate(
        const common_semi_oracle_outcome_guidance_config &, std::string & error);

bool common_semi_oracle_evaluate_with_history(
        common_semi_oracle_store & store,
        const common_semi_oracle_outcome_guidance_config & config,
        const std::string & host_namespace, const std::string & session_id,
        const common_semi_oracle_ruleset & ruleset,
        const common_semi_oracle_query & query,
        common_semi_oracle_result & result, std::string & error,
        const std::string & project_id = {});
