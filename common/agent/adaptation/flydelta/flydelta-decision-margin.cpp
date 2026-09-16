#include "agent/adaptation/flydelta/flydelta-decision-margin.h"

#include <cmath>

float common_flydelta_decision_margin::normalized_delta() const {
    if (positive_token_count == 0 || negative_token_count == 0) return 0.0f;
    return positive_total_logprob / static_cast<float>(positive_token_count) -
        negative_total_logprob / static_cast<float>(negative_token_count);
}

bool common_flydelta_decision_margin_validate(
        const common_flydelta_decision_margin & margin,
        std::string & error) {
    error.clear();
    if (!margin.available) return true;
    if (!std::isfinite(margin.positive_total_logprob) ||
            !std::isfinite(margin.negative_total_logprob) ||
            margin.positive_token_count == 0 || margin.negative_token_count == 0 ||
            !std::isfinite(margin.total_delta()) ||
            !std::isfinite(margin.normalized_delta())) {
        error = "FlyDelta decision margin is invalid";
        return false;
    }
    return true;
}

