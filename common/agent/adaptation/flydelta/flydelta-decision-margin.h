#pragma once

#include <cstddef>
#include <string>

// Model-facing, teacher-forced decision material. It is deliberately
// independent of tool semantics so the same contract can score planning,
// research or other structured alternatives.
struct common_flydelta_decision_margin {
    bool available = false;
    float positive_total_logprob = 0.0f;
    float negative_total_logprob = 0.0f;
    size_t positive_token_count = 0;
    size_t negative_token_count = 0;

    float total_delta() const {
        return positive_total_logprob - negative_total_logprob;
    }
    float normalized_delta() const;
};

bool common_flydelta_decision_margin_validate(
        const common_flydelta_decision_margin & margin,
        std::string & error);

