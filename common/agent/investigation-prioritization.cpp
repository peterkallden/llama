#include "investigation-contracts.h"
#include "agent/prediction/outcome-guidance.h"

#include <algorithm>

std::vector<common_investigation_candidate> common_investigation_rank_candidates(
        std::vector<common_investigation_candidate> candidates, size_t max_candidates) {
    for (auto & candidate : candidates) {
        candidate.posterior_priority = candidate.deterministic_priority;
        if (candidate.history_count < 3 || candidate.useful_history_weight < 0.0 ||
                candidate.unhelpful_history_weight < 0.0) continue;
        common_outcome_guidance_posterior posterior;
        if (common_outcome_guidance_make_posterior(
                candidate.useful_history_weight, candidate.unhelpful_history_weight,
                0.5, {1.0}, {1.0}, 2.0, 2.0, posterior)) {
            // Historical utility only orders candidate investigations. It is
            // not an evidentiary score and never changes claim status.
            candidate.posterior_priority = posterior.p_known * candidate.deterministic_priority;
        }
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto & a, const auto & b) {
        if (a.posterior_priority != b.posterior_priority) return a.posterior_priority > b.posterior_priority;
        if (a.deterministic_priority != b.deterministic_priority) return a.deterministic_priority > b.deterministic_priority;
        return a.id < b.id;
    });
    if (candidates.size() > max_candidates) candidates.resize(max_candidates);
    return candidates;
}
