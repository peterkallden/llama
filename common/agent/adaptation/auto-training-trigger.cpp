#include "agent/adaptation/auto-training-trigger.h"

#include <set>
#include <sstream>

namespace {

bool matches_group(const common_training_candidate & candidate,
        const common_learning_training_group & group) {
    return (group.learning_domain.empty() || candidate.learning_domain == group.learning_domain) &&
        (group.tool_family.empty() || candidate.tool_family == group.tool_family) &&
        (group.provider_kind.empty() || candidate.provider_kind == group.provider_kind);
}

std::string group_key(const common_learning_training_group & group) {
    std::ostringstream out;
    out << group.learning_domain << '\x1f' << group.tool_family << '\x1f'
        << group.provider_kind;
    return out.str();
}

} // namespace

bool common_learning_auto_training_ready(
        const std::vector<common_training_candidate> & candidates,
        const common_learning_training_group & group,
        const common_learning_auto_training_policy & policy,
        common_learning_auto_training_result & result,
        std::string & error) {
    error.clear();
    result = {};
    result.group_key = group_key(group);
    if (!policy.enabled) {
        result.reason = "automatic training is disabled";
        return true;
    }
    if (policy.min_qualified_examples == 0 || policy.max_scan == 0) {
        error = "automatic training policy bounds must be positive";
        return false;
    }

    common_training_candidate_policy candidate_policy;
    std::set<std::string> candidate_ids;
    std::set<std::string> transaction_ids;
    size_t scanned = 0;
    for (const auto & candidate : candidates) {
        if (scanned++ >= policy.max_scan) {
            error = "automatic training candidate scan exceeded bound";
            return false;
        }
        if (!matches_group(candidate, group)) continue;
        std::string qualification_error;
        if (!common_training_candidate_qualifies(candidate, candidate_policy, qualification_error)) continue;
        if (!candidate_ids.insert(candidate.id).second) continue;
        ++result.qualified_examples;
        for (const auto & transaction_id : candidate.transaction_ids) {
            if (!transaction_id.empty()) transaction_ids.insert(transaction_id);
        }
    }
    result.distinct_transaction_ids = transaction_ids.size();
    result.ready = result.qualified_examples >= policy.min_qualified_examples;
    result.reason = result.ready
        ? "qualified example threshold reached"
        : "waiting for qualified examples";
    return true;
}
