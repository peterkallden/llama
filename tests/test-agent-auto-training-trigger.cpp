#include "agent/adaptation/auto-training-trigger.h"

#include <cassert>

static common_training_candidate candidate(const std::string & id, size_t ordinal) {
    common_training_candidate value;
    value.id = id;
    value.transaction_ids = {"learning://transaction/" + std::to_string(ordinal)};
    value.cause = common_learning_cause::model_behavior;
    value.hypothesis = "stable model behavior";
    value.approved_prompt = "redacted input";
    value.approved_target = "redacted target";
    value.observed_occurrences = 3;
    value.verified_recoveries = 2;
    value.confidence = 0.9f;
    value.redaction_policy_id = "stub:caller-asserted-v1";
    value.redaction_method = "caller_asserted";
    value.redaction_status = common_learning_redaction_status::caller_asserted;
    value.status = common_training_candidate_status::approved;
    value.learning_domain = "tool_use";
    value.tool_family = "diagnostics";
    value.provider_kind = ordinal % 2 == 0 ? "native" : "mcp";
    return value;
}

int main() {
    common_learning_training_group group;
    group.learning_domain = "tool_use";
    group.tool_family = "diagnostics";
    common_learning_auto_training_policy policy;
    policy.enabled = true;
    policy.min_qualified_examples = 6;
    common_learning_auto_training_result result;
    std::string error;
    auto candidates = std::vector<common_training_candidate>{
        candidate("learning://candidate/1", 1), candidate("learning://candidate/2", 2),
        candidate("learning://candidate/3", 3), candidate("learning://candidate/4", 4),
        candidate("learning://candidate/5", 5)};
    assert(common_learning_auto_training_ready(candidates, group, policy, result, error));
    assert(!result.ready && result.qualified_examples == 5);
    candidates.push_back(candidate("learning://candidate/6", 6));
    assert(common_learning_auto_training_ready(candidates, group, policy, result, error));
    assert(result.ready && result.qualified_examples == 6 && result.distinct_transaction_ids == 6);
    candidates.push_back(candidates.back());
    assert(common_learning_auto_training_ready(candidates, group, policy, result, error));
    assert(result.qualified_examples == 6);
    auto mismatch = candidates;
    mismatch.front().tool_family = "memory";
    assert(common_learning_auto_training_ready(mismatch, group, policy, result, error));
    assert(result.qualified_examples == 5 && !result.ready);
    policy.enabled = false;
    assert(common_learning_auto_training_ready(candidates, group, policy, result, error));
    assert(!result.ready && result.reason == "automatic training is disabled");
    return 0;
}
