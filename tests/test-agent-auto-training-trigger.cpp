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
    assert(result.qualified_candidate_ids.size() == 6 &&
        result.qualified_candidate_ids.front() == "learning://candidate/1");
    candidates.push_back(candidates.back());
    assert(common_learning_auto_training_ready(candidates, group, policy, result, error));
    assert(result.qualified_examples == 6 && result.qualified_candidate_ids.size() == 6);
    auto aggregate = candidate("learning://candidate/aggregate", 100);
    aggregate.transaction_ids = {"learning://transaction/100", "learning://transaction/101",
        "learning://transaction/102", "learning://transaction/103", "learning://transaction/104",
        "learning://transaction/105"};
    auto aggregate_only = std::vector<common_training_candidate>{aggregate};
    assert(common_learning_auto_training_ready(aggregate_only, group, policy, result, error));
    assert(!result.ready && result.qualified_examples == 1 && result.distinct_transaction_ids == 6);
    assert(result.qualified_candidate_ids.size() == 1);
    auto mismatch = candidates;
    mismatch.front().tool_family = "memory";
    assert(common_learning_auto_training_ready(mismatch, group, policy, result, error));
    assert(result.qualified_examples == 5 && !result.ready);
    auto host_demo = candidate("learning://candidate/oracle-demo", 1000);
    host_demo.provenance = common_training_candidate_provenance::host_verified_demonstration;
    host_demo.cause = common_learning_cause::host_contract;
    host_demo.source = common_adaptation_evidence_source::user_taught_concept;
    host_demo.observed_occurrences = 0;
    host_demo.verified_recoveries = 0;
    host_demo.host_verifier_ref = "semi-oracle://placement_game";
    host_demo.host_verifier_revision = "minimax-v1";
    host_demo.host_proof_ref = "semi-oracle://evaluation/9";
    host_demo.evidence_ref = host_demo.host_proof_ref;
    host_demo.teaching_key = "semi-oracle:ruleset/revision/state/action";
    host_demo.behavior_key = "semi-oracle:placement-action";
    host_demo.learning_case_id = "learning://case/oracle-demo";
    auto nearline_candidates = candidates;
    nearline_candidates.push_back(host_demo);
    policy.min_qualified_examples = 7;
    assert(common_learning_auto_training_ready(nearline_candidates, group, policy, result, error));
    assert(!result.ready && result.qualified_examples == 6);
    policy.enabled = false;
    assert(common_learning_auto_training_ready(candidates, group, policy, result, error));
    assert(!result.ready && result.reason == "automatic training is disabled");
    return 0;
}
