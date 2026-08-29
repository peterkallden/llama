#include "agent/adaptation/evaluation-contract.h"

#include <cassert>

static common_learning_evaluation_report report() {
    common_learning_evaluation_report value;
    value.revision_id = "evaluation:1";
    value.candidate_adapter_id = "adapter:v1";
    value.baseline_profile_id = "agent-baseline";
    value.candidate_profile_id = "agent-candidate";
    value.corpus_revision_id = "learning://corpus/1";
    value.corpus_bundle_hash = "identity:fnv1a64:0123456789abcdef";
    value.base_training_fingerprint = "train:qwen-small";
    value.test_suite_revision = "agent-tests:1";
    value.intended_behavior_passed = true;
    value.retention_passed = true;
    value.agent_regression_passed = true;
    value.evaluated_turns = 10;
    value.runtime_interventions = 2;
    value.status = "passed";
    value.evaluated_at = "2026-08-29T00:00:00Z";
    return value;
}

int main() {
    std::string error;
    auto value = report();
    assert(common_learning_validate_evaluation_report(value, error));
    assert(common_learning_runtime_intervention_rate(value) == 0.2);
    const auto text = common_learning_evaluation_report_to_json(value);
    common_learning_evaluation_report parsed;
    assert(common_learning_evaluation_report_from_json(text, parsed, error));
    assert(parsed.candidate_adapter_id == value.candidate_adapter_id);

    value.retention_passed = false;
    assert(!common_learning_validate_evaluation_report(value, error));
    assert(error.find("status") != std::string::npos);
    value = report();
    value.runtime_interventions = 11;
    assert(!common_learning_validate_evaluation_report(value, error));
    assert(error.find("interventions") != std::string::npos);
    value = report();
    value.baseline_profile_id = value.candidate_profile_id;
    assert(!common_learning_validate_evaluation_report(value, error));
    assert(error.find("profiles") != std::string::npos);
    return 0;
}
