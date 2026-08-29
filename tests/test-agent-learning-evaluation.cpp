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
    value.baseline_runtime_interventions = 3;
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
    assert(parsed.baseline_runtime_interventions == 3);

    common_learning_evaluation_fixture fixture{
        "learning://corpus/1",
        "identity:fnv1a64:0123456789abcdef",
        "train:qwen-small",
        "agent-tests:1",
    };
    common_learning_evaluation_report generated;
    common_learning_profile_evaluation_runner runner =
        [](const std::string & profile_id,
           const common_learning_evaluation_fixture &,
           common_learning_profile_evaluation_result & result,
           std::string &) {
            result.profile_id = profile_id;
            result.intended_behavior_passed = true;
            result.retention_passed = true;
            result.agent_regression_passed = true;
            result.evaluated_turns = 4;
            result.runtime_interventions = profile_id == "agent-baseline" ? 3 : 1;
            return true;
        };
    assert(common_learning_evaluate_candidate(
        fixture, "adapter:v1", "agent-baseline", "agent-candidate",
        "evaluation:generated", "2026-08-29T00:00:00Z", runner, generated, error));
    assert(generated.status == "passed");
    assert(generated.baseline_runtime_interventions == 3);
    assert(generated.runtime_interventions == 1);

    runner = [](const std::string & profile_id,
                const common_learning_evaluation_fixture &,
                common_learning_profile_evaluation_result & result,
                std::string &) {
        result.profile_id = profile_id;
        result.intended_behavior_passed = true;
        result.retention_passed = profile_id == "agent-baseline";
        result.agent_regression_passed = true;
        result.evaluated_turns = 4;
        return true;
    };
    assert(common_learning_evaluate_candidate(
        fixture, "adapter:v2", "agent-baseline", "agent-candidate",
        "evaluation:failed", "2026-08-29T00:00:00Z", runner, generated, error));
    assert(generated.status == "failed");

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
