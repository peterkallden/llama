#pragma once

#include <cstddef>
#include <functional>
#include <string>

struct common_learning_evaluation_report {
    int schema_version = 1;
    std::string revision_id;
    std::string candidate_adapter_id;
    std::string baseline_profile_id;
    std::string candidate_profile_id;
    std::string corpus_revision_id;
    std::string corpus_bundle_hash;
    std::string base_training_fingerprint;
    std::string test_suite_revision;
    bool intended_behavior_passed = false;
    bool retention_passed = false;
    bool agent_regression_passed = false;
    size_t evaluated_turns = 0;
    size_t baseline_runtime_interventions = 0;
    size_t runtime_interventions = 0;
    std::string status = "failed";
    std::string evaluated_at;
};

bool common_learning_validate_evaluation_report(
        const common_learning_evaluation_report & report,
        std::string & error);

double common_learning_runtime_intervention_rate(
        const common_learning_evaluation_report & report);

// Both profiles receive this immutable fixture, so a comparison cannot
// accidentally mix corpus, tool/resource, base-model, or test-suite inputs.
struct common_learning_evaluation_fixture {
    std::string corpus_revision_id;
    std::string corpus_bundle_hash;
    std::string base_training_fingerprint;
    std::string test_suite_revision;
};

struct common_learning_profile_evaluation_result {
    std::string profile_id;
    bool intended_behavior_passed = false;
    bool retention_passed = false;
    bool agent_regression_passed = false;
    size_t evaluated_turns = 0;
    size_t runtime_interventions = 0;
};

using common_learning_profile_evaluation_runner = std::function<bool(
        const std::string & profile_id,
        const common_learning_evaluation_fixture & fixture,
        common_learning_profile_evaluation_result & result,
        std::string & error)>;

// Runs the same fixture against baseline and candidate and populates the
// durable report consumed by canary admission. A failed gate is a valid
// report with status=failed; false means evaluation could not run.
bool common_learning_evaluate_candidate(
        const common_learning_evaluation_fixture & fixture,
        const std::string & candidate_adapter_id,
        const std::string & baseline_profile_id,
        const std::string & candidate_profile_id,
        const std::string & revision_id,
        const std::string & evaluated_at,
        const common_learning_profile_evaluation_runner & runner,
        common_learning_evaluation_report & report,
        std::string & error);

std::string common_learning_evaluation_report_to_json(
        const common_learning_evaluation_report & report);
bool common_learning_evaluation_report_from_json(
        const std::string & text,
        common_learning_evaluation_report & report,
        std::string & error);
