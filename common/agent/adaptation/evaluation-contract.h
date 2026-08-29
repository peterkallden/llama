#pragma once

#include <cstddef>
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
    size_t runtime_interventions = 0;
    std::string status = "failed";
    std::string evaluated_at;
};

bool common_learning_validate_evaluation_report(
        const common_learning_evaluation_report & report,
        std::string & error);

double common_learning_runtime_intervention_rate(
        const common_learning_evaluation_report & report);

std::string common_learning_evaluation_report_to_json(
        const common_learning_evaluation_report & report);
bool common_learning_evaluation_report_from_json(
        const std::string & text,
        common_learning_evaluation_report & report,
        std::string & error);
